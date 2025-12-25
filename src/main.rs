use anyhow::{Context, Result};
use base64::{engine::general_purpose, Engine as _};
use clap::{Parser, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use mupdf::{Colorspace, Matrix};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::fs;
use tokio::sync::Semaphore;
use rayon::prelude::*;
use walkdir::WalkDir;
use regex::Regex;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Extract pages from a PDF to Images
    Extract {
        /// Input PDF file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for images
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// DPI for rasterization
        #[arg(long, default_value_t = 300)]
        dpi: u16,

        /// Limit number of pages to extract
        #[arg(long)]
        limit: Option<usize>,
    },
    // ... Transcribe stays same ...
    Transcribe {
        /// Input directory containing images
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for markdown files
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of concurrent requests
        #[arg(short, long, default_value_t = 50)]
        concurrency: usize,

        /// OpenRouter Model ID (e.g., google/gemini-flash-1.5, anthropic/claude-3.5-sonnet)
        /// Falls back to OPENROUTER_MODEL env var if not specified
        #[arg(long, env = "OPENROUTER_MODEL")]
        model: Option<String>,
        
        /// Limit number of images (for testing)
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run both pipeline steps: Extract then Transcribe
    Pipeline {
        /// Input PDF file
        #[arg(short, long)]
        input: PathBuf,

        /// Base output directory (will create 'images' and 'markdown' subdirs)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// DPI for rasterization
        #[arg(long, default_value_t = 300)]
        dpi: u16,

        /// Number of concurrent requests
        #[arg(short, long, default_value_t = 50)]
        concurrency: usize,

        /// OpenRouter Model ID (e.g., google/gemini-flash-1.5, anthropic/claude-3.5-sonnet)
        /// Falls back to OPENROUTER_MODEL env var if not specified
        #[arg(long, env = "OPENROUTER_MODEL")]
        model: Option<String>,

        /// Limit number of pages to process
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Combine markdown files into a single book with TOC
    Combine {
        /// Input directory containing markdown files
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path (default: input_dir/../{book_name}.md)
        #[arg(short, long)]
        output: Option<PathBuf>,
    }
}

// --- OpenRouter API Structs ---

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: Vec<ContentPart>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlData },
}

#[derive(Serialize)]
struct ImageUrlData {
    url: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Option<Vec<Choice>>,
    error: Option<OpenRouterError>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: Option<ResponseMessage>,
}

#[derive(Deserialize, Debug)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterError {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

// --- Phases ---

fn combine_book(input_dir: &Path, output_file: &Path) -> Result<()> {
    println!("Combining markdown files from {:?} into {:?}", input_dir, output_file);
    
    let mut files = Vec::new();
    // Use standard read_dir or WalkDir. max_depth(1) to avoid recursing if subdirs exist
    for entry in WalkDir::new(input_dir).max_depth(1) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("page_") && name.ends_with(".md") {
                     // Extract number for sorting: page_0001.md -> 1
                     // slice from 5 to len-3
                     let num_part = &name[5..name.len()-3];
                     if let Ok(num) = num_part.parse::<usize>() {
                         files.push((num, entry.path().to_path_buf()));
                     }
                }
            }
        }
    }
    
    // Sort by page number
    files.sort_by_key(|k| k.0);
    
    if files.is_empty() {
        println!("No page_*.md files found.");
        return Ok(());
    }

    // Validate completeness against images directory
    // Assumption: input_dir is .../markdown, images is .../images
    if let Some(parent) = input_dir.parent() {
        let images_dir = parent.join("images");
        if images_dir.exists() {
            let mut img_count = 0;
            for entry in WalkDir::new(&images_dir).max_depth(1) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.starts_with("page_") && name.ends_with(".png") {
                           img_count += 1;
                        }
                    }
                }
            }
            
            if files.len() != img_count {
                return Err(anyhow::anyhow!(
                    "Mismatch: Found {} markdown files but {} images. \
                    Ensure all pages have been transcribed before combining.",
                    files.len(), img_count
                ));
            }
            println!("Verified {} pages (matches {} source images)", files.len(), img_count);
        } else {
             println!("Warning: Could not find sibling 'images' directory to verify completeness.");
        }
    }
    
    let mut combined_content = String::new();
    let mut toc_lines = Vec::new();
    let mut seen_slugs = std::collections::HashMap::new();

    // Regex to match image links containing 'img/' or just general image links for cleanup
    // Python script used: r'!\[.*?\]\([^\)]*?img/[^\)]*\)'
    let img_regex = Regex::new(r"!\[.*?\]\([^\)]*?img/[^\)]*\)")?;

    // Header regex for TOC
    let header_regex = Regex::new(r"^(#+)\s+(.+)$")?;

    for (page_num, path) in files {
        // Read synchronously
        let content = std::fs::read_to_string(&path)?;
        
        // Strip images
        let clean_content = img_regex.replace_all(&content, "");
        let clean_content = clean_content.trim();
        
        combined_content.push_str(&format!("\n<a id='page_{}'></a>\n", page_num));
        
        // Scan headers for TOC, processing line by line
        for line in clean_content.lines() {
             if let Some(cap) = header_regex.captures(line) {
                 let level = cap[1].len();
                 let title = cap[2].trim();
                 
                 // Slug generation
                 let slug_base = title.to_lowercase()
                    .replace(' ', "-")
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-')
                    .collect::<String>();
                 
                 let slug = if let Some(count) = seen_slugs.get_mut(&slug_base) {
                     *count += 1;
                     format!("{}-{}", slug_base, *count)
                 } else {
                     seen_slugs.insert(slug_base.clone(), 0);
                     slug_base
                 };
                 
                 let indent = "  ".repeat(level.saturating_sub(1));
                 toc_lines.push(format!("{}- [{}]({}) *(Page {})*", indent, title, slug, page_num));
             }
        }
        
        combined_content.push_str(&clean_content);
        combined_content.push_str("\n\n---\n\n");
    }
    
    let book_name = output_file.file_stem().unwrap_or_default().to_string_lossy();
    let mut final_doc = format!("# {}\n\n## Table of Contents\n\n", book_name.replace('_', " "));
    final_doc.push_str(&toc_lines.join("\n"));
    final_doc.push_str("\n\n---\n\n");
    final_doc.push_str(&combined_content);
    
    std::fs::write(output_file, final_doc)?;
    println!("Created combined file: {:?}", output_file);
    
    Ok(())
}

fn extract_pdf(input: &Path, output_dir: &Path, dpi: u16, limit: Option<usize>) -> Result<()> {
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).context("Failed to create output dir")?;
    }

    // Open once to get count
    println!("Loading PDF with MuPDF to check page count...");
    let doc_check = mupdf::Document::open(input.to_str().context("Invalid path")?)
        .context("Failed to open PDF")?;
    let total_pages = doc_check.page_count().context("Failed to get page count")? as usize;
    
    let num_pages = limit.map(|l| l.min(total_pages)).unwrap_or(total_pages);
    
    println!("Extracting {} pages (of {}) from {:?} in parallel...", num_pages, total_pages, input);

    let pb = ProgressBar::new(num_pages as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    // Scale factor
    let scale = dpi as f32 / 72.0;

    // Process in parallel
    // Note: mupdf::Document might not be Sync. Safer to open a fresh handle per thread or per page.
    // Given file I/O overhead of opening is small vs rendering, we open per page or use thread local?
    // Let's just open inside the closure. It's robust.
    
    (0..num_pages).into_par_iter().for_each(|page_num| {
        let filename = format!("page_{:04}.png", page_num + 1);
        let output_path = output_dir.join(&filename);

        if output_path.exists() {
             pb.inc(1);
             return;
        }
        
        // Open document for this thread/iteration
        // We handle errors by printing to stderr to avoid panicking the whole parallel iterator easily, 
        // or we could use try_for_each but that stops on first error. 
        // Let's print error and continue others? Or panic? 
        // User probably wants to know if it failed.
        let process = || -> Result<()> {
            let doc = mupdf::Document::open(input.to_str().unwrap())?;
            let page = doc.load_page(page_num as i32)?;
            let matrix = Matrix::new_scale(scale, scale);
            let pixmap = page.to_pixmap(&matrix, &Colorspace::device_rgb(), false, true)?;
            pixmap.save_as(&output_path.to_string_lossy(), mupdf::ImageFormat::PNG)?;
            Ok(())
        };

        if let Err(e) = process() {
            eprintln!("Error processing page {}: {}", page_num + 1, e);
        }
        
        pb.inc(1);
    });
    
    pb.finish_with_message("Extraction complete");
    Ok(())
}

async fn transcribe_images(
    input_dir: PathBuf,
    output_dir: PathBuf,
    concurrency: usize,
    model: String,
    api_key: String,
    limit: Option<usize>,
) -> Result<()> {
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir).await?;
    }

    let client = Client::new();
    let semaphore = Arc::new(Semaphore::new(concurrency));

    let mut paths = Vec::new();
    for entry in WalkDir::new(&input_dir).sort_by_file_name() {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".png") && !name.starts_with("._") {
                    paths.push(path.to_path_buf());
                }
            }
        }
    }

    if let Some(l) = limit {
        paths.truncate(l);
    }

    println!("Found {} images to transcribe", paths.len());
    let m = MultiProgress::new();
    let pb = m.add(ProgressBar::new(paths.len() as u64));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));

    let mut tasks = Vec::new();

    for path in paths {
        let client = client.clone();
        let api_key = api_key.clone();
        let output_dir = output_dir.clone();
        let model = model.clone();
        let permit = semaphore.clone().acquire_owned().await?;
        let pb = pb.clone();

        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
            let output_filename = format!("{}.md", file_stem);
            let final_output = output_dir.join(&output_filename);

            if final_output.exists() {
                pb.inc(1);
                return Ok(());
            }

            pb.set_message(format!("Proc: {}", file_stem));

            // Atomic write prep
            let mut tmp_file = NamedTempFile::new_in(&output_dir)?;
            
            // Process
            let image_data = fs::read(&path).await?;
            let b64_data = general_purpose::STANDARD.encode(&image_data);

            // OpenRouter uses OpenAI-compatible format with data URLs for images
            let request_body = ChatCompletionRequest {
                model: model.clone(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: vec![
                        ContentPart::Text {
                            text: "Transcribe this page from Inside Macintosh. Output strictly formatted Markdown. Use headers, lists, and code blocks where appropriate. IMPORTANT: Transcribe ALL legible text, including page numbers, headers, footers, and captions. Do NOT wrap the entire output in a markdown block.".to_string(),
                        },
                        ContentPart::ImageUrl {
                            image_url: ImageUrlData {
                                url: format!("data:image/png;base64,{}", b64_data),
                            },
                        },
                    ],
                }],
            };

            let url = "https://openrouter.ai/api/v1/chat/completions";
            
            let resp = client
                .post(url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .timeout(Duration::from_secs(120))
                .send()
                .await?;

            if !resp.status().is_success() {
                let txt = resp.text().await?;
                return Err(anyhow::anyhow!("API Error: {}", txt));
            }

            let result: ChatCompletionResponse = resp.json().await?;
            if let Some(err) = result.error {
                let type_str = err.error_type.as_deref().unwrap_or("unknown");
                return Err(anyhow::anyhow!("API Error ({}): {}", type_str, err.message));
            }

            let mut text = result.choices
                .and_then(|c| c.into_iter().next())
                .and_then(|c| c.message)
                .and_then(|m| m.content)
                .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

            // Clean up code blocks if the model wrapped the output
            if text.trim_start().starts_with("```") {
                // Find first newline
                if let Some(newline_pos) = text.find('\n') {
                    text = text[newline_pos + 1..].to_string();
                }
                // Strip trailing fence
                if let Some(last_fence) = text.rfind("```") {
                    text = text[..last_fence].trim_end().to_string();
                }
            }

            // Write to temp
            use std::io::Write;
            tmp_file.write_all(text.as_bytes())?;
            
            // Atomic rename
            tmp_file.persist(&final_output)?;

            pb.inc(1);
            pb.set_message("Done");
            Ok::<(), anyhow::Error>(())
        }));
    }

    let results = futures::future::join_all(tasks).await;
    pb.finish_with_message("Transcription complete");
    
    // Check for errors
    let mut error_count = 0;
    for result in results {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("Task error: {}", e);
                error_count += 1;
            }
            Err(e) => {
                eprintln!("Join error: {}", e);
                error_count += 1;
            }
        }
    }
    
    if error_count > 0 {
        eprintln!("{} tasks failed", error_count);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file (ignore if not present)
    let _ = dotenvy::dotenv();
    
    let args = Args::parse();
    
    match args.command {
        Commands::Extract { input, output, dpi, limit } => {
            let output = match output {
                Some(p) => p,
                None => {
                    let book_name = input.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown_book");
                    PathBuf::from("out").join(book_name).join("images")
                }
            };
            extract_pdf(&input, &output, dpi, limit)?;
        }
        Commands::Transcribe { input, output, concurrency, model, limit } => {
            let api_key = env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY must be set")?;
            let model = model.context("Model must be specified via --model or OPENROUTER_MODEL env var")?;
            
            let output = match output {
                Some(p) => p,
                None => {
                    // Try to deduce structure. If input is .../images, output .../markdown
                     if input.ends_with("images") {
                        input.parent().unwrap_or(&input).join("markdown")
                    } else {
                        // Fallback: out/{input_dir_name}/markdown
                        let dir_name = input.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown_batch");
                        PathBuf::from("out").join(dir_name).join("markdown")
                    }
                }
            };
            
            transcribe_images(input, output, concurrency, model, api_key, limit).await?;
        }
        Commands::Combine { input, output } => {
             let output = match output {
                Some(p) => p,
                None => {
                     // Default: out/book.md (sibling of markdown dir)
                     // Input: out/book/markdown -> parent is out/book -> join book.md
                     let parent = input.parent().unwrap_or(&input);
                     let book_name = parent.file_name().unwrap_or_default();
                     parent.join(format!("{}.md", book_name.to_string_lossy()))
                }
            };
            combine_book(&input, &output)?;
        }
        Commands::Pipeline { input, output, dpi, concurrency, model, limit } => {
            let inputs: Vec<PathBuf> = if input.is_dir() {
                let mut pdfs = Vec::new();
                let mut entries = fs::read_dir(&input).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension() {
                            if ext.eq_ignore_ascii_case("pdf") {
                                pdfs.push(path);
                            }
                        }
                    }
                }
                pdfs.sort();
                if pdfs.is_empty() {
                    println!("No PDF files found in directory: {:?}", input);
                } else {
                    println!("Found {} PDF files in directory: {:?}", pdfs.len(), input);
                }
                pdfs
            } else {
                vec![input.clone()]
            };

            for (i, pdf_path) in inputs.iter().enumerate() {
                let book_name = pdf_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown_book");

                println!("\n=== Processing Book {}/{}: {} ===\n", i + 1, inputs.len(), book_name);

                let output_base = if input.is_dir() {
                    // If input was a directory, output arg is the parent dir for all books
                    match &output {
                        Some(p) => p.join(book_name),
                        None => PathBuf::from("out").join(book_name) // Default structure
                    }
                } else {
                    // Single file mode: match existing behavior
                    match &output {
                        Some(p) => p.clone(),
                        None => PathBuf::from("out").join(book_name)
                    }
                };

                let images_dir = output_base.join("images");
                let markdown_dir = output_base.join("markdown");

                println!("--- Phase 1: Extract ---");
                println!("Output directory: {:?}", output_base);
                
                if let Err(e) = extract_pdf(pdf_path, &images_dir, dpi, limit) {
                    eprintln!("Error extracting {}: {}", book_name, e);
                    continue; // Skip to next book on failure
                }

                println!("--- Phase 2: Transcribe ---");
                let api_key = env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY must be set")?;
                let model_str = model.clone().context("Model must be specified via --model or OPENROUTER_MODEL env var")?;
                
                if let Err(e) = transcribe_images(images_dir, markdown_dir.clone(), concurrency, model_str, api_key, limit).await {
                    eprintln!("Error transcribing {}: {}", book_name, e);
                    continue;
                }
                
                println!("--- Phase 3: Combine ---");
                
                let combined_file = if input.is_dir() {
                    let root = match &output {
                        Some(p) => p.clone(),
                        None => PathBuf::from("out")
                    };
                    let combined_dir = root.join("combined");
                    if !combined_dir.exists() {
                         std::fs::create_dir_all(&combined_dir).context("Failed to create combined output dir")?;
                    }
                    combined_dir.join(format!("{}.md", book_name))
                } else {
                    output_base.join(format!("{}.md", book_name))
                };
                
                 if let Err(e) = combine_book(&markdown_dir, &combined_file) {
                     eprintln!("Warning: Failed to combine files for {}: {}", book_name, e);
                 }
                 
                 println!("\nCompleted pipeline for: {}\n", book_name);
            }
        }
    }

    Ok(())
}
