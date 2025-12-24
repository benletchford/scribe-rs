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
use walkdir::WalkDir;

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
        output: PathBuf,

        /// DPI for rasterization
        #[arg(long, default_value_t = 300)]
        dpi: u16,
    },
    /// Transcribe images to Markdown
    Transcribe {
        /// Input directory containing images
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for markdown files
        #[arg(short, long)]
        output: PathBuf,

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
        output: PathBuf,

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

fn extract_pdf(input: &Path, output_dir: &Path, dpi: u16) -> Result<()> {
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).context("Failed to create output dir")?;
    }

    println!("Loading PDF with MuPDF...");
    let document = mupdf::Document::open(input.to_str().context("Invalid path")?)
        .context("Failed to open PDF")?;

    let total_pages = document.page_count().context("Failed to get page count")?;
    println!("Extracting {} pages from {:?}...", total_pages, input);

    let pb = ProgressBar::new(total_pages as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    // Scale factor for DPI (72 is the default PDF DPI)
    let scale = dpi as f32 / 72.0;
    let matrix = Matrix::new_scale(scale, scale);

    for page_num in 0..total_pages {
        let filename = format!("page_{:04}.png", page_num + 1);
        let output_path = output_dir.join(&filename);

        if output_path.exists() {
            // Idempotency: Skip existing files
            pb.inc(1);
            continue;
        }

        let page = document.load_page(page_num).context("Failed to load page")?;
        
        // Render page to pixmap with the specified DPI
        let pixmap = page.to_pixmap(&matrix, &Colorspace::device_rgb(), false, true)
            .context("Failed to render page")?;

        // Save as PNG
        pixmap.save_as(&output_path.to_string_lossy(), mupdf::ImageFormat::PNG)
            .context("Failed to save PNG")?;

        pb.inc(1);
    }
    
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
                            text: "Transcribe this page from Inside Macintosh. Output strictly formatted Markdown. Use headers, lists, and code blocks where appropriate. Do NOT wrap the entire output in a markdown block.".to_string(),
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
                return Err(anyhow::anyhow!("API Error: {}", err.message));
            }

            let text = result.choices
                .and_then(|c| c.into_iter().next())
                .and_then(|c| c.message)
                .and_then(|m| m.content)
                .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

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
        Commands::Extract { input, output, dpi } => {
            extract_pdf(&input, &output, dpi)?;
        }
        Commands::Transcribe { input, output, concurrency, model, limit } => {
            let api_key = env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY must be set")?;
            let model = model.context("Model must be specified via --model or OPENROUTER_MODEL env var")?;
            transcribe_images(input, output, concurrency, model, api_key, limit).await?;
        }
        Commands::Pipeline { input, output, dpi, concurrency, model } => {
            let images_dir = output.join("images");
            let markdown_dir = output.join("markdown");

            println!("--- Phase 1: Extract ---");
            extract_pdf(&input, &images_dir, dpi)?;

            println!("--- Phase 2: Transcribe ---");
            let api_key = env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY must be set")?;
            let model = model.context("Model must be specified via --model or OPENROUTER_MODEL env var")?;
            transcribe_images(images_dir, markdown_dir, concurrency, model, api_key, None).await?;
        }
    }

    Ok(())
}
