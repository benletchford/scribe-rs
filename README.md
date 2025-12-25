# scribe-rs

**scribe-rs** is a high-performance, oxidized tool for transcribing books and documents into clean Markdown using Large Language Models (LLMs). It automates the entire pipeline: converting PDFs to images, processing them via OpenRouter (Gemini, Claude, GPT-4), and assembling the results into a single, cohesive book with a Table of Contents.

## Features

- **🚀 High Performance**: Built in Rust with async I/O (`tokio`) and parallel processing (`rayon`).
- **📄 PDF to Image Extraction**: Uses `mupdf` for accurate, high-DPI rasterization.
- **🤖 LLM Transcription**: Concurrent batch processing via OpenRouter API (supports Gemini Flash, Claude 3.5 Sonnet, etc.).
- **📚 Smart Combination**: Merges page-level markdown into a single book, generating a Table of Contents and cleaning up artifacts.
- **🔄 Idempotent & Resumable**: Skips already processed files, allowing you to stop and resume large jobs without losing progress.
- **⚡ Zero-Dependency**: Statically links MuPDF for easy deployment (via `mupdf` crate).

## Installation

### Prerequisites
- [Rust](https://rustup.rs/) (latest stable)
- A valid [OpenRouter](https://openrouter.ai/) API key.

### Build
```bash
git clone https://github.com/benletchford/scribe-rs.git
cd scribe-rs
cargo build --release
```
The binary will be located at `target/release/scribe`.

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/google/gemini-3-flash-preview
```

Alternatively, you can pass the API key and model via command-line arguments or environment variables.

## Usage

**scribe-rs** operates with subcommands. You can run the full pipeline or individual steps.

> **Note**: All operations are idempotent. If an output file (image or markdown) already exists, it is skipped. This allows you to safely interrupt and resume long-running jobs.

### 1. Full Pipeline (Recommended)
Run extraction, transcription, and combination in one go:
```bash
cargo run --release -- pipeline --input "path/to/book.pdf" --output "out/my_book" --model "google/gemini-flash-1.5"
```

### 2. Bulk Processing
You can also pass a directory of PDFs to process them consecutively:
```bash
cargo run --release -- pipeline --input "path/to/pdf_folder" --output "out"
```
This will process every PDF in the folder and save the final combined Markdown files into `out/combined/`.

### 3. Manual Steps

**Step 1: Extract Images**
Convert PDF pages to PNGs.
```bash
cargo run --release -- extract --input "book.pdf" --output "out/images" --dpi 300
```

**Step 2: Transcribe Images**
Process images into Markdown files.
```bash
cargo run --release -- transcribe --input "out/images" --output "out/markdown" --concurrency 50
```

**Step 3: Combine**
Merge markdown files into a single book.
```bash
cargo run --release -- combine --input "out/markdown" --output "final_book.md"
```

## CLI Options

| Global / Common Flags | Description |
|-----------------------|-------------|
| `--input, -i` | Input file (PDF) or directory (Images/Markdown). |
| `--output, -o` | Output destination. |
| `--model` | OpenRouter model ID (overrides env var). |
| `--concurrency, -c` | Number of concurrent API requests (Default: 50). |
| `--dpi` | Rasterization quality for PDF extraction (Default: 300). |
| `--limit` | Limit the number of pages to process (useful for testing). |

## License

[MIT](LICENSE)
