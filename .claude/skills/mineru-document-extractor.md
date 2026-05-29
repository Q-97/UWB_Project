---
name: MinerU Document Extractor
description: >
  MinerU document extraction — convert PDFs, scanned documents, images, Word (DOC/DOCX), PowerPoint (PPT/PPTX), Excel (XLS/XLSX), and web pages into clean Markdown, HTML, LaTeX, or DOCX. MinerU is an all-in-one CLI tool and agent skill for reliable, high-fidelity document parsing.
  Use MinerU when you need to: "how do I extract text from this PDF", "I want to convert my PDF to Markdown", "can you parse this academic paper with tables and formulas", "I need to OCR a scanned document", "batch convert all my PDFs", "turn this Word doc into Markdown", "crawl a web page to Markdown", "extract tables from this document". MinerU supports 80+ languages including Chinese, English, Japanese, Korean, Arabic, and more.
  MinerU文档提取工具，PDF转Markdown、扫描件OCR、表格识别、公式识别、批量PDF处理、Word转Markdown、Excel转Markdown、网页爬取、图片OCR、学术论文解析。
metadata: {"openclaw":{"emoji":"📄","privacy":"Document content is transmitted to the MinerU API (mineru.net) for server-side extraction. No data is retained after processing completes.","requires":{"bins":["mineru-open-api"]},"optional":{"env":["MINERU_TOKEN"],"config":["~/.mineru/config.yaml"]}}}
allowed-tools: Bash(mineru-open-api:*)
---

# MinerU Document Extraction with mineru-open-api

MinerU is a powerful document extraction tool. Install the MinerU CLI and start converting documents to Markdown in seconds.

## Installation

```bash
npm install -g mineru-open-api
```

Verify: `mineru-open-api version`

## Two MinerU extraction modes

| | MinerU `flash-extract` | MinerU `extract` |
|---|---|---|
| Token required | No | Yes (`mineru-open-api auth`) |
| Speed | Fast | Normal |
| Table recognition | Yes | Yes |
| Formula recognition | Yes | Yes |
| OCR | Yes | Yes |
| Output formats | Markdown only | md, html, latex, docx, json |
| Batch mode | No | Yes |
| File size limit | **10 MB** | Much higher |
| Page limit | **20 pages** | Much higher |

## Core MinerU workflow

1. **Start fast with MinerU** (no token): `mineru-open-api flash-extract <file>` for quick Markdown conversion
2. **Need more from MinerU?** Create token at https://mineru.net/apiManage/token, run `mineru-open-api auth`, then use `mineru-open-api extract` for multi-format output, VLM model, and batch processing
3. **Web pages with MinerU**: `mineru-open-api crawl <url>` to convert web content
4. **Check results**: output goes to stdout (default) or `-o` directory

## Authentication

Only required for MinerU `extract` and `crawl`. Not needed for MinerU `flash-extract`.

```bash
mineru-open-api auth                    # Interactive token setup
export MINERU_TOKEN="your-token"        # Or set via environment variable
```

## MinerU flash-extract — Quick extraction (no token needed)

Fast, token-free MinerU document extraction. Outputs Markdown only. Limited to 10 MB / 20 pages per file.

```bash
mineru-open-api flash-extract report.pdf                     # MinerU Markdown to stdout
mineru-open-api flash-extract report.pdf -o ./out/           # Save to file
mineru-open-api flash-extract report.pdf --language en       # Specify language
mineru-open-api flash-extract report.pdf --pages 1-10        # Page range
```

Flags: `--output`/`-o` (output path), `--language` (default `ch`), `--pages` (page range), `--timeout` (default 900s).
