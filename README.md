# DocGuard AI

Multimodal document intelligence and financial risk reasoning system — a research-oriented platform for document understanding, identity verification, forgery detection, and financial risk analysis.

## Project Objective

DocGuard AI aims to build an end-to-end system that:

- Understands and extracts structured information from multimodal documents (text, images, tables, forms)
- Enhances OCR quality on degraded or noisy inputs
- Verifies identity using consented and synthetic data
- Detects image and document forgery
- Performs financial risk reasoning over extracted evidence
- Supports reproducible research through experiment tracking, evaluation benchmarks, and failure analysis

This repository is structured for iterative research: baseline experiments first, then supervised fine-tuning, parameter-efficient adaptation, preference alignment, RAG, and robustness evaluation — without coupling implementation phases prematurely.

## Planned Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           React Frontend                                │
│              (document upload, review, experiment dashboards)           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         FastAPI Backend                                 │
│         (inference orchestration, experiment APIs, auth hooks)          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Preprocessing │       │   Inference     │       │  Experiment     │
│  OCR           │       │   Pipeline      │       │  Tracking       │
│  Doc Understanding      │   (VLM/LLM)     │       │  (MLflow/W&B)   │
└───────┬───────┘       └────────┬────────┘       └─────────────────┘
        │                        │
        ▼                        ▼
┌───────────────┐       ┌─────────────────┐
│  RAG Index    │       │  Task Modules   │
│  (retrieval)  │       │  forgery, ID,   │
└───────────────┘       │  financial risk │
                        └─────────────────┘
```

**Data flow (planned):** raw documents → preprocessing/OCR → document understanding → task-specific modules (forgery, identity, financial reasoning) → RAG-augmented LLM/VLM inference → structured outputs → evaluation and failure analysis.

**Training flow (planned):** curated datasets → SFT → LoRA/QLoRA adapters → preference alignment → checkpoint registry under `models/` → deployment via inference pipeline.

## Planned Research Questions

1. **Document understanding:** How accurately can open-weight VLMs extract structured fields from heterogeneous document layouts under real-world noise?
2. **OCR enhancement:** Can lightweight preprocessing and model fine-tuning recover usable text from low-quality scans without full re-OCR pipelines?
3. **Identity verification:** What is the trade-off between synthetic/consented training data and generalization to held-out identity document types?
4. **Forgery detection:** Which multimodal signals (pixel artifacts, metadata, semantic inconsistencies) are most reliable for document and image forgery?
5. **Financial risk reasoning:** Can RAG-augmented LLMs produce auditable risk assessments grounded in extracted document evidence?
6. **Efficient adaptation:** How do full SFT, LoRA, and QLoRA compare on accuracy, memory, and inference latency for domain-specific tasks?
7. **Preference alignment:** Does alignment (e.g., DPO/RLHF-style methods) improve factual grounding and reduce hallucination in risk summaries?
8. **Robustness:** How do models degrade under noisy, adversarial, and out-of-distribution inputs — and where do failures cluster?

## Planned Experiments

| Experiment | Directory | Focus |
|------------|-----------|-------|
| 01 — Baseline | `experiments/experiment_01_baseline/` | Zero-shot / off-the-shelf model baselines on core tasks |
| 02 — SFT | `experiments/experiment_02_sft/` | Supervised fine-tuning on curated task datasets |
| 03 — LoRA | `experiments/experiment_03_lora/` | PEFT/LoRA/QLoRA adapter training and comparison |
| 04 — Preference Alignment | `experiments/experiment_04_preference_alignment/` | Alignment methods for grounded outputs |
| 05 — RAG | `experiments/experiment_05_rag/` | Retrieval-augmented generation for financial reasoning |
| 06 — Robustness | `experiments/experiment_06_robustness/` | Noisy, adversarial, and OOD evaluation |

Each experiment directory will hold configs, run logs, and pointers to checkpoints — not production code.

## Planned Evaluation Strategy

- **Datasets:** held-out splits under `evaluation/datasets/`; task-specific benchmarks under `evaluation/benchmarks/`
- **Metrics:** accuracy, F1, calibration, latency, peak memory, and inference cost under `evaluation/metrics/`
- **Noisy evaluation:** degraded scans, compression artifacts, rotation/skew — `evaluation/noisy/` and `data/noisy/`
- **Adversarial evaluation:** targeted perturbations and forgery attempts — `evaluation/adversarial/` and `data/adversarial/`
- **Failure analysis:** error clustering, confusion patterns, and qualitative review — `evaluation/failure_analysis/`
- **Reproducibility:** fixed seeds, versioned configs in `configs/`, experiment artifacts in `experiments/`

## Planned Technology Stack

| Layer | Technologies (planned, not yet installed) |
|-------|-------------------------------------------|
| Models | Open-weight LLMs/VLMs, Hugging Face Transformers, PEFT, TRL |
| Training | PyTorch, LoRA/QLoRA, SFT, preference alignment |
| Document/OCR | Tesseract or equivalent, OpenCV, custom preprocessing |
| RAG | Vector store, embedding models, chunking pipeline |
| Backend | FastAPI, async inference serving |
| Frontend | React, component library TBD |
| Tracking | MLflow and/or Weights & Biases |
| Evaluation | Custom benchmarks, pytest, notebook analysis |

**Current phase:** Phase 2 — document understanding + VLM baseline (Phase 1 OCR pipeline included).

## Phase 1 — Document Ingestion, Preprocessing, and OCR

Phase 1 provides a CPU-friendly pipeline for loading supported documents, preparing page images for OCR, and returning structured text with bounding boxes and confidence scores.

### Supported formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Rendered page-by-page via PyMuPDF |
| PNG | `.png` | Single-page image input |
| JPEG | `.jpg`, `.jpeg` | Single-page image input |

### Module layout

```
src/
├── ingestion/      # Path validation, metadata extraction, page loading
├── preprocessing/  # Configurable OCR-oriented image preprocessing
└── ocr/            # Tesseract OCR engine and end-to-end pipeline
```

### Document ingestion

`src/ingestion/` accepts a local file path and:

- verifies the file exists
- rejects unsupported extensions cleanly
- validates PDF/image content
- generates a unique `document_id` (UUID4)
- records filename, file type, file size, and page count

### Preprocessing pipeline

`src/preprocessing/` applies OCR-oriented steps through `PreprocessingConfig`:

| Step | Default | Purpose |
|------|---------|---------|
| Deskew | on | Correct small rotational skew when detected |
| Grayscale | on | Normalize color scans for OCR |
| Resize | on | Downscale very large pages (`max_dimension=3000`) |
| Contrast enhancement | on | CLAHE-based local contrast boost |
| Denoise | on | Remove speckle while preserving edges |
| Threshold | off | Adaptive or Otsu binarization for low-contrast scans |

Steps are configurable and should be enabled selectively. Thresholding is off by default because it can harm photos and already-clean scans.

### OCR architecture

```
document path
    → ingestion (validate + metadata + page images)
    → preprocessing (per-page, configurable)
    → TesseractOCREngine (pytesseract)
    → structured OCRPipelineResult
```

`TesseractOCREngine`:

- accepts a preprocessed page image
- uses Tesseract word-level detection
- groups words into line-level blocks
- preserves page numbers
- orders blocks by Tesseract block/paragraph/line indices
- normalizes confidence to `0.0–1.0`

### Expected output schema

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "pages": [
    {
      "page_number": 1,
      "text": "Line one\nLine two",
      "blocks": [
        {
          "text": "Line one",
          "bbox": [left, top, right, bottom],
          "confidence": 0.93
        }
      ]
    }
  ]
}
```

Metadata-only ingestion output from `scripts/test_ocr.py --metadata-only`:

```json
{
  "document_id": "...",
  "filename": "document.png",
  "file_type": "png",
  "file_size_bytes": 12345,
  "page_count": 1,
  "source_path": "..."
}
```

### Phase 1 dependencies

Install Python packages (inside the project virtual environment):

```powershell
.\.venv\Scripts\activate
pip install opencv-python pytesseract pymupdf pytest
```

Install the Tesseract OCR binary separately:

```powershell
winget install UB-Mannheim.TesseractOCR
```

Ensure `tesseract` is available on `PATH` before running OCR.

### Run tests

```powershell
.\.venv\Scripts\activate
pytest tests/test_preprocessing.py tests/test_ocr.py -v
```

Ingestion and schema tests run without OCR installed. OpenCV-dependent preprocessing tests and live OCR tests are skipped automatically when dependencies are missing.

### Run the CLI

```powershell
.\.venv\Scripts\python.exe scripts/test_ocr.py path\to\document.png
.\.venv\Scripts\python.exe scripts/test_ocr.py path\to\document.pdf --pretty
.\.venv\Scripts\python.exe scripts/test_ocr.py path\to\document.png --metadata-only --pretty
```

Optional preprocessing flags: `--no-grayscale`, `--no-deskew`, `--threshold`.

## Phase 2 — Document Understanding + VLM Baseline

Phase 2 adds a modular vision-language model (VLM) layer on top of Phase 1 OCR. The pipeline passes the original page image, OCR text, OCR bounding boxes, and a task-specific prompt to a configurable VLM, then parses structured JSON output.

### Architecture

```
document/image
    ↓
Phase 1: ingestion → preprocessing → OCR
    ↓
Document (image + OCR text + OCR blocks + metadata)
    ↓
multimodal prompt construction
    ↓
VLM inference (configurable model)
    ↓
JSON parsing + validation
    ↓
structured document understanding output
```

### Module layout

```
src/document_understanding/
├── config.py          # VLMConfig, TaskType, env-based model settings
├── vlm_config.py      # BaselineVLMConfig for the image-only VLM baseline
├── vlm_inference.py   # Baseline image + prompt inference
├── prompts.py         # Modular prompts for supported tasks
├── models.py          # Document, DocumentPage, DocumentUnderstandingResult
├── vlm_interface.py   # Abstract VLMModel interface
├── vlm_loader.py      # Hugging Face VLM implementation + factory
├── inference.py       # VLM inference orchestration
├── parser.py          # JSON extraction and validation
└── pipeline.py        # OCR + VLM end-to-end pipeline
```

### VLM abstraction

The rest of the project depends on `VLMModel`, not a specific checkpoint:

- `load()` — load weights and processor on demand
- `generate(image, prompt)` — run multimodal inference
- `unload()` — release memory

The default implementation uses Hugging Face Transformers (`AutoModelForImageTextToText`). Swap models by changing `VLM_MODEL_ID` without changing the pipeline.

### Model configuration

Set environment variables in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLM_MODEL_ID` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Hugging Face model id |
| `VLM_DEVICE` | `cpu` | Inference device |
| `VLM_MAX_NEW_TOKENS` | `512` | Generation limit |
| `VLM_DTYPE` | `float32` | Weight dtype (`float32` recommended on CPU) |
| `VLM_LOCAL_FILES_ONLY` | `false` | Require locally cached weights |

**Default model:** `HuggingFaceTB/SmolVLM-256M-Instruct` (~256M parameters). This is the smallest SmolVLM instruct checkpoint and is suitable for CPU-only experimentation. Larger alternatives such as `HuggingFaceTB/SmolVLM-500M-Instruct` or `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` can be configured later, but expect slower CPU inference.

Models are **not downloaded automatically** during setup. Download a model explicitly before running the CLI, for example:

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoProcessor, AutoModelForImageTextToText; AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct'); AutoModelForImageTextToText.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')"
```

### CPU limitation

This development setup is CPU-only:

- Do **not** install CUDA or `bitsandbytes`
- Use `float32` on CPU for the default model
- Expect roughly 10–60 seconds per page depending on image size and `max_new_tokens`
- Do not use 7B+ VLMs locally in this phase

If no usable local model is configured or cached, the CLI fails with an actionable error message.

### Supported tasks

| Task | CLI value | Purpose |
|------|-----------|---------|
| General understanding | `document_understanding` | Type, fields, tables, summary |
| Key-value extraction | `key_value_extraction` | Structured field extraction |
| Classification | `document_classification` | Document type labeling |
| Table understanding | `table_understanding` | Table extraction |
| OCR correction | `ocr_correction` | Correct OCR against the image |

### OCR + VLM integration

Phase 2 builds a `Document` object from Phase 1 output:

```
Document
├── document_id
├── pages[]
│   ├── image
│   ├── ocr_text
│   └── ocr_blocks[]
└── metadata
```

The VLM receives the page image plus OCR text and bounding boxes in the prompt. Confidence in structured output is set to `null` unless the model provides a calibrated score.

### Expected structured output

```json
{
  "document_type": "invoice",
  "fields": {
    "invoice_number": "INV-001",
    "date": "2026-01-01",
    "customer_name": "Example Customer",
    "company": "Example Corp",
    "total_amount": "20.00"
  },
  "tables": [],
  "summary": "Invoice from Example Corp"
}
```

`confidence` is omitted from the default prompt schema unless the model provides a calibrated score. When JSON parsing fails, `parse_errors` and `raw_response` are preserved for analysis. Schema violations are recorded separately in `schema_errors`.

Parsed results also expose three distinct signals:

- `parse_success` — valid JSON was extracted and parsed
- `schema_valid` — the model output already matched the nested schema (no structural repair)
- `schema_normalized` — known flat fields were relocated into `fields` for evaluation compatibility (not model improvement)

### Phase 2 dependencies

Phase 1 dependencies plus (installed separately on CPU):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
```

### Run Phase 2 tests

```powershell
.\.venv\Scripts\activate
pytest tests/test_document_understanding.py tests/test_prompts.py tests/test_document_pipeline.py -v
```

**Unit tests** use a mock VLM and do not download models.

**Integration tests** (`@pytest.mark.integration`) run only when a cached VLM is available locally:

```powershell
pytest tests/test_document_pipeline.py -m integration -v
```

### Run the VLM CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm.py --image path\to\document.png --task document_understanding --pretty
```

Examples:

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm.py --image path\to\invoice.pdf --task key_value_extraction --pretty
.\.venv\Scripts\python.exe scripts\run_vlm.py --image path\to\scan.png --task ocr_correction --no-deskew --pretty
```

### Evaluation schema

Benchmark utilities live under `evaluation/benchmarks/document_understanding/`:

- `schema.py` — `EvaluationResult`, `BenchmarkSample`, `BenchmarkRunSummary`
- `metrics.py` — field extraction, classification, OCR correction, JSON parse success, schema compliance, schema normalization rate, and field extraction count helpers

Evaluation results can store `model`, `dataset`, `task`, `metric`, `score`, `latency`, `memory`, and `timestamp`. Scores are computed only from provided predictions and ground truth; no benchmark scores are fabricated.

### Baseline VLM Failure Analysis

**Observed problem:** During the first end-to-end DocGuard inference test (ingestion → preprocessing → OCR → SmolVLM-256M-Instruct), the VLM completed inference successfully (~110s on CPU) but produced OCR-like textbox output instead of the requested document-understanding JSON. JSON parsing failed with `Malformed JSON in model output`, leaving `document_type`, `fields`, `tables`, and `summary` empty/null. OCR also showed realistic errors (for example `Al Document Processing` instead of `AI Document Processing`, `%20,000` instead of `₹20,000`).

**Likely hypothesis:** The small instruction-tuned VLM needs stronger constrained prompting and robust output parsing for structured document extraction. Without explicit JSON-only instructions, schema examples, and OCR-as-context framing, the model tends to echo OCR layout rather than emit a single valid JSON object.

**Experiment (Phase 2.1):** Improve prompt/schema specification and parser robustness without changing model weights:

- Prompts now require JSON-only output (no Markdown, no explanations, no nested `{"text": {...}}` wrappers), include OCR text/blocks as supporting context alongside the image, and show an explicit invoice-oriented schema for `document_understanding`.
- `parser.py` extracts the first valid JSON object from free-form output (including ```json``` fences), records `parse_errors` (JSON failures) and `schema_errors` (schema violations) separately, applies a schema-normalization compatibility layer for known flat invoice fields, preserves `raw_response`, and never fabricates missing field values.
- Optional `ocr_correction` task lets the VLM propose OCR fixes using the image as evidence; the OCR layer itself remains unchanged.
- Each inference records `inference_logs` with `model_id`, `task`, `latency_seconds`, `parse_success`, `schema_valid`, `schema_normalized`, `document_type`, `number_of_extracted_fields`, `ocr_text_length`, `timestamp`, `page_number`, `parse_errors`, and `schema_errors`.

**Status:** These changes are not claimed to fully solve structured extraction until a real inference rerun confirms improved JSON validity and field population. Re-run:

```powershell
.\.venv\Scripts\python.exe scripts\run_vlm.py --image path\to\document.png --task document_understanding --pretty
```

Check `results[*].parse_success`, `results[*].schema_valid`, `results[*].schema_normalized`, `results[*].parse_errors`, `results[*].schema_errors`, and `inference_logs` in the CLI output.

**Research observation (Phase 2.2):** Prompting improved JSON validity, but the baseline VLM produced a flat field structure instead of the requested nested schema. This motivates measuring schema compliance separately from JSON validity. The schema-normalization layer relocates known flat invoice fields into `fields` for evaluation compatibility; it is not model improvement and normalized fields do not count as schema compliance.

**Next research hypothesis:** If prompting and parsing improvements increase JSON validity but schema compliance and field accuracy remain low, the next experiment should compare (a) task-specific few-shot examples in the prompt, (b) constrained decoding / JSON grammars if supported by the runtime, and (c) a slightly larger SmolVLM checkpoint—still without fine-tuning—while measuring field extraction against labeled samples.

## Baseline VLM (Experiment 01)

The baseline VLM is a **separate branch** from OCR: it takes the page image and
a free-form prompt, and returns raw model text. It does not read OCR output and
does not perform OCR correction.

```
                 ┌──→ Tesseract OCR ──→ structured OCR result
document → image ┤
                 └──→ baseline VLM ──→ visual reasoning (free-form text)
```

### Module layout

| File | Purpose |
|------|---------|
| `src/document_understanding/vlm_config.py` | `BaselineVLMConfig`: model id, device, dtype, `max_new_tokens`, temperature, cache dir |
| `src/document_understanding/vlm_loader.py` | `load_baseline_vlm()`: cached model + processor loading, device/dtype resolution, actionable load and OOM errors |
| `src/document_understanding/vlm_inference.py` | `run_vlm_inference(image, prompt, config)`: generation only |
| `src/document_understanding/prompts.py` | Reusable baseline prompts: `describe`, `classify`, `key_value`, `qa` |
| `scripts/run_vlm.py` | CLI (`--prompt` / `--prompt-template` runs the baseline path) |
| `scripts/create_synthetic_invoice.py` | Generates the synthetic test invoice |

Model loading is cached per (model id, device, dtype, cache dir), so repeated
inference calls do not reload or re-download weights.

### Selected model

**`HuggingFaceTB/SmolVLM-256M-Instruct` (~256M parameters, ~500 MB in float32).**

Why: it is the smallest instruction-tuned open-weight VLM supported by
`AutoModelForImageTextToText` in the installed Transformers version, it runs on
CPU in ~10–15 s per prompt, and it needs no CUDA, no quantization, and no
`bitsandbytes`. Larger checkpoints (SmolVLM-500M, Qwen2-VL-2B, 7B-class models)
are better at structured extraction but are impractical on a CPU-only laptop.

The model id is configurable — set `VLM_MODEL_ID` or pass `--model-id`. Nothing
else in the project hard-codes a checkpoint.

### Hardware limitations

- CPU-only: no CUDA, no GPU offloading, no quantized kernels
- Use `float32` on CPU (float16 CPU kernels are missing or slow); the config
  picks `float16` automatically only when CUDA is detected
- Expect ~10–15 s per prompt at 128–256 new tokens, and ~1.6 GB process RSS
- GPU execution is recommended before drawing any quality conclusions or moving
  to larger checkpoints

### Installation

```powershell
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
```

`psutil` is optional and only used to report process memory in results.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLM_MODEL_ID` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Hugging Face model id |
| `VLM_DEVICE` | `auto` | `auto` detects CUDA, otherwise CPU |
| `VLM_DTYPE` | device-dependent | `float32` on CPU, `float16` on CUDA |
| `VLM_MAX_NEW_TOKENS` | `256` | Generation limit |
| `VLM_TEMPERATURE` | `0.0` | `0.0` means greedy decoding |
| `VLM_CACHE_DIR` | unset | Hugging Face cache directory |
| `VLM_LOCAL_FILES_ONLY` | `false` | Require locally cached weights |

### Run the baseline

```powershell
python scripts/create_synthetic_invoice.py
python scripts/run_vlm.py --image data/synthetic/synthetic_invoice.png --prompt "Describe this document."
python scripts/run_vlm.py --image data/synthetic/synthetic_invoice.png --prompt-template classify --pretty
python scripts/run_vlm.py --image data/synthetic/synthetic_invoice.png --prompt-template qa --question "What is the invoice number?" --pretty
```

Optional overrides: `--model-id`, `--device`, `--max-new-tokens`.

### Expected output

```json
{
  "response": "This document is an invoice ...",
  "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
  "device": "cpu",
  "inference_time": 12.026,
  "memory": {"process_rss_mb": 1573.71}
}
```

### Test data

Only synthetic documents are used. `scripts/create_synthetic_invoice.py`
generates a fictional invoice (`INV-001` / `Alice Example` / `Laptop`); no real
identity documents, bank statements, personal data, or scraped documents are
included in this repository.

### Baseline tests

```powershell
pytest tests/test_vlm_baseline.py -v          # unit tests, no model download
pytest tests/test_vlm_smoke.py -m integration -v   # live model smoke test
```

Measured results for the baseline run are recorded in
`experiments/experiment_01_baseline/`.

## Repository Layout

```
DocGuard-AI/
├── data/           # Raw, processed, synthetic, noisy, adversarial datasets
├── configs/        # Experiment and model configuration files
├── models/         # Base weights, checkpoints, adapters
├── src/            # Core Python modules (preprocessing → inference)
├── training/       # Dataset prep, SFT, LoRA, alignment scripts
├── evaluation/     # Benchmarks, metrics, robustness suites
├── experiments/    # Per-experiment artifacts and run configs
├── backend/        # FastAPI service (future)
├── frontend/       # React application (future)
├── notebooks/      # Exploratory and paper-reproduction notebooks
├── tests/          # Unit and integration tests
└── scripts/        # Utility and automation scripts
```

## Getting Started

1. Create a virtual environment and install core dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy environment template and adjust paths:

   ```bash
   cp .env.example .env
   ```

3. Place datasets according to `data/README.md`. Do not commit raw sensitive data.

## License

TBD.

## Contributing

TBD — research contributions via experiment branches and documented reproducibility checks.
