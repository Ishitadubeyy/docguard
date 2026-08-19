# Experiment 02 — Document Understanding Baseline

Measures what the **unmodified** baseline VLM (`HuggingFaceTB/SmolVLM-256M-Instruct`,
~256M parameters, CPU, float32) can do on five document understanding tasks over a
synthetic document set. No fine-tuning, no OCR context, no output correction.

This is the reference point that SFT / LoRA / QLoRA / preference alignment will later
be compared against on the same documents, tasks, and metrics.

## Setup

| Item | Value |
|------|-------|
| Model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Device / dtype | cpu / float32 (no CUDA available) |
| Generation | greedy, `max_new_tokens=256`, `temperature=0.0` |
| Runtime | Python 3.10.12, torch 2.13.0+cpu, transformers 5.15.0, Linux CPU VM |
| Documents | 4 synthetic: invoice, bank statement, salary slip, identity document |
| Tasks | classification, key_value_extraction, table_extraction, summarization, question_answering |
| Runs | 20 (4 documents × 5 tasks) |
| OCR context | not used — image only, so the numbers measure the VLM alone |

All document content is fictional. No real identities, accounts, or scraped documents.

## Reproduce

```bash
python scripts/create_synthetic_documents.py
python scripts/evaluate_document_understanding.py
```

Single task on one document:

```bash
python scripts/run_document_analysis.py \
  --image data/synthetic/documents/synthetic_invoice.png \
  --task question_answering --question "What is the total amount?" --pretty
```

## Results (status: RUN)

| Task | Metric | Result |
|------|--------|--------|
| Classification | accuracy | **4/4 = 1.00** |
| Key-value extraction | field-level exact accuracy | **10/25 = 0.40** |
| Table extraction | row accuracy / cell accuracy | **0/7 = 0.00 / 0/24 = 0.00** |
| Question answering | exact match | **3/4 = 0.75** |
| Summarization | none | no numerical score in this phase |

Per-document key-value accuracy: salary slip 5/7, invoice 3/6, identity document 2/5,
bank statement 0/7.

Latency (CPU, per task run): mean 17.3 s, min 14.2 s, max 23.2 s, total 345.5 s for 20 runs.
Process RSS stays around 1.6 GB.

Raw model output, parsed output, per-field scores, and timings for every run are in
`results.json`. Failure analysis is in `analysis.md`.

## Limitations

- 256M parameters: the model transcribes reliably but follows output-structure
  instructions poorly, which is where most of the loss comes from.
- Table extraction failed completely — the model echoed the prompt's schema
  placeholders (`{"columns": ["<column>"], "rows": [["<cell or null>"]]}`) instead of
  extracting rows. Placeholder echoes are rejected by the parser rather than counted.
- Key-value extraction returns `{"fields": [ {...}, {...} ]}` (a list of loosely-named
  objects) instead of the requested flat object, so correct values often land under
  wrong keys; those values are preserved in `additional_fields` but scored as misses.
- Exact-match scoring is strict: a right value under a wrong key is a miss.
- CPU-only float32 inference; timings are not comparable to GPU runs.
- Four documents, one page each — the sample is far too small for statistical claims.
