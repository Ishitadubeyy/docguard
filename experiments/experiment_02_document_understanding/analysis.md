# Experiment 02 — Analysis

All observations below come from `results.json`; model output was recorded verbatim and
never corrected.

## What worked

**Classification: 4/4.** Every document was labeled correctly, and three of the four
responses declared the exact target label (`bank_statement`, `salary_slip`, `INVOICE`).
The identity document was labeled `Identity Card`, mapped to `identity_document` by
alias matching.

**Question answering: 3/4.** Single-field lookups were answered with the bare value:
`INV-001`, `35000`, `ID-987654`. This matches the Phase 2 finding that pointed lookup is
the model's strongest structured behavior.

**Transcription.** Summaries reproduced the visible content of every document with no
invented values observed — including the bank statement, where the model listed all
three transactions in order. The summaries read as transcriptions rather than
abstractive summaries, which for grounding purposes is the safer failure mode.

## What failed

**Table extraction: 0/7 rows, 0/24 cells.** For all four documents the model returned
the prompt's own schema placeholders:

```json
{"columns": ["<column>"], "rows": [["<cell or null>"]]}
```

This is prompt echo, not extraction. The parser drops placeholder rows, so the task is
scored as a complete failure rather than as one bogus row. Notably, the same
transactions the model failed to tabulate were transcribed correctly during
summarization — the information is visible to the model; it is the requested
*structure* that it cannot produce.

**Key-value extraction: 10/25 fields.** The model consistently answered with
`{"fields": [ {...}, {...} ]}` — a list of objects with improvised key names — instead of
the requested flat object:

```json
{"fields": [{"field": "Invoice Number:", "num": "INV-001", "customer": "Alice Example"}]}
```

`INV-001` is present and correct, but under `num`, so `invoice_number` scores as a miss
(the value is kept in `additional_fields` for inspection). The bank statement scored
0/7 for the same reason: the values appear, attached to the wrong keys, mixed with
statement-period duplicates.

**Question answering error.** Asked for the closing balance of the bank statement, the
model answered `25000` (row 2) instead of `22500` (row 3) — a row-alignment error in a
multi-row table, consistent with the table-extraction failure.

## Interpretation

The 256M baseline separates cleanly into two regimes:

| Capability | Status |
|------------|--------|
| Reading text off the page | reliable |
| Single-label / single-value answers | reliable |
| Multi-row and multi-field structure | unreliable |
| Following a requested output schema | unreliable |

The dominant error is instruction-following on output format, not optical reading. That
matters for the roadmap: prompt engineering alone is unlikely to fix schema compliance
at this size, whereas supervised fine-tuning on the exact target JSON is directly aimed
at this failure. Constrained decoding is the cheaper alternative worth testing first.

## Honest caveats

- Four documents; every metric moves in large steps (one field = 4% of key-value score).
- Parsing is deliberately generous with key names (aliases, list flattening) and
  deliberately strict with values (exact match). A different parser would move the
  key-value number without any change in model capability.
- The bank statement question has a plausible alternative reading (`balance` at the last
  row vs. an explicitly labeled "closing balance" line, which the document does not
  have); the answer is still wrong for the intended ground truth.
- No OCR context was supplied. Feeding OCR text into the analyzer is supported and may
  change these numbers; that comparison is not part of this experiment.

## Next steps to compare against this baseline

1. Constrained/grammar-based decoding for the table and key-value schemas (no training).
2. Few-shot exemplars showing one filled-in JSON object per document type.
3. A slightly larger checkpoint (SmolVLM-500M) on the same benchmark.
4. SFT → LoRA → QLoRA → preference alignment, each re-scored with
   `scripts/evaluate_document_understanding.py` on the same synthetic set.
