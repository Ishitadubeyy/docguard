# Data Directory

This directory holds all datasets for DocGuard AI. **Do not commit raw sensitive or personally identifiable data to version control.**

## Layout

| Subdirectory | Purpose |
|--------------|---------|
| `raw/` | Original, unmodified source documents and annotations |
| `processed/` | Cleaned, normalized, and split datasets ready for training or evaluation |
| `synthetic/` | Generated or augmented samples (e.g., synthetic IDs, forged variants for research) |
| `noisy/` | Degraded inputs for robustness evaluation (blur, compression, skew, low DPI) |
| `adversarial/` | Adversarial or attack-oriented samples for security evaluation |

## Conventions (planned)

- Use consistent naming: `{task}_{split}_{version}` (e.g., `doc_extract_train_v1`)
- Document provenance, consent status, and license in a sidecar `metadata.json` per dataset
- Prefer processed parquet/JSONL for structured fields; keep raw files immutable
- Synthetic and adversarial data must be clearly labeled and separated from production training paths

## Privacy and compliance

- Identity verification data must be consented or fully synthetic
- Never store production credentials or live customer documents in this repository
- Refer to project policy before adding new raw data
