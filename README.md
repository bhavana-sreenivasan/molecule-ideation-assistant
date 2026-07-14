# Molecule Ideation Assistant
### Powered by NVIDIA BioNeMo (MolMIM) + Llama 3.3 70B (NVIDIA NIM)

## What it does
An AI-powered drug analog ideation pipeline that takes a real FDA-approved 
lead compound and generates optimised analog candidates under three simultaneous 
constraints — potency, toxicity, and synthesis feasibility.

**Reduces analog ideation from 2–4 weeks of manual work to under 5 minutes.**

---
```
## Pipeline
Lead Compound (ChEMBL API)
↓
NVIDIA BioNeMo MolMIM → Generate 10 analogs
↓
RDKit → QED, LogP, Lipinski, PAINS, hERG, Structural alerts
↓
Llama 3.3 70B (NVIDIA NIM) → Rank + explain top 3
↓
2D Visualizations + Evaluation Metrics

```
---

## Tech Stack
| Tool | Role |
|------|------|
| NVIDIA BioNeMo (MolMIM) | GPU-accelerated analog generation |
| Llama 3.3 70B (NVIDIA NIM) | Multi-constraint LLM reasoning + ranking |
| RDKit | PAINS filtering, hERG screening, structural alerts, QED, Lipinski |
| ChEMBL API | Programmatic FDA-approved lead compound sourcing |
| Google Colab T4 GPU | NVIDIA GPU compute |
| Python | Pipeline orchestration |

---

## Results
| Lead Compound | QED Improvement | Lipinski | PAINS | Toxicity |
|---------------|----------------|----------|-------|----------|
| Aspirin | +28.4% | 10/10 | 10/10 | 10/10 |
| Imatinib (Gleevec) | +29.7% | 10/10 | 10/10 | 10/10 |
| Prazosin (ChEMBL) | +24.8% | 10/10 | 9/10 | 9/10 |

---

## Key Features
-  Real FDA-approved lead compounds from ChEMBL API
-  GPU-accelerated molecular generation via NVIDIA BioNeMo
-  Multi-layer toxicity screening — PAINS, hERG proxy, structural alerts
-  LLM reasoning with two prompt versions (basic + PAINS-aware)
-  2D molecule structure visualizations (RDKit)
-  Evaluation metrics — QED improvement, Tanimoto diversity, filter pass rates
-  Plain-English Llama 3.3 explanations for every ranked analog

---


## Leads Tested
- Aspirin — pain reliever (initial test)
- Imatinib (Gleevec) — cancer drug (CML)
- Prazosin (CHEMBL2) — antihypertensive (ChEMBL API)

---
