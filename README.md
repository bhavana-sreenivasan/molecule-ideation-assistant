# Molecule Ideation Assistant

**Notebook:** `DRUG_IDEATION_v2.ipynb`
**Domain:** Drug discovery | **Approach:** Generative AI + cheminformatics + LLM reasoning + agentic workflow

An AI-assisted pipeline that takes an approved drug (the *lead compound*) and proposes optimised analog candidates. Each candidate is scored on **drug-likeness**, **toxicity risk**, **synthetic accessibility** and **estimated potency**, then ranked by an LLM that explains its choices in plain medicinal-chemistry language.

The aim is to cut early analog ideation from weeks of manual effort to a few minutes. It is an ideation and triage tool. It does not replace docking, assays or expert review.

---

## How it works

```
Lead SMILES
    │
    ▼
1. GENERATE   NVIDIA BioNeMo MolMIM        → 10 analog SMILES (optimised for QED, min. similarity 0.3)
    │
    ▼
2. VALIDATE   RDKit + ChEMBL               → dedupe, QED, Lipinski, PAINS, hERG heuristic,
    │                                         reactive-group alerts, SA score, potency estimate
    ▼
3. RANK       Muse Glimmer (NVIDIA NIM)    → top 3 with rationale and concerns
```

The final section wraps these stages in a **LangGraph** agent. The agent regenerates when too few candidates pass the filters, and it pauses for **human review** before the final ranking.

### Scoring components

| Dimension | Method | Notes |
|---|---|---|
| Drug-likeness | RDKit `QED` (0–1) | Higher is better. This is also MolMIM's optimisation target. |
| Rule of 5 | MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10 | Pass/fail. |
| Lipophilicity flag | logP > 5 or < 0 | Simple warning. |
| Assay interference | RDKit **PAINS** filter catalog | Flags pan-assay interference substructures. |
| Reactive groups | SMARTS alerts | Hydroxylamine, acid halide, alkyl halide, azide, epoxide. |
| hERG risk | **Heuristic**: +1 each for basic aliphatic amine, logP ≥ 3.5, ≥ 2 aromatic rings (0–1 LOW, 2 MODERATE, 3 HIGH) | A rule of thumb, **not** a hERG prediction model. |
| Synthesis | RDKit Contrib **SA score** (1 = easy, 10 = hard) | < 4 Easy, < 6 Moderate, otherwise Complex. |
| Potency | Similarity-weighted kNN (k = 5) on ChEMBL pChEMBL values, Morgan fingerprints | Reports confidence from nearest-neighbour Tanimoto: HIGH ≥ 0.6, MEDIUM ≥ 0.4, otherwise LOW. |

---

## Notebook walkthrough

| Step | What it does |
|---|---|
| 1 | Installs dependencies. |
| 2 | Loads API keys from Colab Secrets. |
| 3 | Defines the pipeline and smoke-tests both APIs. It auto-resolves the Muse Glimmer model id from the NIM catalog. |
| 3b | SA score loader (falls back to downloading from the RDKit GitHub repo) and the hERG heuristic. |
| 4–5b | Pipeline functions and a first test on **Aspirin**, with 2D structure grids. |
| 6 | Adds PAINS, hERG and reactive-group screening. Runs **Imatinib** and four other hardcoded drugs as leads. |
| 7 | Pulls approved small-molecule leads directly from the **ChEMBL** API (`max_phase=4`). |
| 8 | PAINS toxicity screening function and test. |
| Potency | Builds a target-specific potency reference from ChEMBL and a leave-one-out error check. |
| Evaluation | QED gain, Tanimoto diversity, filter pass rates, SA distribution, potency summary and charts. |
| LangGraph | Agentic loop (generate → validate → diagnose → regenerate or human review → rank) and a comparison against the one-shot pipeline. |

### Agent graph

```
generate → validate → diagnose ──(pass rate < threshold and iterations left)──► generate
                         │
                         └──► human_review ──(approved)──► rank → END
                                   └──(rejected)──► generate
```

- **Regeneration:** MolMIM takes no text feedback, so the loop reseeds from the best-QED analog of the previous round.
- **Defaults:** `max_iterations = 4`, `pass_threshold = 0.9`.
- **Human review:** implemented with LangGraph `interrupt` and `MemorySaver`. The notebook simulates approval; in a real app this would come from a UI.

---

## Setup

### Requirements

The notebook is written for **Google Colab**. It uses `google.colab.userdata` for secrets, and the code that reads them would need changing to run elsewhere.

```bash
pip install rdkit pandas requests python-dotenv openai chembl-webresource-client langgraph langgraph-checkpoint matplotlib numpy
```

`streamlit` is also installed in Step 1 but is not used in the notebook yet.

### API keys (Colab Secrets)

| Secret | Used for |
|---|---|
| `NVIDIA_API_KEY` | BioNeMo MolMIM analog generation. |
| `LLM_API_KEY` | Muse Glimmer ranking via NVIDIA NIM. Falls back to `LLAMA_API_KEY`, then `NVIDIA_API_KEY`. |

Both are NVIDIA API keys. Keys are read at runtime and never hardcoded.

### Endpoints and model

- MolMIM: `https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate`
- LLM: `https://integrate.api.nvidia.com/v1/chat/completions`, model `meta/muse-glimmer-30b` (resolved from the catalog if the id differs)

### Running

1. Add the secrets in Colab and run the notebook top to bottom.
2. Step 3 prints `Both APIs working. Continue.` when both keys are valid. If it does not, a 401 or 403 means a wrong key, 404 a wrong model id, and 429 a rate limit.
3. To use your own lead, set `lead` to a SMILES string and call `generate_analogs`, `validate_and_score`, `check_toxicity_pains` and `rank_with_llm_v2` in sequence. Alternatively, run `app.invoke(...)` with the LangGraph state.

---

## Example results (from the saved outputs)

**Lead: Prazosin (CHEMBL2)**, potency reference on the Alpha-1A adrenergic receptor (2,241 known compounds):

- Lead QED 0.727, best analog QED 0.929 (+27.9%), average analog QED 0.882.
- 10/10 analogs passed Lipinski, PAINS and overall toxicity filters. All were rated Easy to synthesise (mean SA 2.31).
- Average Tanimoto similarity to the lead was 0.482, which indicates reasonable diversity.
- Potency estimator leave-one-out check: **MAE 0.67 pChEMBL units, Pearson r = 0.73**.
- Estimated potency was below the lead for every analog (best estimate 8.89 vs lead 9.23).

**One-shot vs agent loop** (5 ChEMBL leads: Prazosin, Nicotine, Ofloxacin, Nalidixic acid, Indomethacin):

- The loop was equal or better on best QED in 4 of 5 leads.
- The loop improved pass rate on Nicotine (0.90 → 1.00) and Indomethacin (0.70 → 0.89).
- The loop increased diversity on 4 of 5 leads.
- The loop also produced more usable candidates for Prazosin and Nalidixic acid, because one-shot generation lost some to deduplication.

Results vary between runs because MolMIM generation is stochastic.

---

## Limitations

- **QED-driven generation.** MolMIM optimises QED, so the analogs tend to score well on it by construction. A high QED gain says little about biological activity.
- **Heuristic safety screens.** The hERG check, reactive-group alerts and logP flags are screening heuristics, not validated toxicity models. PAINS covers assay interference, not in-vivo toxicity.
- **Potency is a ligand-based estimate.** It uses nearest-neighbour similarity on one target. It only applies to ChEMBL leads with human pChEMBL data. It is unreliable for structures far from known actives (see the confidence column), and it cannot capture new binding modes.
- **LLM rationale is not ground truth.** The ranking explanation is generated text and can contain chemical claims that need checking by a chemist.
- **Small samples.** Each run has about 10 analogs per lead and the evaluation covers only a few leads, so the metrics are indicative, not statistically robust.
- **Lead selection.** The ChEMBL fetch takes the first approved small molecules returned by the API, not a curated set.
- **No synthesis planning, docking or ADME modelling.**

---

## Technologies

NVIDIA BioNeMo MolMIM · Muse Glimmer via NVIDIA NIM · RDKit · ChEMBL web resource client · LangGraph · pandas / NumPy / matplotlib

---

## Disclaimer

This project is for research and educational purposes. Generated molecules are computational suggestions only. They have not been synthesised or tested, and nothing here is medical advice.
