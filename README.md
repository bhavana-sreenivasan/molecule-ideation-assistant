# Molecule Ideation Assistant

**AI-assisted analog generation for drug discovery, built on NVIDIA BioNeMo (MolMIM), RDKit, an LLM reasoner (Muse Glimmer via NVIDIA NIM), and a LangGraph agent loop.**

Give it a lead compound (a SMILES string, or an approved drug pulled from ChEMBL) and it generates analogs, scores them against drug-likeness, toxicity, synthesizability and estimated potency, then returns a ranked top 3 with a plain-English medicinal chemistry rationale. A multi-round agent version can regenerate when too many candidates fail the filters and pauses for human approval before the final ranking.

> **Status:** research / prototype notebook. The scores are screening heuristics, not experimental data or clinical guidance. See [Limitations](#limitations).

---

## What it does

```
Lead SMILES ──► Generate (MolMIM) ──► Validate & score (RDKit + ChEMBL) ──► Rank & explain (Muse Glimmer)
```

| Stage | Tool | Role |
|-------|------|------|
| 1. Generate | NVIDIA BioNeMo **MolMIM** | Produces 10 analogs of the lead, optimizing QED with a minimum similarity of 0.3 |
| 2. Validate | **RDKit** + **ChEMBL** | Dedupes, then scores each analog on QED, LogP, Lipinski, PAINS, hERG-risk heuristic, reactive-group alerts, SA score and estimated potency |
| 3. Rank | **Muse Glimmer** (NIM, `meta/muse-glimmer-30b`) | Ranks the top 3 with strengths and concerns for each |

### Scoring dimensions

| Dimension | Method |
|-----------|--------|
| **Drug-likeness** | RDKit QED (higher is better); Lipinski rule of 5 |
| **Toxicity** | LogP window, PAINS filter catalog, SMARTS alerts for reactive groups (hydroxylamine, acid halide, alkyl halide, azide, epoxide), hERG-risk heuristic |
| **Synthesis feasibility** | RDKit Contrib SA score (1 = easy, 10 = hard). Under 4 is Easy, under 6 is Moderate, otherwise Complex |
| **Potency (estimate)** | Similarity-weighted kNN over known ChEMBL activities (pChEMBL) on the lead's most-measured human target, with HIGH/MEDIUM/LOW confidence from nearest-neighbor Tanimoto similarity |

**hERG-risk heuristic.** One point each for a basic aliphatic amine, LogP ≥ 3.5, and ≥ 2 aromatic rings. 0–1 is LOW, 2 is MODERATE, 3 is HIGH. This is a rule of thumb, not a hERG prediction model.

---

## Notebook walkthrough

| Step | Content |
|------|---------|
| 1–2 | Install dependencies; load API keys from Colab Secrets |
| 3 | Connectivity test for both the LLM and MolMIM endpoints |
| 3b | SA score and hERG-risk heuristic helpers |
| 4–5b | Core pipeline functions, an **aspirin** smoke test, results table and 2D structure grids |
| 6 | Real drug leads (Imatinib, Gefitinib, Oseltamivir, Metformin, Ciprofloxacin) with PAINS/hERG/alert screening; full run on **Imatinib** |
| 7 | Pulls approved small molecules (max phase 4) from the **ChEMBL API** instead of a hardcoded list |
| 8 | Toxicity screening function (PAINS, hERG heuristic, structural alerts) |
| Potency | ChEMBL-based potency estimator with leave-one-out validation |
| Evaluation | QED gain, Tanimoto diversity, filter pass rates, synthesis feasibility, potency summary, and plots |
| LangGraph | Stateful agent loop with regeneration and human-in-the-loop review |
| Eval | One-shot vs. agent-loop comparison across the five ChEMBL leads |

### Agent loop (LangGraph)

```
generate ─► validate ─► diagnose ──(pass rate < threshold and iterations left)──► generate
                           │
                           └──► human_review ──(approved)──► rank ─► END
                                      │
                                      └──(rejected)──► generate
```

- **generate:** the first round seeds MolMIM with the lead; later rounds reseed from the best-QED analog of the previous round (MolMIM takes no text feedback, so reseeding is the regeneration step).
- **validate:** full RDKit, PAINS, hERG and potency scoring, and the pass rate (candidates clearing Lipinski, PAINS and overall toxicity).
- **diagnose:** finds the most common failure mode and writes feedback.
- **human_review:** uses LangGraph `interrupt()`, so the run pauses until an approve/reject decision is supplied.
- **Defaults:** `max_iterations = 4`, `pass_threshold = 0.5`.

---

## Setup

### Requirements

- Google Colab (the notebook uses `google.colab.userdata` for secrets), or adapt `get_key()` to read environment variables elsewhere
- An NVIDIA API key with access to BioNeMo MolMIM and NIM-hosted models
- Python packages (installed in the first cells):

```bash
pip install rdkit pandas requests python-dotenv openai streamlit chembl-webresource-client -q
pip install -q langgraph langgraph-checkpoint
```

`matplotlib` and `numpy` are also used and ship with Colab.

### API keys

Add these in **Colab → Secrets** (🔑 icon) and enable notebook access:

| Secret | Used for |
|--------|----------|
| `NVIDIA_API_KEY` | MolMIM molecule generation (required) |
| `LLM_API_KEY` | Muse Glimmer ranking. Falls back to `LLAMA_API_KEY`, then `NVIDIA_API_KEY` |

Keys are never hardcoded. Do not commit them to the repo.

### Running

1. Open the notebook in Colab and add the secrets above.
2. Run the cells top to bottom. Step 3 verifies both endpoints and prints a hint for each failure (401/403 means wrong key, 404 means wrong model id, 429 means rate limit).
3. To use your own lead, set `lead = "<your SMILES>"` and call `generate_analogs` → `validate_and_score` → `rank_with_llm_v2`.
4. Potency estimates require a ChEMBL ID, so they are available for ChEMBL leads only (`build_potency_reference("CHEMBLxxxx")`).

If the model id in `LLM_MODEL` differs from the NIM catalog, the notebook looks up a Glimmer model from `/v1/models` automatically.

---

## Example results (from the saved run)

**Lead: Prazosin (CHEMBL2)**

| Metric | Value |
|--------|-------|
| Lead QED | 0.727 |
| Best analog QED | 0.925 (+27.3%) |
| Average analog QED | 0.854 |
| Avg. Tanimoto similarity to lead | 0.436 |
| Lipinski pass | 10/10 |
| Overall toxicity / PAINS pass | 8/10 each |
| Synthesis | 10/10 Easy (mean SA 2.33) |
| Potency reference | Alpha-1A adrenergic receptor (CHEMBL229), 2,241 known compounds |
| Potency-model leave-one-out | MAE 0.67 pChEMBL units, Pearson r = 0.73 |
| Best analog estimated pChEMBL | 8.50 vs. lead 9.23; **0/10 analogs estimated ≥ lead** |

**One-shot vs. agent loop (5 leads).** Best-QED and QED-gain figures were similar between the two approaches. The agent loop raised the pass rate on one lead (Ofloxacin, 0.7 → 1.0), lowered it on another (Indomethacin, 1.0 → 0.7), and left the rest unchanged or marginally different. The notebook does not show a consistent advantage for the loop on these metrics. With one run per lead and 10 molecules per run, the comparison is anecdotal.

Imatinib was also run end to end (10 analogs generated, 6 unique after deduplication).

---

## Limitations

- **Heuristic safety.** hERG risk and reactive-group alerts are simple rules, and PAINS flags assay interference, not toxicity. None of these replace ADMET models or experiments.
- **QED optimization bias.** MolMIM optimizes QED, which favors small, simple, "pretty" molecules. Analogs may drift far from the lead's pharmacophore (e.g. losing the 2-amino group in Prazosin analogs) and the pipeline does not check target binding.
- **Potency is a ligand-based estimate.** It is a kNN over ChEMBL data, only as good as the nearest known compounds, and not a docking or affinity prediction. Treat LOW-confidence values as unreliable.
- **LLM rationales are not validated.** Muse Glimmer ranks and explains based on the numbers it is given and may overstate or misjudge chemistry. Review its output critically.
- **Small samples.** Each run generates 10 molecules (often fewer after deduplication and validity checks). Results vary between runs.
- **ChEMBL lead selection.** Step 7 takes the first five matches returned by the API, not a curated set.
- **External dependencies.** Requires live NVIDIA and ChEMBL APIs; the SA scorer is downloaded from the RDKit GitHub repo if not bundled.

---

## Possible next steps

- Replace heuristics with trained ADMET / hERG models
- Add docking or a target-specific activity model for potency
- Add scaffold-preservation constraints and retrosynthesis checks
- Wrap in a Streamlit UI (`streamlit` is already installed)
- Run repeated trials per lead for a statistically meaningful loop vs. one-shot comparison

---

## Acknowledgments

[NVIDIA BioNeMo MolMIM](https://build.nvidia.com/nvidia/molmim-generate) · [RDKit](https://www.rdkit.org/) · [ChEMBL](https://www.ebi.ac.uk/chembl/) · [LangGraph](https://github.com/langchain-ai/langgraph) · NVIDIA NIM (Muse Glimmer)

## Disclaimer

For research and educational use only. Outputs are computational suggestions and must not be used for clinical or therapeutic decisions.
