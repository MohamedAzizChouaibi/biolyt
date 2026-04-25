# BiolytAI Take-Home Assignment — Clinical Trial Data Cleaning & Biomarker Extraction

**Dataset:** 5,000 clinical trials from ClinicalTrials.gov (`trials_sample.csv`)

---

## Approach

### 1. Drug / Intervention Name (`intervention_name`)

**Method:** Two-stage — curated override dictionary + RapidFuzz token-sort clustering.

Many entries are pipe-separated multi-drug strings (e.g. `Pembrolizumab + Carboplatin`), so each component is harmonized independently. Stage 1 covers ~120 common drugs via exact and substring matching (brand names → INN, aliases → canonical). Stage 2 clusters remaining high-frequency single-drug entries using `token_sort_ratio ≥ 88`, picking the longest title-cased variant as canonical.

**Why:** Pure fuzzy matching is noisy for drugs — `Keytruda` and `Kymriah` share enough characters to falsely cluster. Seeding with a domain-specific override map keeps precision high; fuzzy handles the long tail of spelling variants and suffix noise (` hydrochloride`, ` injection`).

---

### 2. Disease / Condition Name (`condition`)

**Method:** Same two-stage approach. Conditions are also pipe-separated; each label is harmonized independently with deduplication within the same row.

The override map covers ~90 key disease categories with their common abbreviations (NSCLC, AML, HCC, etc.) and spelling variants. Fuzzy clustering (threshold 88) merges remaining near-duplicates.

**Why:** Conditions are more heterogeneous than drugs — the same entity can appear as a formal MeSH term (`Carcinoma, Non-Small-Cell Lung`), a plain-English phrase (`non-small cell lung cancer`), or an acronym (`NSCLC`). Substring matching catches most cases; the fuzzy pass catches typos and word-order variation.

---

### 3. Sponsor Name

**Method:** Override map for ~30 major pharma/biotech/academic sponsors (exact + partial token match after stripping legal suffixes: LLC, Ltd, Inc, GmbH, etc.) followed by fuzzy clustering (`token_sort_ratio ≥ 90`) on sponsors appearing ≥ 2 times.

**Why:** Legal-suffix stripping is essential — `Pfizer`, `Pfizer Inc.`, `Pfizer, Inc` differ only in boilerplate. The fuzzy pass then handles abbreviations and minor transcription differences (e.g. `Bristol Myers Squibb` vs `Bristol-Myers Squibb`).

---

### 4. Country

**Method:** Explicit lookup table covering common aliases (USA/US/United States, UK/United Kingdom, etc.). Pipe- and comma-separated multi-country fields are split, each token normalized, then re-joined sorted.

**Why:** Country names are low-cardinality and highly structured — a lookup table is faster, more precise, and more interpretable than any ML approach. Fuzzy matching would be overkill and risks merging genuinely different countries.

---

### 5. Biomarker Extraction

**Model:** `Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2.1` — Qwen2.5-0.5B-Instruct with a LoRA adapter fine-tuned on biomedical text.

**Prompt design:** Each trial produces one prompt containing its `brief_title`, `primary_outcome` (full), and `secondary_outcome` (truncated to 600 chars). A system message instructs the model to return a comma-separated list of biomarkers or `NONE`.

**Batching:** Batch size 16 (GPU) / 4 (CPU), left-padded, greedy decoding (`do_sample=False`), `max_new_tokens=128`. Greedy decoding is chosen over sampling because biomarker lists are factual — temperature adds noise without benefit.

**Post-processing:** Raw output is stripped of chat template tokens, split on `,`/`;`, de-duplicated case-insensitively, and bullet/numbering prefixes are removed. `NONE` outputs and empty strings produce an empty list, stored as `""` in the CSV.

**Hardware used:** Tested on CPU (development). Full GPU run expected on CUDA device.

**Estimated inference time:** ~4–6 ms/trial on an A100 (batch=16). On CPU: ~200–400 ms/trial.

---

## Trade-offs

| Decision | Speed | Accuracy | Notes |
|---|---|---|---|
| Override map first, fuzzy second | Fast | High precision | Miss rate on truly novel names |
| `token_sort_ratio ≥ 88` threshold | Moderate | Balanced | Lower = more merging (risk FP); higher = less coverage |
| Greedy decoding | Faster | Deterministic | Sampling could surface more biomarkers at cost of consistency |
| Truncate secondary outcomes to 600 chars | Faster | Minor loss | Most biomarkers appear in first 600 chars |
| Batch size 16 | Balanced | — | Larger batches faster but need more VRAM |

---

## What I'd improve with more time

1. **Drug harmonization:** Query RxNorm/ChEMBL APIs to map brand names to normalized INN + concept IDs, enabling cross-study deduplication that goes beyond string matching.
2. **Disease harmonization:** Map conditions to MeSH / ICD-10 codes via the UMLS API or `scispacy` NER, giving a stable ontology-backed identifier rather than a free-text canonical.
3. **Sponsor:** Use OpenCorporates or a company database to resolve legal entity variants to a canonical company ID.
4. **Biomarker extraction:** Fine-tune the extraction prompt on a small set of manually annotated trials from this dataset to reduce hallucination on non-clinical-trial text patterns. Post-filter against a known biomarker lexicon (HGNC gene symbols, UniProt IDs) to flag low-confidence extractions.
5. **Confidence scores:** Surface per-field harmonization confidence so downstream consumers can decide when to trust the canonical label vs. keep the original.

---

## Repository layout

```
.
├── trials_sample.csv                  # raw input
├── 01_harmonization.ipynb             # data cleaning (run first)
├── 02_biomarker_extraction.ipynb      # biomarker extraction (run second)
├── README.md
└── output/
    ├── trials_harmonized_step1.csv    # intermediate (after harmonization)
    ├── trials_cleaned_final.csv       # final output with all clean columns + biomarkers
    ├── drug_dictionary.json           # raw variant → canonical drug name
    ├── disease_dictionary.json        # raw variant → canonical disease name
    ├── sponsor_dictionary.json        # raw variant → canonical sponsor name
    ├── country_dictionary.json        # raw variant → canonical country name
    ├── harmonization_metrics.json     # unique-value counts before/after
    ├── biomarker_metrics.json         # extraction rate, top biomarkers, timing
    └── edge_cases.json                # flagged edge cases
```

---

## Running the notebooks

```bash
pip install pandas numpy rapidfuzz tqdm transformers peft torch accelerate jupyter

# Step 1 — harmonization (CPU, ~2 min for 5k rows)
jupyter nbconvert --to notebook --execute 01_harmonization.ipynb --output 01_harmonization_executed.ipynb

# Step 2 — biomarker extraction (GPU recommended)
jupyter nbconvert --to notebook --execute 02_biomarker_extraction.ipynb --output 02_biomarker_extraction_executed.ipynb
```

Or open them interactively in JupyterLab and run all cells in order.
