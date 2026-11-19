# Weight-Space Programming for SE.LLMA

An empirical and critical investigation into whether ideas from **Sparks of Weight-Space Programming** (Takehi et al.) are useful for **SE.LLMA**'s goal of building strong, sovereign Swedish language models.

All interactive exploration, diagnostics, and report previews are implemented with [marimo](https://marimo.io/), alongside modular Python runners in `src/`.

## Repository Structure

```text
.
├── data/
│   └── swedish_test_suite.json            # 32-prompt Swedish/English diagnostic & stop-point suite (12 dev / 20 eval)
├── notebooks/
│   ├── 01_explore_models_cpu.py           # CPU screening & resource tracker across small open models
│   ├── 02_eval_qwen_suite.py              # Decoupled suite runner + conditional external evaluation viewer
│   ├── 03_wsp_interventions.py            # 6-arm WSP vs. Awake DES vs. Micro-LoRA interactive dashboard
│   └── report.py                          # Interactive report & one-click Markdown/PDF exporter
├── src/
│   ├── wsp_experiment.py                  # Reproducible 6-arm CPU experiment & Matplotlib plot generator
│   └── report_builder.py                  # Research note (Markdown) & WeasyPrint A4 PDF builder
├── results/                               # Saved JSON outcomes, model revisions, seeds, and figures (.png)
├── report/
│   ├── research_note.md                   # Concise ~1–2 page Markdown research note
│   └── report.pdf                         # Final formatted 9-page PDF report
├── pyproject.toml                         # Python dependencies (managed with uv)
└── README.md
```

## Key Findings Summary

- **Baseline Diagnostics (`Qwen/Qwen2.5-0.5B-Instruct`, `bfloat16`, `seed=42`):** `8/32` (`25.0%`) strict pass overall (`3/16` Swedish `target` vs. `5/16` `non_negotiable` stop-points). Context-grounded Swedish reading comprehension passed `3/3`, while ungrounded Swedish collapsed (`0/8` `everyday_sv`) despite fluent English (`3/4`, `2.00/2`).
- **Static WSP vs. Awake DES Layer Localization:** Data-free `lm_head` → `mlp.down_proj` alignment shows near-zero correlation with awake DES layer sensitivity (**Pearson `r = -0.0747`**, **Spearman `ρ = -0.1527`**), confirming the *paradox of non-locality*.
- **6-Arm Held-Out (`eval`) NLL Trade-Off:** Gradient-free column edits (Static WSP `ΔSV = +0.001`, Awake DES `ΔSV = +0.003`) produced zero held-out Swedish gain over baseline (`2.6328` NLL), matching 3-seed Random edits (`ΔSV = +0.004 ± 0.0013`), though DES stop-point calibration avoided English regression (`ΔEN = -0.011` vs. Random `+0.015`). `12`-shot `dev` Micro-LoRA overfit ungrounded Swedish (`ΔSV = +0.075`, `ΔEN = +0.051`) yet improved **4/4** held-out `coding_sv` prompts (`Δ = -0.056`), illustrating *unbricking* of latent skills rather than creation of missing Swedish fluency.

## Setup & Reproduction

### Prerequisites

- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended)

### Installation

```bash
uv sync
```

### Reproducing the Experiments & Report

1. **Screen candidate small models on CPU (`results/cpu_prompt_benchmark.json`):**
   ```bash
   uv run marimo run notebooks/01_explore_models_cpu.py
   ```
2. **Run the 32-prompt Swedish diagnostic suite (`results/qwen_baseline_responses.json`):**
   ```bash
   uv run marimo run notebooks/02_eval_qwen_suite.py
   ```
3. **Run the 6-arm WSP vs. Awake DES vs. Micro-LoRA experiment (`results/experiment_details.json`, `results/summary.json`, and `results/*.png`):**
   ```bash
   # NLL + layer mapping (~91 min on 8-thread Ryzen 7 5800U CPU, ~68 min in Micro-LoRA backprop):
   uv run python -m src.wsp_experiment --nll-only

   # Or run with 20-prompt held-out generations per arm (~+16-20 min CPU) / inspect interactively:
   uv run marimo run notebooks/03_wsp_interventions.py
   ```
4. **Build `report/research_note.md` and `report/report.pdf` (or preview interactively in marimo):**
   ```bash
   uv run python -m src.report_builder
   uv run marimo run notebooks/report.py
   ```

## Deliverables Checklist

- [x] **Critique & Hypothesis**: Critical examination of *Sparks of Weight-Space Programming* (including replication/cost evidence from `eomcaleb/weight-space-bench`, `MatinKhajavi/Weight-Space-Programming`, `WeightBench`, sovereignty considerations for SE.LLMA, and Dr. Jesús Martín-Fernández's awake cognitive DES mapping analogy) and falsifiable 3-part hypothesis (`H1`–`H3`).
- [x] **Code**: Reproducible `marimo` notebooks and `src/` modules implementing all 6 control/comparison arms (`baseline`, `wsp_static`, `des_awake_edit`, 3-seed `random_edit`, `prompt_control`, `micro_lora`) with strict `dev`/`eval` split separation.
- [x] **Results**: Saved model revision (`7ae557604adf67be50417f59c2c2f167def9a775`), random seeds (`42`, `[42, 123, 999]`), evaluation artifacts, per-item NLL outcomes, and diagnostic figures in `results/`.
- [x] **Research Note & Report**: Concise Markdown research note ([report/research_note.md](report/research_note.md)) and 9-page PDF report ([report/report.pdf](report/report.pdf)).
- [x] **AI Usage**: Documented below and in Section 6.2 of the report.

## AI Usage

- **Tools & Contributions:** **Gemini Hydrogen** (via Antigravity agent workflows) was used to:
  1. Propose the 32-prompt Swedish/English evaluation suite (`data/swedish_test_suite.json`), draft rubric-aligned reference answers, and evaluate the baseline responses in `results/qwen_baseline_eval.json` alongside automated Python subprocess tests and regex checks.
  2. Synthesize replication benchmarks and verify external literature references (`WeightBench`, `eomcaleb/weight-space-bench`, `MatinKhajavi/Weight-Space-Programming`, Muon, and Dr. Jesús Martín-Fernández's awake brain mapping publications).
  3. Scaffold the four `marimo` notebooks (`01_explore_models_cpu.py`, `02_eval_qwen_suite.py`, `03_wsp_interventions.py`, `report.py`) and implement `src/wsp_experiment.py` and `src/report_builder.py`.
- **Verification:** All hypotheses, prompt evaluations, control arms, model outputs, automated Python unit-test subprocess checks, and reported numbers were executed locally on CPU and directly verified against saved JSON artifacts and rendered PDF pages.
