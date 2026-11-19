# Weight-Space Programming for SE.LLMA — Research Note

**Status:** Final (All 6 Arms Complete) · **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (`bfloat16`, CPU, `seed=42`) · **Updated:** `2026-11-19 20:35 UTC`

## 1. Critical Evaluation of *Sparks of Weight-Space Programming*
- **Strongest evidence:** The MIT team explores a new paradigm: letting an LLM interpret and rewrite the weights of another model based on a user's text input. In the cases shown in their blog, they first test whether a frontier LLM (`GPT-6 Astra`) can "understand" the parameters of a model trained to play a minigame (predicting which game it is), and whether it can fix issues in another LLM suffering from a repetition loop (such as `Qwen3.5-0.8B`). These test cases are intriguing, and if the idea and methods are further improved, this could become an interesting new direction for AI.
- **Critical opinion:**
    1. **Frontier-model dependence & resource costs:** My first doubt whenever a study discovers a new functionality using the newest frontier models is whether it is applicable to LLM technology in general, or just limited to the newest and largest models. In their blog, they use `GPT-6 Astra`—one of the latest LLMs to date—but luckily a follow-up study by `eomcaleb` (`eomcaleb/weight-space-bench`) answers two of my main concerns: the costs and the applicability to other models. And it seems that if you want to use this new approach, you need to use frontier models—which are the most expensive to run, private, subsidized, and consume heavy resources (**19k–55k output tokens**, ~`$19` across 6 models just for tiny MinAtar networks)—since smaller or alternative LLMs do not perform that well (`GPT-6 Astra` `96.3` and `GPT-6.1 Sol` `86.5` vs. `Claude Sonnet 5.5` `23.2` and `DeepSeek V4.1 Flash` `0.7`; echoed by `WeightBench` `arXiv:2611.18553`).
    2. **Technical approach (the LLM as an "intelligent optimizer"):** From a technical approach, I can summarize what this study is doing as sending a request to an LLM with a list of numbers—the weights of a model—asking the LLM to understand that list of numbers with respect to the request, and in a next step modifying those numbers in the target model to improve a specific behavior requested in the prompt. Basically, they can pass the same list of weights and ask the LLM to fix a known issue by changing the weights, just as they could ask for any random request—and the LLM will still give an answer. Although the authors' intent can be summarized as building an **intelligent optimizer**—since they want to **(1)** understand the user request, **(2)** understand the network, and **(3)** optimize the weights to fulfill the user's request—in practice most gains rely on multi-turn trial-and-error against evaluation feedback rather than one-shot weight understanding.
    3. **Data-flow critique (the text-tokenization detour):** Following the flow of the data in practice, users start with floating-point numbers that are the weights of a network they want to optimize, convert those numbers into a text string, and pass that string through the tokenizer of the LLM—where the numbers in the string can easily be misunderstood. What if there are dots, commas, other punctuation, or text in between? Will the tokenizer conserve the intended numbers? Then from token IDs it goes back to vector embeddings, attention, and neural network layers. Many steps of this expensive process could be simplified considering that the goal is a conditioned optimization of floating-point numbers: it makes more sense to design a selective optimizer that skips the first steps, sees all the weights in their original numerical form, and through a conditioning input and attention mechanisms targets specific weights—while also avoiding SE.LLMA's sovereignty risk of depending on private foreign frontier APIs instead of sovereign gradient methods (`AdamW`, `Muon`, `LoRA`).
- **Awake Mapping Analogy:** This study of modifying weights in a neural network reminded me of neurosurgeon **Dr. Jesús Martín-Fernández** (`2022`, `2025`), who operates on brain tumors in awake patients. With the neural network as a brain, one can think that during training—where neurons establish their connections and learn—there is neuroplasticity as in a real brain: when modifying weights and allowing for fine-tuning, internal paths can be rewired so other neurons take charge of new tasks, at the risk of losing previous abilities. This neurosurgeon first establishes explicit constraints—abilities that must be kept—and maps the brain with small localized perturbations (Direct Electrical Stimulation, DES) to test how they affect the patient's abilities (*connectome-stop points*). In this analogy, when fine-tuning or optimizing a neural network, I first establish tests for the LLM abilities that I do not want to modify (or want affected the least), and only later start the optimization—whether testing different layers, using an "intelligent optimizer", or applying gradient-based optimizers and `LoRA`.

## 2. Playground (`notebooks/01_explore_models_cpu.py`)
With limited local hardware resources (`AMD Ryzen 7 5800U` CPU, `6.7 GiB` RAM, no GPU) and a 4-hour budget, I had to choose a small language model that could run and be modified locally. In `notebooks/01_explore_models_cpu.py` (`results/cpu_prompt_benchmark.json`), I built a playground to screen small candidate models on a Swedish prompt while tracking RAM, CPU usage, and throughput. I chose **`Qwen/Qwen2.5-0.5B-Instruct`** (`bfloat16`, `1842.4 MB` peak RSS, `5.08 tok/s`) because it runs comfortably within memory and already produces understandable Swedish, yet shows clear, concrete flaws to fix—such as grammar/agreement errors (*"ett tekniskt utvecklade system"*) and mid-sentence English code-switching (*"Siri, Apple's smart home assistant"*). That said, I could also have used the same workflow to try fixing the Swedish issues on the other candidate models included in the notebook (`HuggingFaceTB/SmolLM2-360M-Instruct`, `HuggingFaceTB/SmolLM2-135M-Instruct`, or `AI-Sweden-Models/gpt-sw3-126m`).

## 3. Hypothesis
In `Qwen/Qwen2.5-0.5B-Instruct`, I test three falsifiable claims:
- **H1 (Localization):** Data-free static weight alignment (`lm_head` → `mlp.down_proj`) does not strongly correlate with causal Swedish sensitivity measured by awake DES perturbations on `dev`.
- **H2 (Surgical Edits):** Gradient-free column scaling (`K=256`) fails to yield meaningful, stable held-out Swedish NLL gains over 3-seed random controls without risking English/reasoning stop-points.
- **H3 (Practical Alternatives):** Prompting and lightweight gradient adaptation (`Micro-LoRA`) offer a better quality, stability, and CPU-cost trade-off for SE.LLMA.
- *What would refute this:* Strong correlation (`r ≥ 0.6`) between static WSP and DES sensitivity, plus held-out Swedish NLL gains beating random, prompting, and Micro-LoRA with zero stop-point regression.

## 4. Experimental Design & Controls
- **Dataset (`data/swedish_test_suite.json`):** `32` prompts split into `12 dev` (for probing, calibration, and LoRA training) and `20 eval` (held-out). Half target Swedish (`everyday_sv`, `public_services_sv`) and half act as non-negotiable stop-points (`reasoning_sv`, `coding_sv`, `control_en`).
- **Six Arms (`src/wsp_experiment.py`):**
  1. **Unchanged Baseline** (`Qwen/Qwen2.5-0.5B-Instruct`).
  2. **Static WSP:** Data-free `lm_head` SV−EN direction → top-`256` `mlp.down_proj` columns scaled by `1.15×`.
  3. **Awake DES:** `0.85×` layer probing on `dev`, selecting the top-4 selective layers, aligning `256` columns to `dev` activations, and calibrating scale `α` under `ΔEN_dev ≤ 0.05`.
  4. **Random Edit:** `256` random columns scaled by `1.15×` across `3` seeds (`42, 123, 999`).
  5. **Prompting Control:** Strict Swedish system instruction with zero weight edits.
  6. **Micro-LoRA:** Frozen-base LoRA (`r=4`, `layers 18..23`, `138,240` params) trained for `18` steps on `dev`.

## 5. Results
- **CPU Screening (`results/cpu_prompt_benchmark.json`):** `Qwen/Qwen2.5-0.5B-Instruct` (`bfloat16`, `1.84 GB` peak RSS, `5.08 tok/s`) outperformed `SmolLM2-360M-Instruct` (`6.46 tok/s`, incoherent pseudo-Swedish) and showed two clear, measurable Swedish flaws: broken agreement (*"ett tekniskt utvecklade system"*) and mid-sentence English code-switching.
- **32-Prompt Baseline (`Qwen/Qwen2.5-0.5B-Instruct`):** Overall strict accuracy was **8/32 (25.0%)**.
  - *Swedish target (`3/16`, `18.8%`):* `0/8` on everyday Swedish and `3/8` on Swedish public services (`3/3` when facts were given in context, `0/5` from parametric memory).
  - *Non-negotiable stop-points (`5/16`, `31.2%`):* `3/4` on fluent English control, `1/6` on Swedish reasoning, and `1/6` on Swedish Python coding.
  - *Most common errors:* `grammar` (13), `truncation` (13), `hallucination` (8), `lexical_hallucination` (8), and `code_switching` (7).

- **Static vs. Awake Layer Localization:** Static `lm_head`→`mlp.down_proj` alignment did **not** match awake DES causal sensitivity (Pearson `r = -0.0747`, Spearman `ρ = -0.1527`; Top-4 DES layers: `[1, 22, 16, 19]`).

| Arm | SV Target NLL (Δ) | SV PPL | Non-Neg NLL | EN Stop NLL (Δ) | Params (L2) | CPU (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Unchanged Baseline | 2.6328 (+0.000) | 13.91 | 0.8383 | 0.8507 (+0.000) | 0 (0.00) | 130.93 |
| 2. Static Weight Edit (WSP) | 2.6334 (+0.001) | 13.92 | 0.8425 | 0.8513 (+0.001) | 229,376 (1.23) | 83.78 |
| 3. Awake DES Edit (Martín-Fernández) | 2.6353 (+0.003) | 13.95 | 0.8420 | 0.8397 (-0.011) | 229,376 (0.43) | 821.08 |
| 4. Random / Untargeted Edit (3 seeds) | 2.6365 ± 0.0013 (+0.004) | 13.96 | 0.8431 | 0.8661 ± 0.0120 (+0.015) | 229,376 (1.31) | 233.59 |
| 5a. Prompting / Inference Control | 2.6345 (+0.002) | 13.94 | 0.8513 | 0.8507 (+0.000) | 0 (0.00) | 89.41 |
| 5b. Micro-LoRA (layers 18..23, r=4) | 2.7075 (+0.075) | 14.99 | 0.8466 | 0.9020 (+0.051) | 138,240 (0.96) | 4115.96 |

![Baseline Diagnostic Summary](../results/baseline_diagnostic_summary.png)

![Static WSP vs. Awake DES Layer Mapping](../results/layer_mapping_static_vs_des.png)

![6-Arm Held-Out NLL Comparison](../results/arm_comparison_nll.png)

## 6. Interpretation, Limitations & Recommendation
- **Static and surgical edits gave zero held-out gain:** Static WSP (`Δ=+0.001`) and Awake DES (`Δ=+0.003`) performed no better than 3-seed Random edits (`Δ=+0.004`). However, DES stop-point calibration protected English fluency (`ΔEN=-0.011` vs. Random `+0.015`).
- **Unbricking vs. new language skill:** Strict prompting barely changed NLL (`ΔSV=+0.002`). Meanwhile, `12`-shot `dev` Micro-LoRA overfit ungrounded Swedish (`ΔSV=+0.075`, `ΔEN=+0.051`) yet improved **4/4** held-out `coding_sv` prompts (`Δ=-0.056`)—confirming that small edits/tuning can *unbrick* existing skills (Python) but cannot conjure missing Swedish fluency.
- **Limitations:** 4-hour CPU scope (`~91 min` for the `--nll-only` run), `0.5B` model scale, `32`-prompt suite, and teacher-forced NLL evaluation across the 6 arms.
- **Recommendation for SE.LLMA:** **Do not pursue LLM-driven or data-free static weight-space programming** for training or repairing Swedish models; focus instead on sovereign continued pretraining, `LoRA`/`Muon` fine-tuning, and **awake stop-point diagnostic suites**. *Next experiment:* Test awake DES stop-point selectivity against selective vs. full LoRA on a 7B–14B Swedish checkpoint using ScandEval/EuroEval plus English/code stop-points.

## 7. Reproducibility & AI Usage
- **Reproduce:** `uv sync && uv run python -m src.wsp_experiment --nll-only && uv run python -m src.report_builder`
- **AI Usage:** LLM coding/research assistants helped synthesize replication benchmarks, scaffold `marimo` notebooks and `src/` modules, and draft reference answers/error tags; all experimental controls, code runs, and numbers were verified by the author.
- **Key References:** Takehi et al. (*Sparks of Weight-Space Programming*); `eomcaleb/weight-space-bench`; Marasini et al. (`WeightBench`, `arXiv:2611.18553`); Martín-Fernández et al. (`2022` PMID `35624947`, `2025` PMID `40110086`).
