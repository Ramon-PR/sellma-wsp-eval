import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="WSP vs. Awake DES Mapping & Micro-LoRA — SE.LLMA",
)


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.wsp_experiment import run_experiment

    results_dir = root_dir / "results"
    details_path = results_dir / "experiment_details.json"
    return details_path, json, mo, os, results_dir, run_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Experiment 2 — Static WSP vs. Awake DES Mapping, Prompting & Micro-LoRA

    This notebook executes and visualizes the 6-arm CPU experiment on **`Qwen/Qwen2.5-0.5B-Instruct`** (`bfloat16`),
    strictly separating `dev` (12 prompts used for DES layer probing, `alpha` calibration, and Micro-LoRA training)
    from held-out `eval` (20 prompts):

    1. **Unchanged Baseline**
    2. **Static Weight Edit (WSP)** — data-free `lm_head` → `mlp.down_proj` alignment (`K=256` columns)
    3. **Awake DES Edit (Martín-Fernández analogy)** — layer-wise causal perturbation probing on `dev` under stop-point constraint
    4. **Random / Untargeted Edit** — 3 seeds (`42, 123, 999`), matched parameter count and magnitude
    5. **Prompting / Inference Control** — strict Swedish system prompt (no weight edits)
    6. **Micro-LoRA (`layers 18..23`, `r=4`)** — lightweight gradient baseline on `dev` (`138,240` trainable params)
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    _cpu_count = os.cpu_count() or 4

    mode_select = mo.ui.dropdown(
        options={
            "Fast NLL + Layer Mapping (~2-3 min)": "nll_only",
            "Full (NLL + 20 eval generations ~18 min)": "full",
        },
        value="Fast NLL + Layer Mapping (~2-3 min)",
        label="Execution mode",
    )
    threads_slider = mo.ui.slider(
        start=1,
        stop=_cpu_count,
        step=1,
        value=min(8, _cpu_count),
        label="PyTorch CPU threads",
    )
    seed_input = mo.ui.number(
        start=0,
        stop=999999,
        step=1,
        value=42,
        label="Main random seed",
    )
    run_btn = mo.ui.run_button(label="Run 6-arm experiment")

    mo.vstack(
        [
            mo.hstack([mode_select, threads_slider, seed_input], widths="equal"),
            run_btn,
        ]
    )
    return mode_select, run_btn, seed_input, threads_slider


@app.cell(hide_code=True)
def _(
    details_path,
    json,
    mo,
    mode_select,
    results_dir,
    run_btn,
    run_experiment,
    seed_input,
    threads_slider,
):
    _run_err = None
    if run_btn.value:
        try:
            with mo.status.spinner(title="Running 6-arm experiment on CPU..."):
                run_experiment(
                    nll_only=(mode_select.value == "nll_only"),
                    num_threads=int(threads_slider.value),
                    seed=int(seed_input.value),
                )
        except Exception as _exc:
            _run_err = f"{type(_exc).__name__}: {_exc}"

    if _run_err:
        experiment_view = mo.callout(mo.md(f"**Experiment failed:** `{_run_err}`"), kind="danger")
    elif not details_path.exists():
        experiment_view = mo.callout(
            mo.md(
                "No `results/experiment_details.json` found yet. Click **Run 6-arm experiment** above "
                "(or run `uv run python -m src.wsp_experiment --nll-only` in a terminal)."
            ),
            kind="info",
        )
    else:
        _details = json.loads(details_path.read_text(encoding="utf-8"))
        _meta = _details.get("metadata", {})
        _mapping = _details.get("layer_mapping", {})
        _corr = _mapping.get("static_vs_des_correlation", {})
        _arms = _details.get("arms", {})

        _arm_keys = [
            "baseline",
            "wsp_static",
            "des_awake_edit",
            "random_edit",
            "prompt_control",
            "micro_lora",
        ]
        _base_sv = _arms.get("baseline", {}).get("eval_nll", {}).get("sv_target_nll", 0.0)
        _base_en = _arms.get("baseline", {}).get("eval_nll", {}).get("en_ctrl_nll", 0.0)

        _rows = [
            "| Arm | SV Target NLL | Δ SV NLL | SV PPL | Non-Neg NLL | EN Stop-Point NLL | Δ EN NLL | Params (L2 norm) | Runtime |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _k in _arm_keys:
            if _k not in _arms:
                continue
            _arm = _arms[_k]
            _ev = _arm.get("eval_nll", {})
            _sv = _ev.get("sv_target_nll", 0.0)
            _en = _ev.get("en_ctrl_nll", 0.0)
            _std = f" ± {_arm['sv_target_nll_std']}" if "sv_target_nll_std" in _arm else ""
            _rows.append(
                f"| {_arm.get('name')} | {_sv}{_std} | {_sv - _base_sv:+.3f} | "
                f"{_ev.get('sv_target_ppl', '-')} | {_ev.get('sv_non_neg_nll', '-')} | "
                f"{_en} | {_en - _base_en:+.3f} | "
                f"{_arm.get('edited_params', 0):,} ({_arm.get('edit_l2_norm', 0.0)}) | {_arm.get('runtime_s', '-')}s |"
            )

        _lora_curve = _arms.get("micro_lora", {}).get("loss_curve", [])
        _lora_txt = (
            f"`{_lora_curve[0]} → {_lora_curve[-1]}` (18 steps on `dev`)"
            if _lora_curve
            else "N/A"
        )

        _summary_md = mo.md(
            "\n".join(
                [
                    "## Results Overview",
                    f"- **Model:** `{_meta.get('model_id')}` (revision `{_meta.get('revision')}`)",
                    f"- **Config:** `dtype={_meta.get('dtype')}` · `seed={_meta.get('seed')}` · "
                    f"`threads={_meta.get('torch_threads')}` · `nll_only={_meta.get('nll_only')}` · "
                    f"peak_RSS=`{_meta.get('peak_rss_mb')} MB`",
                    f"- **Static WSP vs. Awake DES Layer Correlation:** "
                    f"Pearson `r = {_corr.get('pearson_r')}`, Spearman `ρ = {_corr.get('spearman_rho')}` · "
                    f"Top-4 DES Layers: `{_mapping.get('selected_des_layers')}`",
                    f"- **Micro-LoRA Training Loss (`dev`):** {_lora_txt}",
                    "",
                    *_rows,
                ]
            )
        )

        _img_views = []
        for _fname, _cap in (
            ("layer_mapping_static_vs_des.png", "Static WSP vs. Awake DES Layer Mapping"),
            ("arm_comparison_nll.png", "Held-Out Target Improvement vs. Stop-Point Regression"),
        ):
            _p = results_dir / _fname
            if _p.exists():
                _img_views.append(mo.image(src=str(_p), caption=_cap))

        experiment_view = mo.vstack([_summary_md, *_img_views])

    experiment_view
    return


if __name__ == "__main__":
    app.run()
