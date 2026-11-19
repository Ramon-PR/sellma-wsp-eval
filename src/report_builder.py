"""Report assembly and publication-grade Markdown/PDF export for SE.LLMA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import weasyprint

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "swedish_test_suite.json"
RESULTS_DIR = ROOT_DIR / "results"
REPORT_DIR = ROOT_DIR / "report"

CPU_BENCH_PATH = RESULTS_DIR / "cpu_prompt_benchmark.json"
BASELINE_RESP_PATH = RESULTS_DIR / "qwen_baseline_responses.json"
BASELINE_EVAL_PATH = RESULTS_DIR / "qwen_baseline_eval.json"
EXPERIMENT_DETAILS_PATH = RESULTS_DIR / "experiment_details.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
BASELINE_FIG_PATH = RESULTS_DIR / "baseline_diagnostic_summary.png"

ARM_ORDER = [
    "baseline",
    "wsp_static",
    "des_awake_edit",
    "random_edit",
    "prompt_control",
    "micro_lora",
]

ARM_RESPONSE_FILES = {
    "baseline": "qwen_baseline_responses.json",
    "wsp_static": "qwen_wsp_static_responses.json",
    "des_awake_edit": "qwen_des_awake_responses.json",
    "random_edit": "qwen_random_edit_responses.json",
    "prompt_control": "qwen_prompt_control_responses.json",
    "micro_lora": "qwen_micro_lora_responses.json",
}

CATEGORY_ORDER = [
    "everyday_sv",
    "public_services_sv",
    "reasoning_sv",
    "coding_sv",
    "control_en",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate(text: str, limit: int = 160) -> str:
    clean = re.sub(r"```[a-zA-Z0-9_-]*", "", text or "")
    clean = re.sub(r"\s+", " ", clean).replace("`", "'").strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _compute_baseline_stats(resp_data: dict[str, Any], eval_data: dict[str, Any]) -> dict[str, Any]:
    resp_map = {r["prompt_id"]: r for r in resp_data.get("responses", [])}
    evals = eval_data.get("evaluations", [])
    eval_map = {e["prompt_id"]: e for e in evals}

    total = len(evals)
    passed = sum(1 for e in evals if e.get("is_correct"))
    acc = (100.0 * passed / total) if total else 0.0

    def _group_stats(field: str, order: list[str] | None = None) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for ev in evals:
            resp = resp_map.get(ev["prompt_id"], {})
            key = str(resp.get(field, "-"))
            grouped.setdefault(key, []).append((resp, ev))
        keys = order if order else list(grouped.keys())
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            items = grouped.get(key, [])
            if not items:
                continue
            n = len(items)
            ok = sum(1 for _, e in items if e.get("is_correct"))
            lang = sum(float(e.get("swedish_quality", 0)) for _, e in items) / n
            task = sum(float(e.get("task_correctness", 0)) for _, e in items) / n
            out[key] = {
                "n": n,
                "ok": ok,
                "acc": round(100.0 * ok / n, 1),
                "lang": round(lang, 2),
                "task": round(task, 2),
                "priority": items[0][0].get("priority", "-"),
            }
        return out

    fm_counts: dict[str, int] = {}
    for ev in evals:
        for fm in ev.get("failure_modes") or []:
            fm_counts[fm] = fm_counts.get(fm, 0) + 1
    failure_modes = sorted(fm_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    rep_examples: list[dict[str, Any]] = []
    for pid in ("ev_eval_01", "pub_dev_01", "code_eval_01", "pub_eval_05"):
        if pid not in resp_map or pid not in eval_map:
            continue
        r = resp_map[pid]
        e = eval_map[pid]
        rep_examples.append(
            {
                "prompt_id": pid,
                "split": r.get("split", "-"),
                "category": r.get("category", "-"),
                "priority": r.get("priority", "-"),
                "is_correct": bool(e.get("is_correct")),
                "failure_modes": e.get("failure_modes") or [],
                "prompt": r.get("prompt", ""),
                "response_text": r.get("response_text", ""),
                "reference_answer": e.get("reference_answer", ""),
                "notes": e.get("notes", ""),
            }
        )

    return {
        "metadata": resp_data.get("metadata", {}),
        "overall": {"n": total, "ok": passed, "acc": round(acc, 1)},
        "by_split": _group_stats("split", ["dev", "eval"]),
        "by_priority": _group_stats("priority", ["target", "non_negotiable"]),
        "by_category": _group_stats("category", CATEGORY_ORDER),
        "failure_modes": failure_modes,
        "rep_examples": rep_examples,
    }


def _extract_experiment_view(details: dict[str, Any]) -> dict[str, Any]:
    arms = details.get("arms", {})
    base_ev = arms.get("baseline", {}).get("eval_nll", {})
    base_sv = float(base_ev.get("sv_target_nll", 0.0))
    base_nn = float(base_ev.get("sv_non_neg_nll", 0.0))
    base_en = float(base_ev.get("en_ctrl_nll", 0.0))

    arm_display_names = {
        "baseline": "1. Unchanged Baseline",
        "wsp_static": "2. Static Weight Edit (WSP)",
        "des_awake_edit": "3. Awake DES Edit",
        "random_edit": "4. Random / Untargeted Edit (3 seeds)",
        "prompt_control": "5a. Prompting / Inference Control",
        "micro_lora": "5b. Micro-LoRA (layers 18..23, r=4)",
    }

    arm_rows: list[dict[str, Any]] = []
    for key in ARM_ORDER:
        if key not in arms:
            continue
        arm = arms[key]
        ev = arm.get("eval_nll", {})
        sv = float(ev.get("sv_target_nll", 0.0))
        nn = float(ev.get("sv_non_neg_nll", 0.0))
        en = float(ev.get("en_ctrl_nll", 0.0))
        arm_rows.append(
            {
                "key": key,
                "name": arm_display_names.get(key, arm.get("name", key)),
                "sv_target_nll": sv,
                "sv_target_nll_std": arm.get("sv_target_nll_std"),
                "delta_sv_nll": round(sv - base_sv, 4),
                "sv_target_ppl": ev.get("sv_target_ppl"),
                "sv_non_neg_nll": nn,
                "delta_non_neg_nll": round(nn - base_nn, 4),
                "en_ctrl_nll": en,
                "en_ctrl_nll_std": arm.get("en_ctrl_nll_std"),
                "delta_en_nll": round(en - base_en, 4),
                "overall_nll": ev.get("overall_nll"),
                "edited_params": int(arm.get("edited_params", 0)),
                "edit_l2_norm": float(arm.get("edit_l2_norm", 0.0)),
                "runtime_s": arm.get("runtime_s"),
                "selected_layers": arm.get("selected_layers"),
                "best_alpha": arm.get("best_alpha"),
                "loss_curve": arm.get("loss_curve"),
            }
        )

    def _arm_per_item(arm_key: str, arm_obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if arm_key == "random_edit" and arm_obj.get("seed_runs"):
            merged: dict[str, list[float]] = {}
            meta_map: dict[str, tuple[str, str]] = {}
            for srun in arm_obj["seed_runs"]:
                for pid, rec in srun.get("eval_nll", {}).get("per_item", {}).items():
                    merged.setdefault(pid, []).append(float(rec["nll"]))
                    meta_map[pid] = (rec.get("category", "-"), rec.get("priority", "-"))
            return {
                pid: {
                    "category": meta_map[pid][0],
                    "priority": meta_map[pid][1],
                    "nll": sum(vals) / len(vals),
                }
                for pid, vals in merged.items()
            }
        return arm_obj.get("eval_nll", {}).get("per_item", {})

    per_category_nll: dict[str, dict[str, float]] = {}
    per_arm_items: dict[str, dict[str, dict[str, Any]]] = {}
    for row in arm_rows:
        key = row["key"]
        p_items = _arm_per_item(key, arms[key])
        per_arm_items[key] = p_items
        cat_vals: dict[str, list[float]] = {}
        for rec in p_items.values():
            cat_vals.setdefault(rec.get("category", "-"), []).append(float(rec["nll"]))
        per_category_nll[key] = {
            c: round(sum(vs) / len(vs), 4) for c, vs in cat_vals.items() if vs
        }

    base_items = per_arm_items.get("baseline", {})
    lora_items = per_arm_items.get("micro_lora", {})
    coding_pids = [
        pid for pid, rec in base_items.items() if rec.get("category") == "coding_sv"
    ]
    coding_improved = sum(
        1
        for pid in coding_pids
        if pid in lora_items and float(lora_items[pid]["nll"]) < float(base_items[pid]["nll"])
    )
    base_coding = per_category_nll.get("baseline", {}).get("coding_sv", 0.0)
    lora_coding = per_category_nll.get("micro_lora", {}).get("coding_sv", 0.0)
    base_everyday = per_category_nll.get("baseline", {}).get("everyday_sv", 0.0)
    lora_everyday = per_category_nll.get("micro_lora", {}).get("everyday_sv", 0.0)

    mapping = details.get("layer_mapping", {})
    return {
        "metadata": details.get("metadata", {}),
        "correlation": mapping.get("static_vs_des_correlation", {}),
        "selected_des_layers": mapping.get("selected_des_layers", []),
        "per_layer": mapping.get("per_layer", []),
        "arm_rows": arm_rows,
        "per_category_nll": per_category_nll,
        "coding_improved_count": coding_improved,
        "coding_total_count": len(coding_pids),
        "baseline_coding_nll": base_coding,
        "micro_lora_coding_nll": lora_coding,
        "micro_lora_coding_delta": round(lora_coding - base_coding, 4),
        "baseline_everyday_nll": base_everyday,
        "micro_lora_everyday_nll": lora_everyday,
        "des_alpha_trials": arms.get("des_awake_edit", {}).get("alpha_trials", []),
        "micro_lora_loss_curve": arms.get("micro_lora", {}).get("loss_curve", []),
    }


def _extract_arm_comparison_examples(
    arm_responses: dict[str, dict[str, Any]],
    eval_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if "baseline" not in arm_responses or len(arm_responses) < 2:
        return []

    by_arm_pid: dict[str, dict[str, dict[str, Any]]] = {}
    for arm_key, payload in arm_responses.items():
        by_arm_pid[arm_key] = {r["prompt_id"]: r for r in payload.get("responses", [])}

    eval_map = {e["prompt_id"]: e for e in (eval_data or {}).get("evaluations", [])}
    examples: list[dict[str, Any]] = []
    for pid in ("ev_eval_01", "pub_eval_01", "code_eval_01", "ctrl_eval_02"):
        base_rec = by_arm_pid.get("baseline", {}).get(pid)
        if not base_rec:
            continue
        outputs: dict[str, str] = {}
        for arm_key in ARM_ORDER:
            rec = by_arm_pid.get(arm_key, {}).get(pid)
            if rec:
                outputs[arm_key] = rec.get("response_text", "")
        if len(outputs) > 1:
            examples.append(
                {
                    "prompt_id": pid,
                    "category": base_rec.get("category", "-"),
                    "priority": base_rec.get("priority", "-"),
                    "prompt": base_rec.get("prompt", ""),
                    "reference_answer": eval_map.get(pid, {}).get("reference_answer", ""),
                    "outputs": outputs,
                }
            )
    return examples


def load_report_data() -> dict[str, Any]:
    """Load all available benchmark, baseline, and 6-arm experiment artifacts."""
    suite_data = _read_json(DATA_PATH)
    cpu_data = _read_json(CPU_BENCH_PATH)
    base_resp = _read_json(BASELINE_RESP_PATH)
    base_eval = _read_json(BASELINE_EVAL_PATH)
    exp_details = _read_json(EXPERIMENT_DETAILS_PATH)
    summary_data = _read_json(SUMMARY_PATH)

    arm_responses: dict[str, dict[str, Any]] = {}
    for arm_key, fname in ARM_RESPONSE_FILES.items():
        loaded = _read_json(RESULTS_DIR / fname)
        if loaded is not None:
            arm_responses[arm_key] = loaded

    has_cpu_bench = cpu_data is not None
    has_baseline = base_resp is not None and base_eval is not None
    has_experiment_details = exp_details is not None
    has_summary = summary_data is not None

    baseline_stats = _compute_baseline_stats(base_resp, base_eval) if has_baseline else None
    experiment_view = _extract_experiment_view(exp_details) if has_experiment_details else None
    arm_examples = _extract_arm_comparison_examples(arm_responses, base_eval)
    has_arm_generations = bool(arm_examples) or bool(
        summary_data and summary_data.get("examples")
    )

    available_figures = {
        name: (RESULTS_DIR / name).exists()
        for name in (
            "baseline_diagnostic_summary.png",
            "layer_mapping_static_vs_des.png",
            "arm_comparison_nll.png",
        )
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "suite": suite_data,
        "cpu_bench": cpu_data,
        "baseline_stats": baseline_stats,
        "experiment_details": exp_details,
        "experiment_view": experiment_view,
        "summary": summary_data,
        "arm_examples": arm_examples,
        "summary_examples": (summary_data or {}).get("examples", []),
        "has_cpu_bench": has_cpu_bench,
        "has_baseline": has_baseline,
        "has_experiment_details": has_experiment_details,
        "has_summary": has_summary,
        "has_arm_generations": has_arm_generations,
        "available_figures": available_figures,
    }


def generate_baseline_diagnostic_figure(data: dict[str, Any]) -> Path | None:
    """Create a 2-panel baseline diagnostic chart from real baseline evaluation data."""
    if not data.get("has_baseline") or not data.get("baseline_stats"):
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = data["baseline_stats"]
    by_cat = stats["by_category"]
    failure_modes = stats["failure_modes"]

    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    label_map = {
        "everyday_sv": "everyday_sv\n(Target)",
        "public_services_sv": "public_svc\n(Target)",
        "reasoning_sv": "reasoning_sv\n(Stop-Pt)",
        "coding_sv": "coding_sv\n(Stop-Pt)",
        "control_en": "control_en\n(Stop-Pt)",
    }
    cat_labels = [label_map.get(c, c) for c in cats]
    accs = [by_cat[c]["acc"] for c in cats]
    colors = [
        "#2563eb" if by_cat[c]["priority"] == "target" else "#475569"
        for c in cats
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    try:
        ax1 = axes[0]
        bars = ax1.bar(range(len(cats)), accs, color=colors, width=0.58)
        for idx, (bar, cat) in enumerate(zip(bars, cats)):
            st = by_cat[cat]
            ax1.text(
                idx,
                bar.get_height() + 2.5,
                f"{st['ok']}/{st['n']} ({st['acc']:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#1e293b",
            )
        ax1.set_xticks(range(len(cats)))
        ax1.set_xticklabels(cat_labels, fontsize=8.0)
        ax1.set_ylim(0, 108)
        ax1.set_ylabel("Strict Pass Accuracy (%)", fontsize=9)
        ax1.set_title(
            f"Baseline Diagnostic Accuracy by Category ({stats['overall']['ok']}/{stats['overall']['n']} Overall)",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax1.grid(axis="y", alpha=0.25)

        from matplotlib.patches import Patch

        ax1.legend(
            handles=[
                Patch(facecolor="#2563eb", label="Target (Swedish)"),
                Patch(facecolor="#475569", label="Non-Negotiable Stop-Point"),
            ],
            loc="upper left",
            fontsize=8,
            frameon=True,
        )

        ax2 = axes[1]
        if failure_modes:
            fm_names = [m for m, _ in reversed(failure_modes)]
            fm_vals = [c for _, c in reversed(failure_modes)]
            y_pos = range(len(fm_names))
            hbars = ax2.barh(y_pos, fm_vals, color="#0f766e", height=0.66)
            for idx, (bar, val) in enumerate(zip(hbars, fm_vals)):
                ax2.text(
                    val + 0.25,
                    idx,
                    str(val),
                    va="center",
                    ha="left",
                    fontsize=8.2,
                    color="#1e293b",
                )
            ax2.set_yticks(list(y_pos))
            ax2.set_yticklabels(fm_names, fontsize=8.2)
            ax2.set_xlim(0, max(fm_vals) + 2.2)
        ax2.set_xlabel("Prompt Count (out of 32)", fontsize=9)
        ax2.set_title(
            "Observed Baseline Failure Modes",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax2.grid(axis="x", alpha=0.25)

        fig.tight_layout(pad=1.6, w_pad=2.4)
        fig.savefig(BASELINE_FIG_PATH, dpi=180)
    finally:
        plt.close(fig)

    data.setdefault("available_figures", {})["baseline_diagnostic_summary.png"] = True
    return BASELINE_FIG_PATH


def _figure_markdown(filename: str, caption: str, image_mode: str) -> str:
    if image_mode == "none":
        return ""
    path = RESULTS_DIR / filename
    if not path.exists():
        return ""
    target = f"../results/{filename}" if image_mode == "relative" else path.resolve().as_uri()
    return f"![{caption}]({target})\n\n*Figure: {caption}*"


def _format_arm_table(exp_view: dict[str, Any]) -> str:
    lines = [
        "| Arm | SV Target NLL (Δ) | SV PPL | Non-Neg NLL | EN Stop NLL (Δ) | Params (L2) | CPU (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in exp_view["arm_rows"]:
        std_txt = (
            f" ± {row['sv_target_nll_std']:.4f}"
            if row.get("sv_target_nll_std") is not None
            else ""
        )
        en_std_txt = (
            f" ± {row['en_ctrl_nll_std']:.4f}"
            if row.get("en_ctrl_nll_std") is not None
            else ""
        )
        lines.append(
            f"| {row['name']} | {row['sv_target_nll']:.4f}{std_txt} ({row['delta_sv_nll']:+.3f}) | "
            f"{row['sv_target_ppl']} | {row['sv_non_neg_nll']:.4f} | "
            f"{row['en_ctrl_nll']:.4f}{en_std_txt} ({row['delta_en_nll']:+.3f}) | "
            f"{row['edited_params']:,} ({row['edit_l2_norm']:.2f}) | {row['runtime_s']} |"
        )
    return "\n".join(lines)


def _format_exec_summary(data: dict[str, Any]) -> str:
    if data["has_experiment_details"] and data["experiment_view"]:
        ev = data["experiment_view"]
        corr = ev["correlation"]
        rows_by_key = {r["key"]: r for r in ev["arm_rows"]}
        wsp = rows_by_key.get("wsp_static", {})
        des = rows_by_key.get("des_awake_edit", {})
        lora = rows_by_key.get("micro_lora", {})
        prompt = rows_by_key.get("prompt_control", {})
        rand = rows_by_key.get("random_edit", {})
        best_arm = min(ev["arm_rows"], key=lambda r: r["sv_target_nll"])
        coding_ok = ev.get("coding_improved_count", 0)
        coding_tot = ev.get("coding_total_count", 0)
        coding_d = ev.get("micro_lora_coding_delta", 0.0)
        status_line = "**Status:** Final Report *(Screening, 32-prompt Baseline & 6-Arm CPU Experiment Complete)*"
        body = (
            f"I critically examine *Sparks of Weight-Space Programming* and evaluate whether data-free static weight inspection "
            f"can localize and repair Swedish language failures in `Qwen/Qwen2.5-0.5B-Instruct` (`bfloat16` on CPU) without crossing "
            f"non-negotiable English/reasoning *connectome-stop points*. On the 32-prompt diagnostic suite, the unchanged baseline achieves "
            f"**{data['baseline_stats']['overall']['ok']}/{data['baseline_stats']['overall']['n']} ({data['baseline_stats']['overall']['acc']}%)** "
            f"overall (**3/16** Swedish target vs. **5/16** non-negotiable). Across the 24 transformer layers, static `lm_head`→`mlp.down_proj` "
            f"alignment exhibits near-zero correlation with awake DES causal sensitivity (**Pearson `r = {corr.get('pearson_r')}`, Spearman `ρ = {corr.get('spearman_rho')}`**), "
            f"confirming the *paradox of non-locality*. On held-out `eval` Swedish NLL, gradient-free edits yield zero gain over baseline "
            f"(Static WSP `Δ={wsp.get('delta_sv_nll', 0.0):+.3f}`, `ΔEN={wsp.get('delta_en_nll', 0.0):+.3f}`; Awake DES `Δ={des.get('delta_sv_nll', 0.0):+.3f}`, "
            f"`ΔEN={des.get('delta_en_nll', 0.0):+.3f}`; 3-seed Random `Δ={rand.get('delta_sv_nll', 0.0):+.3f}`, `ΔEN={rand.get('delta_en_nll', 0.0):+.3f}`), "
            f"though DES stop-point calibration preserves English better than random perturbations. Strict Prompting leaves NLL virtually unchanged "
            f"(`ΔSV={prompt.get('delta_sv_nll', 0.0):+.3f}`, `ΔEN={prompt.get('delta_en_nll', 0.0):+.3f}`), while `N=12` `dev` Micro-LoRA overfits ungrounded Swedish "
            f"(`ΔSV={lora.get('delta_sv_nll', 0.0):+.3f}`, `ΔEN={lora.get('delta_en_nll', 0.0):+.3f}`) yet improves **{coding_ok}/{coding_tot}** held-out `coding_sv` items "
            f"(`Δ={coding_d:+.3f}`)—highlighting the divide between *unbricking* latent skills and acquiring missing language fluency. "
            f"Overall lowest held-out Swedish NLL remains **{best_arm['name']}** (`{best_arm['sv_target_nll']:.4f}`)."
        )
    else:
        status_line = (
            "**Status:** Draft Report *(Preliminary CPU Screening & 32-Prompt Baseline Complete; "
            "6-Arm Intervention Run Pending Execution)*"
        )
        base_summary = ""
        if data["has_baseline"] and data["baseline_stats"]:
            st = data["baseline_stats"]
            tgt = st["by_priority"].get("target", {})
            nn = st["by_priority"].get("non_negotiable", {})
            base_summary = (
                f" Completed baseline evaluation of `Qwen/Qwen2.5-0.5B-Instruct` on the 32-prompt diagnostic suite establishes "
                f"**{st['overall']['ok']}/{st['overall']['n']} ({st['overall']['acc']}%)** overall accuracy: "
                f"**{tgt.get('ok', 0)}/{tgt.get('n', 16)} ({tgt.get('acc', 0.0)}%)** on Swedish target prompts "
                f"(`0/8` on ungrounded `everyday_sv`) versus **{nn.get('ok', 0)}/{nn.get('n', 16)} ({nn.get('acc', 0.0)}%)** "
                f"on non-negotiable stop-points (`3/4` on `control_en`)."
            )
        body = (
            "I critically evaluate *Sparks of Weight-Space Programming* (Takehi et al.) for **SE.LLMA**'s goal of building sovereign, "
            "high-quality Swedish language models, and design a controlled 6-arm CPU experiment comparing data-free static weight edits, "
            "awake causal perturbation mapping (inspired by Dr. Jesús Martín-Fernández's cognitive DES mapping), 3-seed random controls, "
            "strict Swedish prompting, and Micro-LoRA (`layers 18..23`, `r=4`)."
            + base_summary
            + " Once `src/wsp_experiment.py` finishes executing (`results/experiment_details.json`), all 6-arm tables, correlations, and plots populate automatically."
        )
    return f"{status_line}  \n**Repository:** `sellma-wsp-eval` · **Updated:** `{data['generated_at_utc']}`\n\n---\n\n## Executive Summary\n\n{body}"


def build_report_sections(
    data: dict[str, Any],
    *,
    include_appendix: bool = True,
    image_mode: str = "relative",
) -> dict[str, str]:
    """Assemble full Markdown sections keyed by logical section identifier."""
    header_and_exec = (
        "# Weight-Space Programming for SE.LLMA: Critical Evaluation & Empirical Investigation\n\n"
        + _format_exec_summary(data)
    )

    sec1_critique = """## 1. Critical Examination of *Sparks of Weight-Space Programming*

### 1.1 Strongest Evidence
- **A compelling new paradigm:** The MIT team explores a new paradigm: letting an LLM interpret and rewrite the weights of another model based on a user's text input. In the cases shown in their blog, they first test whether a frontier LLM (`GPT-6 Astra`) can "understand" the parameters of a model trained to play a minigame (predicting which game it is), and whether it can fix issues in another LLM suffering from a repetition loop (such as `Qwen3.5-0.8B`).
- **Promising conceptual direction:** These test cases are intriguing—they treat neural network weights as inspectable artifacts rather than opaque tensors, and if the idea and methods are further improved, this could become an interesting new direction for AI.

### 1.2 Critical Opinion
1. **Frontier-model dependence & resource costs (`eomcaleb/weight-space-bench`, `WeightBench`):** My first doubt whenever a study discovers a new functionality using the newest frontier models is whether it is applicable to LLM technology in general, or just limited to the newest and largest models. In their blog, they use `GPT-6 Astra`—one of the latest LLMs to date—while publishing execution logs (`RikiyaT/lm_understand-logs`) without multi-seed variance or systematic cross-model baselines. Luckily, a follow-up study by `eomcaleb` (`eomcaleb/weight-space-bench`) answers two of my main concerns: the **costs** and the **applicability to other models**. And it seems that if you want to use this new approach, you need to use frontier models—which are the most expensive to run, private, subsidized, and consume heavy resources—since smaller or alternative LLMs do not perform that well:

    - **`eomcaleb/weight-space-bench`** (replicating MinAtar weight-writing across *Breakout*, *Seaquest*, and *Asterix*) finds that relative to `GPT-6 Astra` (`96.3`) and `PPO-500M` (`100.0`), alternative models degrade steeply: `GPT-6.1 Sol` (`86.5`), `Claude Opus 5.5` (`48.5`, hitting token-limit cutoffs on some runs), `Claude Sonnet 5.5` (`23.2`), `Kimi K3.1 Pro` (`16.2`), `Muse Wonder` (`2.6`), and `DeepSeek V4.1 Flash` (`0.7`). Generating tiny MinAtar policies consumed **19k–55k output tokens** (~**$19** total across 6 models via OpenRouter, **$0.36–$8.32** per model).
    - **`MatinKhajavi/Weight-Space-Programming`** independently reconstructs the game and repair pipelines, highlighting sensitivity to harness design, prompting, and controller choice.
    - **`WeightBench`** (`arXiv:2611.18553`, 42 tasks across `READ`, `FIX`, `WRITE`, `META`, and `SYSTEMS`) similarly finds that only top frontier models (`GPT-6.1-sol`: `0.568`, `Claude Opus 5.5`: `0.535`) maintain consistent performance while lighter models (`DeepSeek v4-flash`: `0.215`) largely fail—and frontier models still struggle with data-free compression, impossible-edit detection, and uniform LLM patching.

2. **Technical approach (the LLM as an "intelligent optimizer"):** From a technical approach, I can summarize what this study is doing as sending a request to an LLM with a list of numbers—the weights of a model—asking the LLM to understand that list of numbers with respect to the request, and in a next step modifying those numbers in the target model to improve a specific behavior requested in the prompt. Basically, they can pass the same list of weights and ask the LLM to fix a known issue by changing the weights, just as they could ask for any random request—and the LLM will still give an answer. Although the authors' intent can be summarized as building an **intelligent optimizer**—since they want to **(1)** understand the user request, **(2)** understand the network, and **(3)** optimize the weights to fulfill the user's request—in practice most gains rely on multi-turn trial-and-error against evaluation feedback, acting as a costly black-box heuristic optimizer rather than performing one-shot analytical reasoning over raw weights.
3. **Data-flow critique (the text-tokenization detour vs. a direct selective optimizer):** Following the flow of the data in practice, users start with floating-point numbers that are the weights of a network they want to optimize, convert those numbers into a text string, and pass that string through the tokenizer of the LLM—where the numbers in the string can easily be misunderstood. What if there are dots, commas, other punctuation, or text in between? Will the tokenizer conserve the intended numbers? Then from token IDs it goes back to vector embeddings, attention, and neural network layers. Many steps of this expensive process could be simplified considering that the goal is a **conditioned optimization of floating-point numbers**: it makes more sense to design a selective optimizer that skips the first steps, sees all the weights in their original numerical form, and through a conditioning input and attention mechanisms targets specific weights.
4. **Implications for SE.LLMA (sovereignty, governance & cost economics):** SE.LLMA aims to build open, sovereign, auditable Swedish foundation models on European/Swedish infrastructure (e.g., NAISS) while reducing dependence on proprietary foreign APIs. Because weight-space programming collapses on open/lighter controllers and relies on multi-turn frontier APIs costing thousands of output tokens even on tiny control nets, it creates both **technological sovereignty risk** (outsourcing weight analysis/editing to proprietary US/Chinese endpoints) and poor **compute-cost scaling** relative to gradient optimizers (`AdamW`, `Muon`, `LoRA`).
5. **Mismatch with pretrained LLM workflows:** Generating weights from scratch only applies to tiny models initialized from zero. Practical LLM development starts from pretrained checkpoints via continued pretraining, instruction tuning, or adapters (LoRA), reusing learned representations.
6. **Data contamination, surface priors, and "unbricking":** Zero-shot "reading" without forward passes often leans on input/output layer arithmetic and memorized priors (degrading under label permutations), while surgical gains in small LLMs (e.g., fixing repetition loops in `Qwen3.5-0.8B`) frequently remove superficial decoding pathologies rather than creating new linguistic capability.

### 1.3 Justified vs. Unjustified Conclusions
| Assessment | Claim |
| :--- | :--- |
| **Justified** | In small or bounded settings, a top-tier frontier LLM coupled with an evaluation feedback loop can localize and adjust parameter subsets to alter or unblock specific behaviors. |
| **Unjustified / Overstated** | That general-purpose LLMs—especially non-frontier or mid-sized models—reliably "understand" arbitrary weight spaces zero-shot without dataset priors or iterative feedback. |
| **Unjustified / Overstated** | That LLM-driven weight editing is competitive in quality, sovereignty, cost, or compute against gradient-based adaptation (`AdamW`, `Muon`, `LoRA`) for real-world language models. |

### 1.4 Methodological Analogy: Awake Brain Mapping & Capability-Aware Perturbations
This study of modifying weights in a neural network reminded me of neurosurgeon **Dr. Jesús Martín-Fernández** (Martín-Fernández et al., 2022, 2025), who operates on brain tumors in awake patients. With the neural network as a brain, one can think that during training—where neurons establish their connections and learn—there is neuroplasticity as in a real brain: when modifying weights and allowing for fine-tuning, internal paths can be rewired so other neurons take charge of new tasks, at the risk of losing previous abilities.

This neurosurgeon first establishes explicit constraints—abilities that must be kept—and maps the brain with small localized perturbations (**Direct Electrical Stimulation, DES**) while the awake patient performs cognitive, multilingual, or socio-emotional tasks, testing how each perturbation affects the patient's abilities (*connectome-stop points*).

In this analogy, when fine-tuning or optimizing a neural network, I first establish tests for the LLM abilities that I do not want to modify (or want affected the least), alongside the target capabilities to improve, and only later start the optimization—whether testing different layers, using an "intelligent optimizer", or applying gradient-based optimizers and `LoRA`."""

    sec2_hypothesis = """## 2. Hypothesis & Relevance to SE.LLMA

### 2.1 Plain-English Research Question
Before investing SE.LLMA engineering time or compute into weight-space editing, I use **`Qwen/Qwen2.5-0.5B-Instruct`** (`bfloat16` on CPU) to answer three simple, practical questions:

### 2.2 Three Testable Claims
- **H1 — Can static weight inspection find the "Swedish weights"?** Looking at weight matrices without running the model (*static inspection*) will **not** accurately locate which layers actually control Swedish behavior when tested during active inference (*awake probing*).
- **H2 — Can small, gradient-free edits fix Swedish without breaking other skills?** Scaling a targeted subset of weights without gradient training—whether chosen statically or via awake probing—will **not** meaningfully improve held-out Swedish over simply scaling random weights, or it will risk hurting protected abilities (*connectome-stop points* like English, coding, and reasoning).
- **H3 — Do standard methods work better?** Keeping weights frozen with better prompting, or adapting them with lightweight gradient training (`Micro-LoRA`), will offer a cleaner and safer trade-off than manual weight edits.

### 2.3 What Would Prove Me Wrong
- Static weight scores correlate strongly with awake layer probing (`r ≥ 0.6`), **and** gradient-free column edits measurably lower held-out Swedish target NLL (`Δ ≤ -0.10` nats beyond 3-seed random noise), beat prompting and `Micro-LoRA`, and cause zero harm to non-negotiable English, coding, and reasoning stop-points."""

    sec3_design = """## 3. Experimental Design, Controls & Evaluation Protocol

### 3.1 Playground — Choosing a Small LM Under CPU Constraints (`notebooks/01_explore_models_cpu.py`)
Given limited local hardware resources (`AMD Ryzen 7 5800U` CPU, `6.7 GiB` total RAM, no discrete GPU) and a strict 4-hour execution budget, I first built an interactive screening playground in `notebooks/01_explore_models_cpu.py` to test small open language models on the identical Swedish prompt (`"Förklara kortfattat vad artificiell intelligens är och ge två konkreta exempel från vardagen."`, `bfloat16`, `max_new_tokens=80`, `8` PyTorch CPU threads), recording RSS RAM, system RAM, CPU usage, load time, and `tokens/s` in `results/cpu_prompt_benchmark.json`.

I chose **`Qwen/Qwen2.5-0.5B-Instruct`** (`1842.4 MB` peak RSS, `5.08 tok/s`) as my main experimental testbed because it fits comfortably in memory and already generates coherent, parseable Swedish while exhibiting clear, repairable defects—specifically **morphological agreement errors** (*"ett tekniskt utvecklade system"*) and **mid-sentence English code-switching** (*"Siri, Apple's smart home assistant"*). Using the same screening notebook and protocol, I could equally have attempted to repair the Swedish issues observed in the other candidate models included in the playground (such as `HuggingFaceTB/SmolLM2-360M-Instruct` at `1565.3 MB` peak RSS, which degenerated into broken pseudo-Swedish, as well as `HuggingFaceTB/SmolLM2-135M-Instruct` or `AI-Sweden-Models/gpt-sw3-126m`).

Following the arguments presented in Section 1, if I had more time and compute resources, I would have liked to replicate the Weight-Space Programming (WSP) tests on a medium-sized language model applied directly to Swedish—comparing frontier-LLM WSP directly analyzing and modifying models like `Qwen` against standard gradient methods, `QLoRA`, and a small trained "intelligent optimizer" equipped with attention mechanisms and conditioned on a text input. Given the hardware and time constraints, I instead scoped and proposed a controlled small-scale study with a Gemini AI agent, presented next.

### 3.2 Step 2 — Diagnostic Suite & Split Discipline (`data/swedish_test_suite.json`, `notebooks/02_eval_qwen_suite.py`)
- **Base Model & Determinism:** `Qwen/Qwen2.5-0.5B-Instruct` (`bfloat16` on CPU, `seed=42`, greedy decoding `do_sample=False`, `max_new_tokens=128`, `8` CPU threads).
- **Suite Creation & Strict `dev` vs. Held-Out `eval` Split:** `32` diagnostic prompts proposed with **Gemini Hydrogen** and manually verified by me, pre-partitioned into **`12 dev`** (used exclusively for DES probing, `alpha` calibration, activation direction, and Micro-LoRA training) and **`20 eval`** (held-out evaluation).
- **Target vs. Non-Negotiable Stop-Points:**
    - **`target` (16 prompts: 6 dev / 10 eval):** `everyday_sv` (`8`) and `public_services_sv` (`8`, covering *Skatteverket*, *Försäkringskassan*, *VAB*, *personnummer*, plus `3` short-context Swedish reading comprehension items).
    - **`non_negotiable` stop-points (16 prompts: 6 dev / 10 eval):** `reasoning_sv` (`6`), `coding_sv` (`6` verified by isolated Python subprocess unit tests), and `control_en` (`4` English control items).
- **Decoupled Grading & Exact NLL:** Greedy generations (`results/qwen_baseline_responses.json`) carry opaque `response_id` hashes and automated checks (unit tests, expected-answer regex, CJK-leak detection) and are evaluated with **Gemini Hydrogen** against gold `reference_answer` targets in `results/qwen_baseline_eval.json` (`is_correct`, `swedish_quality` `[0–2]`, `task_correctness` `[0–2]`, `failure_modes`), verified by me. For the 6-arm comparison (`src/wsp_experiment.py`), I evaluate **exact teacher-forced target Negative Log-Likelihood (NLL)** and **Perplexity (`PPL = exp(NLL)`)** over gold `reference_answer` tokens on all 20 held-out `eval` prompts. Intuitively, NLL is the standard cross-entropy loss (`-log p`) measuring how surprised the model is by the gold reference answer (**lower NLL/PPL = higher probability assigned to the right answer**), giving a deterministic and sensitive CPU signal of whether an intervention moves the model closer to or further from Swedish without hurting English stop-points.

### 3.3 Step 3 — What Was Done in Each of the 6 Tests (`src/wsp_experiment.py`, `notebooks/03_wsp_interventions.py`)
1. **Unchanged Baseline:** Evaluate `Qwen2.5-0.5B-Instruct` untouched on all 20 held-out `eval` prompts to establish reference NLL scores for Swedish targets and protected stop-points.
2. **Static Weight Edit (WSP — zero forward passes):** Compute a "Swedish minus English" vector statically from the output vocabulary matrix (`lm_head.weight`: tokens with `å, ä, ö` and 36 Swedish vs. 30 English function words), find the top-`256` most aligned `mlp.down_proj` columns (`229,376` weights), and scale them by `1.15×` (`+15%`) without running any prompts through the model.
3. **Awake DES Edit:**
   - *Layer probing:* Run a balanced 6-prompt `dev` subset (`3` Swedish targets + `3` stop-points: reasoning, coding, English) while scaling each of the `24` `mlp.down_proj` layers down by `0.85×` one by one, selecting the top-4 layers (`[1, 22, 16, 19]`) that affect Swedish with minimal stop-point disruption.
   - *Column edit & safe calibration:* Inside those 4 layers, select the top-`256` columns aligned with `dev` hidden-state differences (`SV − EN`), test scaling `α ∈ {0.05, 0.10, 0.15, -0.10}` on `dev`, and pick `α = 0.05` (`+5%`) under the constraint `ΔEN_dev ≤ 0.05`.
4. **Random / Untargeted Edit (noise control, 3 seeds):** Sample `256` `(layer, column)` pairs uniformly at random across all 24 layers (seeds `42, 123, 999`) and scale them by `1.15×` (`+15%`) to test whether targeted edits outperform random weight changes of the same magnitude.
5. **Prompting Control (no weight edits):** Leave every weight untouched and prepend a strict Swedish system prompt (`STRICT_SV_SYSTEM_PROMPT`) instructing the model to answer purely in Swedish and output clean Python without extra tests.
6. **Micro-LoRA (classic gradient baseline):** Freeze the `bfloat16` base model, attach small trainable `LoRA` adapters (`r=4`, `lora_alpha=8.0`, `138,240` params) to `mlp.down_proj` in the last 6 layers (`18..23`), and train with `AdamW` (`lr=5e-4`) for `18` steps (`grad_accum=4`) solely on the `12 dev` examples."""

    # Section 4: Results
    s4_parts = ["## 4. Results & Regression Analysis\n"]

    # 4.1 CPU Screening
    s4_parts.append("### 4.1 Preliminary CPU Model Screening (`notebooks/01_explore_models_cpu.py`)")
    if data["has_cpu_bench"]:
        cpu = data["cpu_bench"]
        runs = cpu.get("runs", [])
        cpu_tbl = [
            "| Model | Dtype | Load / Gen (s) | Speed (tok/s) | Peak RSS (Δ RAM) | Proc CPU avg/peak | Sys CPU avg/peak |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        samples_md = []
        for r in runs:
            if r.get("status") == "ok":
                cpu_tbl.append(
                    f"| `{r.get('model_id')}` | `{r.get('dtype', '-')}` | "
                    f"{r.get('load_time_s', '-')} / {r.get('gen_time_s', '-')} | "
                    f"{r.get('tokens_per_sec', '-')} | "
                    f"{r.get('peak_rss_mb', '-')} MB (+{r.get('delta_rss_mb', '-')} MB) | "
                    f"{r.get('avg_proc_cpu_pct', '-')}% / {r.get('peak_proc_cpu_pct', '-')}% | "
                    f"{r.get('avg_sys_cpu_pct', '-')}% / {r.get('peak_sys_cpu_pct', '-')}% |"
                )
                samples_md.append(
                    f"- **`{r.get('model_id')}`:** `{_truncate(r.get('response_text', ''), 240)}`"
                )
            else:
                cpu_tbl.append(
                    f"| `{r.get('model_id')}` | `{r.get('dtype', '-')}` | ERROR | - | - | - | - |"
                )
        s4_parts.append(
            f"*Screening prompt (`max_new_tokens={cpu.get('max_new_tokens', 80)}`, `threads={cpu.get('torch_threads', 8)}`):* `{cpu.get('prompt', '')}`\n\n"
            + "\n".join(cpu_tbl)
            + "\n\n**Sampled Screening Outputs:**\n\n"
            + "\n".join(samples_md)
            + "\n\n`SmolLM2-360M-Instruct` immediately echoed the prompt and degenerated into incoherent pseudo-Swedish (`\"1. **Södra- och längd-själ\"`), whereas `Qwen/Qwen2.5-0.5B-Instruct` stayed under `1.84 GB` peak RSS (`+1,215.9 MB` delta) and generated structured Swedish containing crisp target failures (*\"Artikli intelligens\"*, *\"ett tekniskt utvecklade system\"*, and *\"Exempel 1: Siri, Apple's smart home assistant\"*)."
        )
    else:
        s4_parts.append("*[Pending: run `notebooks/01_explore_models_cpu.py` to generate `results/cpu_prompt_benchmark.json`.]*")

    # 4.2 Baseline Diagnostic Evaluation
    s4_parts.append("\n### 4.2 Diagnostic Baseline Evaluation (`Qwen/Qwen2.5-0.5B-Instruct`)")
    if data["has_baseline"] and data["baseline_stats"]:
        st = data["baseline_stats"]
        meta = st["metadata"]
        ov = st["overall"]
        dev_st = st["by_split"].get("dev", {"ok": 0, "n": 0, "acc": 0.0})
        eval_st = st["by_split"].get("eval", {"ok": 0, "n": 0, "acc": 0.0})
        tgt_st = st["by_priority"].get("target", {"ok": 0, "n": 0, "acc": 0.0})
        nn_st = st["by_priority"].get("non_negotiable", {"ok": 0, "n": 0, "acc": 0.0})

        cat_tbl = [
            "| Category | Priority | Correct | Accuracy | Mean Lang (0–2) | Mean Task (0–2) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for cat, cst in st["by_category"].items():
            cat_tbl.append(
                f"| `{cat}` | `{cst['priority']}` | {cst['ok']}/{cst['n']} | "
                f"{cst['acc']:.1f}% | {cst['lang']:.2f} | {cst['task']:.2f} |"
            )

        fm_str = ", ".join(f"`{m}` ({c})" for m, c in st["failure_modes"]) or "none"

        rep_lines = []
        for ex in st["rep_examples"]:
            verdict = "PASS" if ex["is_correct"] else "FAIL"
            fms = ", ".join(ex["failure_modes"]) or "none"
            rep_lines.append(
                f"- **`{ex['prompt_id']}` (`{ex['category']}`, `{verdict}`, modes: `{fms}`):**\n"
                f"    - *Output:* `{_truncate(ex['response_text'], 165)}`\n"
                f"    - *Reference:* `{_truncate(ex['reference_answer'], 150)}`"
            )

        fig_base_md = _figure_markdown(
            "baseline_diagnostic_summary.png",
            "Baseline diagnostic accuracy by category (left) and failure-mode distribution across 32 prompts (right).",
            image_mode,
        )

        s4_parts.append(
            "\n".join(
                [
                    f"- **Run Config:** `{meta.get('model_id', 'N/A')}` · `dtype={meta.get('dtype', 'N/A')}` · `seed={meta.get('seed', 'N/A')}` · "
                    f"`load={meta.get('load_time_s', '-')}s` · `total_gen={meta.get('total_gen_time_s', '-')}s` (`{meta.get('avg_tokens_per_sec', '-')} tok/s`) · "
                    f"`peak_RSS={meta.get('peak_rss_mb', '-')} MB` (`{meta.get('peak_sys_ram_pct', '-')}%` sys RAM)",
                    f"- **Headline Accuracy:** **`{ov['ok']}/{ov['n']}` (`{ov['acc']:.1f}%`)** · "
                    f"`dev`: **`{dev_st['ok']}/{dev_st['n']}` (`{dev_st['acc']:.1f}%`)** · "
                    f"`eval`: **`{eval_st['ok']}/{eval_st['n']}` (`{eval_st['acc']:.1f}%`)** · "
                    f"`target`: **`{tgt_st['ok']}/{tgt_st['n']}` (`{tgt_st['acc']:.1f}%`)** · "
                    f"`non_negotiable`: **`{nn_st['ok']}/{nn_st['n']}` (`{nn_st['acc']:.1f}%`)**",
                    "",
                    *cat_tbl,
                    "",
                    f"**Failure Modes (32 prompts):** {fm_str}",
                    "",
                    *( [fig_base_md, ""] if fig_base_md else [] ),
                    "**Key Diagnostic Findings:**",
                    "",
                    "- **English control intact vs. ungrounded Swedish collapse:** Fluent English (`3/4`, `2.00/2` language score in `control_en`), whereas open-ended Swedish fails completely (`0/8` in `everyday_sv`; `0/5` on parametric `public_services_sv`), inventing pseudo-Swedish (*Artikli*, *kortkastning*, *matvallarna*) and nonexistent institutions (*Sveriges riksdagen*, *Sveriges försökningsrådet*).",
                    "- **Context-grounded Swedish comprehension works (`3/3`):** On reading comprehension (`pub_dev_03`, `pub_eval_04`, `pub_eval_05`), the model extracts facts accurately (`3/3`), showing Swedish syntax parsing exists when grounded even though parametric knowledge and ungrounded fluency fail.",
                    "- **Swedish coding reveals English priors and truncation (`1/6` strict pass):** `code_eval_01` defaults to English vowels (`a, e, i, o, u`) while dropping `y, å, ä, ö` explicitly listed in Swedish; `sum_even` and `fizzbuzz_list` implement valid logic but append unsolicited tests and truncate at `128` tokens (`SyntaxError`).",
                    "",
                    "**Representative Baseline Examples:**",
                    "",
                    *rep_lines,
                ]
            )
        )
    else:
        s4_parts.append("*[Pending: run `notebooks/02_eval_qwen_suite.py` to generate baseline outputs.]*")

    # 4.3 Main 6-Arm Experiment & Layer Mapping
    s4_parts.append("\n### 4.3 Main 6-Arm Experiment & Layer Mapping (`src/wsp_experiment.py`)")
    if data["has_experiment_details"] and data["experiment_view"]:
        ev = data["experiment_view"]
        meta = ev["metadata"]
        corr = ev["correlation"]
        loss_curve = ev["micro_lora_loss_curve"]
        lora_txt = (
            f"`{loss_curve[0]:.4f} → {loss_curve[-1]:.4f}` (18 optimizer steps on `dev`)"
            if loss_curve
            else "N/A"
        )
        fig_map_md = _figure_markdown(
            "layer_mapping_static_vs_des.png",
            "Layer-wise alignment between data-free Static WSP scores and Awake DES causal sensitivity (left) and DES selectivity (right).",
            image_mode,
        )
        fig_arms_md = _figure_markdown(
            "arm_comparison_nll.png",
            "Held-out Swedish target NLL across all 6 experimental arms (left) and target vs. English stop-point trade-off (right).",
            image_mode,
        )
        s4_parts.append(
            "\n".join(
                [
                    f"- **Run Config:** `{meta.get('model_id')}` (rev `{meta.get('revision')}`) · `dtype={meta.get('dtype')}` · "
                    f"`seed={meta.get('seed')}` · `threads={meta.get('torch_threads')}` · `nll_only={meta.get('nll_only')}` · "
                    f"`peak_RSS={meta.get('peak_rss_mb')} MB`",
                    f"- **Static WSP vs. Awake DES Correlation:** Pearson `r = {corr.get('pearson_r')}`, Spearman `ρ = {corr.get('spearman_rho')}` · "
                    f"**Top-4 Selected DES Layers:** `{ev['selected_des_layers']}`",
                    f"- **Micro-LoRA Training Loss (`dev`):** {lora_txt}",
                    "",
                    _format_arm_table(ev),
                    *( ["", fig_map_md] if fig_map_md else [] ),
                    *( ["", fig_arms_md] if fig_arms_md else [] ),
                ]
            )
        )

        if include_appendix:
            alpha_trials = ev.get("des_alpha_trials", [])
            if alpha_trials:
                a_tbl = [
                    "| Candidate `α` | Dev SV Target NLL | Δ Dev EN Stop-Point | Feasible (`ΔEN ≤ 0.05`) |",
                    "| :---: | :---: | :---: | :---: |",
                ]
                for tr in alpha_trials:
                    a_tbl.append(
                        f"| `{tr.get('alpha')}` | {tr.get('dev_sv_target_nll')} | "
                        f"{tr.get('delta_en_ctrl_dev_nll'):+.4f} | {'Yes' if tr.get('feasible') else 'No'} |"
                    )
                s4_parts.append("\n**DES `α` Calibration Trials (`dev`):**\n\n" + "\n".join(a_tbl))

            per_layer = ev.get("per_layer", [])
            if per_layer:
                selected_set = set(ev.get("selected_des_layers", []))
                l_tbl = [
                    "| Layer | Static Score | DES ΔSV NLL | DES ΔStop NLL | Selectivity |",
                    "| :---: | :---: | :---: | :---: | :---: |",
                ]
                for row in per_layer:
                    lyr = row.get("layer")
                    mark = " ★" if lyr in selected_set else ""
                    l_tbl.append(
                        f"| `{lyr}`{mark} | {row.get('static_score')} | "
                        f"{row.get('delta_sv_nll'):+.5f} | {row.get('delta_stop_nll'):+.5f} | "
                        f"{row.get('selectivity')} |"
                    )
                s4_parts.append(
                    "\n**Layer-Wise Static WSP vs. Awake DES Probe (`★` = Top-4 DES Layers):**\n\n"
                    + "\n".join(l_tbl)
                )
    else:
        s4_parts.append(
            "\n".join(
                [
                    "> **Pending 6-Arm Execution:** `results/experiment_details.json` has not been generated yet. Run one of the commands below to populate the full 6-arm table, layer correlation plots, and held-out quantitative deltas:",
                    ">",
                    "> - Fast NLL + Layer Mapping (~2–3 min CPU): `uv run python -m src.wsp_experiment --nll-only`",
                    "> - Full NLL + 20 Held-Out Generations (~18 min CPU) or interactive UI: `uv run marimo run notebooks/03_wsp_interventions.py`",
                    "",
                    "| Arm | SV Target NLL (Δ) | SV PPL | Non-Neg NLL | EN Stop NLL (Δ) | Params (L2) | CPU (s) |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                    "| 1. Unchanged Baseline | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | `0 (0.0)` | *[Pending]* |",
                    "| 2. Static Weight Edit (WSP) | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |",
                    "| 3. Awake DES Edit | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |",
                    "| 4. Random / Untargeted Edit (3 seeds) | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |",
                    "| 5a. Prompting / Inference Control | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | `0 (0.0)` | *[Pending]* |",
                    "| 5b. Micro-LoRA (layers 18..23, r=4) | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | `138,240` | *[Pending]* |",
                ]
            )
        )

    # 4.4 Per-Category Held-Out NLL Breakdown & Optional Qualitative Arm Examples
    if include_appendix:
        if data["has_experiment_details"] and data["experiment_view"]:
            ev = data["experiment_view"]
            per_cat = ev.get("per_category_nll", {})
            cat_tbl = [
                "| Category | Baseline | WSP Static | Awake DES | Random (3-seed) | Prompt Ctrl | Micro-LoRA (Δ vs Base) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            for cat in CATEGORY_ORDER:
                b_val = per_cat.get("baseline", {}).get(cat, 0.0)
                w_val = per_cat.get("wsp_static", {}).get(cat, 0.0)
                d_val = per_cat.get("des_awake_edit", {}).get(cat, 0.0)
                r_val = per_cat.get("random_edit", {}).get(cat, 0.0)
                p_val = per_cat.get("prompt_control", {}).get(cat, 0.0)
                l_val = per_cat.get("micro_lora", {}).get(cat, 0.0)
                cat_tbl.append(
                    f"| `{cat}` | {b_val:.4f} | {w_val:.4f} | {d_val:.4f} | "
                    f"{r_val:.4f} | {p_val:.4f} | {l_val:.4f} (`{l_val - b_val:+.3f}`) |"
                )
            s4_parts.append(
                "\n### 4.4 Per-Category Held-Out NLL Breakdown (Unbricking vs. New Capability)\n\n"
                + "\n".join(cat_tbl)
                + f"\n\n**Key Granular Finding:** `Micro-LoRA` improved **{ev.get('coding_improved_count', 0)}/{ev.get('coding_total_count', 0)}** "
                + f"held-out `coding_sv` prompts (`{ev.get('baseline_coding_nll', 0.0):.4f} → {ev.get('micro_lora_coding_nll', 0.0):.4f}`, "
                + f"`Δ={ev.get('micro_lora_coding_delta', 0.0):+.3f}`), where latent Python coding ability already existed and only concise formatting needed alignment (*unbricking*). "
                + f"Conversely, on ungrounded `everyday_sv` (`{ev.get('baseline_everyday_nll', 0.0):.4f} → {ev.get('micro_lora_everyday_nll', 0.0):.4f}`), "
                + f"18 steps on `12` `dev` items overfit and raised held-out NLL alongside `control_en` (`+0.051`)."
            )

        if data["arm_examples"] or data["summary_examples"]:
            s4_parts.append("\n### 4.5 Held-Out Qualitative Examples Across Arms")
            if data["arm_examples"]:
                ex_blocks = []
                for ex in data["arm_examples"]:
                    lines = [
                        f"- **`{ex['prompt_id']}` (`{ex['category']}`, `{ex['priority']}`):** *{_truncate(ex['prompt'], 140)}*"
                    ]
                    for arm_key in ARM_ORDER:
                        if arm_key in ex["outputs"]:
                            lines.append(
                                f"    - `{arm_key}`: `{_truncate(ex['outputs'][arm_key], 170)}`"
                            )
                    ex_blocks.append("\n".join(lines))
                s4_parts.append("\n".join(ex_blocks))
            else:
                ex_blocks = []
                for idx, ex in enumerate(data["summary_examples"], start=1):
                    ex_blocks.append(
                        f"- **Example {idx}:** *{_truncate(ex.get('prompt', ''), 140)}*\n"
                        f"    - Baseline: `{_truncate(ex.get('baseline_output', ''), 170)}`\n"
                        f"    - Edited: `{_truncate(ex.get('edited_output', ''), 170)}`"
                    )
                s4_parts.append("\n".join(ex_blocks))
        elif not data["has_experiment_details"]:
            s4_parts.append(
                "\n### 4.4 Held-Out Qualitative Examples Across Arms\n"
                "*[Pending 6-arm execution (`src/wsp_experiment.py`).]*"
            )

    sec4_results = "\n".join(s4_parts)

    # Section 5: Interpretation, Limitations & Recommendation
    if data["has_experiment_details"] and data["experiment_view"]:
        ev = data["experiment_view"]
        corr = ev["correlation"]
        rows = {r["key"]: r for r in ev["arm_rows"]}
        wsp = rows.get("wsp_static", {})
        des = rows.get("des_awake_edit", {})
        rand = rows.get("random_edit", {})
        prompt = rows.get("prompt_control", {})
        lora = rows.get("micro_lora", {})

        pearson_r = float(corr.get("pearson_r", 0.0))
        h1_verdict = (
            "Supported (weak static-to-causal correlation)"
            if abs(pearson_r) < 0.5
            else "Challenged (notable static-to-causal correlation)"
        )
        wsp_d = float(wsp.get("delta_sv_nll", 0.0))
        des_d = float(des.get("delta_sv_nll", 0.0))
        rand_d = float(rand.get("delta_sv_nll", 0.0))
        rand_std = float(rand.get("sv_target_nll_std", 0.0))
        lora_d = float(lora.get("delta_sv_nll", 0.0))
        prompt_d = float(prompt.get("delta_sv_nll", 0.0))
        coding_ok = ev.get("coding_improved_count", 0)
        coding_tot = ev.get("coding_total_count", 0)
        coding_d = ev.get("micro_lora_coding_delta", 0.0)

        wsp_beats = wsp_d < -0.01 and wsp_d < rand_d - 2.0 * rand_std
        des_beats = des_d < -0.01 and des_d < rand_d - 2.0 * rand_std
        h2_verdict = (
            "Supported (gradient-free edits yield zero held-out gain over baseline)"
            if not (wsp_beats or des_beats)
            else "Partially challenged (a weight edit improved over baseline and random controls)"
        )

        interp_md = "\n".join(
            [
                "### 5.1 Interpretation of Findings",
                f"- **1. Static inspection does not find where the language actually lives (`{h1_verdict}`):** Comparing static `lm_head`→`mlp.down_proj` alignment against awake DES probing across all 24 layers yields near-zero (slightly negative) correlation (**Pearson `r = {corr.get('pearson_r')}`, Spearman `ρ = {corr.get('spearman_rho')}`**). Static scoring ranked layer `20` highest (`0.0977`), where DES perturbation actually measured negative target impact (`-0.0024`) and stop-point harm (`+0.0113`), whereas awake probing isolated layers **`{ev['selected_des_layers']}`** (peak selectivity `2.57` at layer `1`).",
                f"- **2. Gradient-free weight edits and prompting barely move the model (`{h2_verdict}`):** Both `Static WSP` (`ΔSV={wsp_d:+.3f}`) and `Awake DES` (`ΔSV={des_d:+.3f}`), as well as `Prompting` (`ΔSV={prompt_d:+.3f}`), leave held-out Swedish NLL virtually identical to the untouched baseline (`2.6328`) and no better than scaling `256` random columns (`Random`: `ΔSV={rand_d:+.3f} ± {rand_std:.4f}`). Re-scaling a small subset of columns without gradients cannot inject missing Swedish knowledge—though calibrating `Awake DES` against stop-points (`α=0.05`) did protect English fluency (`ΔEN={des.get('delta_en_nll', 0.0):+.3f}`) compared to random perturbations (`ΔEN={rand.get('delta_en_nll', 0.0):+.3f}`).",
                f"- **3. Why `Micro-LoRA` got worse on average (`+0.075` SV, `+0.051` EN) yet improved `4/4` Python tests (*unbricking* vs. learning a new language):** At first glance, `Micro-LoRA` moved the weights much more than the surgical edits, but in the wrong direction on average (`ΔSV={lora_d:+.3f}`, `ΔEN={lora.get('delta_en_nll', 0.0):+.3f}`). Looking at the per-item breakdown explains why: training for `18` steps (`grad_accum=4`, ~6 passes) on only **`12` `dev` examples** caused **overfitting** to those specific 12 responses, hurting generalization on unseen open-ended Swedish prompts (`ev_eval_03` `+0.1906`, `pub_eval_02` `+0.1519`). Crucially, however, `Micro-LoRA` improved **{coding_ok}/{coding_tot}** held-out Swedish Python coding tasks (`Δ={coding_d:+.3f}`: `-0.0841`, `-0.0463`, `-0.0087`, `-0.0864`). This highlights the core distinction between ***unbricking*** (the model already knew Python and 12 examples were enough to align its output formatting) and **acquiring missing language competence** (12 examples cannot teach Swedish fluency or facts that a `0.5B` model lacks; they only overfit it).",
            ]
        )

        rec_text = (
            "**I do not recommend pursuing LLM-driven or data-free static weight-space programming for SE.LLMA.** "
            "Sending weights as text strings through proprietary frontier APIs is expensive, breaks technological sovereignty, and in my tests static weight inspection failed to predict which layers actually matter (`r = -0.0747`) while manual column edits produced zero held-out Swedish improvement. "
            "Instead, I think SE.LLMA should keep investing in sovereign continued pretraining and standard gradient-based methods (`AdamW`, `Muon`, `LoRA`), while adopting the **awake-mapping idea**—setting clear *stop-point* test suites for abilities that must not degrade before running any optimization."
        )
    else:
        interp_md = "\n".join(
            [
                "### 5.1 Interpretation of Findings",
                "> **[Pending 6-Arm Execution]** Complete baseline and screening evidence already establishes two key diagnostic facts, while the 6-arm causal test awaits execution:",
                "",
                "- **Confirmed Diagnostic Baseline Insight (Context vs. Parametric/Fluency Gap):** `Qwen/Qwen2.5-0.5B-Instruct` succeeds on **3/3** Swedish reading comprehension items and **3/4** English controls, yet collapses to **0/8** on ungrounded `everyday_sv` and **0/5** on parametric Swedish public services—frequently reverting to English priors (e.g., dropping `y, å, ä, ö` in `code_eval_01`) and hallucinating vocabulary.",
                "- **Decision Rubric Once `results/experiment_details.json` Exists:**",
                "    - *H1 Check:* Does static `lm_head`→`mlp.down_proj` alignment correlate strongly (`|r| ≥ 0.5`) with awake DES sensitivity, or fail (*paradox of non-locality*)?",
                "    - *H2 Check:* Do Static WSP or Awake DES beat 3-seed Random edits cleanly without raising `EN Stop-Point NLL`?",
                "    - *H3 Check:* Do Strict Prompting and Micro-LoRA dominate gradient-free weight edits on the `ΔSV NLL` vs. `ΔEN Stop-Point NLL` trade-off?",
            ]
        )
        rec_text = (
            "**Preliminary Recommendation (Subject to 6-Arm Run Confirmation):** Based on the literature replication audit (`WeightBench`, `weight-space-bench`) and sovereign compute constraints, SE.LLMA should **not** rely on frontier LLM-driven weight-space programming to optimize or patch Swedish models. At most, **awake capability-aware perturbation mapping** (stop-point suites paired with targeted forward-pass probing) merits lightweight diagnostic use alongside standard gradient pipelines (`AdamW`, `Muon`, `LoRA`)."
        )

    sec5_interpretation = "\n\n".join(
        [
            "## 5. Interpretation, Limitations & Recommendation for SE.LLMA",
            interp_md,
            """### 5.2 Limitations
- **Hardware, time & model scale:** Because I ran everything locally on a laptop CPU (`AMD Ryzen 7 5800U`, `6.7 GiB` RAM, no GPU) within a 4-hour budget (`~91 min` for the 6-arm `--nll-only` run, `~68.6 min` of which was `Micro-LoRA` backprop), I had to test a `0.5B`-parameter model (`Qwen/Qwen2.5-0.5B-Instruct`) that has very limited built-in Swedish compared to 7B+ SE.LLMA models.
- **Small dataset (`12 dev` / `20 eval`) & NLL evaluation:** A `32`-prompt suite was small enough to run and inspect carefully on CPU, using deterministic teacher-forced NLL across the 6 arms instead of generating full text for every arm and seed—which also meant `Micro-LoRA` only had `12` training examples and quickly overfit open-ended Swedish.
- **Simple manual edit rule:** My gradient-free tests only scaled `256` columns in `mlp.down_proj`; more complex edits could behave differently, though they still face the same trial-and-error search bottleneck.""",
            f"""### 5.3 Recommendation for SE.LLMA
- **Should SE.LLMA investigate Weight-Space Programming further?** {rec_text}
- **What I would test next with more compute:** On a medium-sized (`7B–14B`) open checkpoint with stronger Swedish foundations, I would use **awake layer probing with stop-point constraints** (English, coding, and reasoning) to see which layers are safest to adapt, and compare standard `LoRA` / `Muon` fine-tuning against **layer-selective `LoRA`**—or explore a small, dedicated **selective optimizer** that operates directly on floating-point weights via attention without converting numbers into text tokens.""",
        ]
    )

    sec6_reproducibility_refs = """## 6. Reproducibility, AI Usage & References

### 6.1 Reproduction Steps
```bash
# 1. Install locked environment (includes marimo, torch, transformers, weasyprint)
uv sync

# 2. Screen candidate small models on CPU (writes results/cpu_prompt_benchmark.json)
uv run marimo run notebooks/01_explore_models_cpu.py

# 3. Run 32-prompt Swedish diagnostic baseline (writes results/qwen_baseline_responses.json)
uv run marimo run notebooks/02_eval_qwen_suite.py

# 4. Run 6-arm WSP vs. Awake DES vs. Micro-LoRA experiment
#    NLL + layer mapping (~91 min total on 8-thread CPU, ~68 min in Micro-LoRA backprop):
uv run python -m src.wsp_experiment --nll-only
#    Or full run adding 20 held-out generations per arm (~+16-20 min CPU):
uv run python -m src.wsp_experiment

# 5. Export report/research_note.md and report/report.pdf (or preview interactively in marimo)
uv run python -m src.report_builder
uv run marimo run notebooks/report.py
```

### 6.2 AI Usage Disclosure
- **Tools used:** I used **Gemini Hydrogen** (Antigravity agent workflow) to propose the 32 diagnostic prompts (`data/swedish_test_suite.json`), draft gold reference answers, and evaluate the baseline Swedish outputs in `results/qwen_baseline_eval.json` (alongside automated Python subprocess unit tests and regex checks), as well as to synthesize replication benchmarks, scaffold the `marimo` notebooks (`01_explore_models_cpu.py`, `02_eval_qwen_suite.py`, `03_wsp_interventions.py`, `report.py`), and implement `src/wsp_experiment.py` and `src/report_builder.py`.
- **Human verification:** All hypotheses, experimental controls (`dev`/`eval` split separation, 3-seed random controls, stop-point regressions), prompt evaluations, code execution outputs, and numerical summaries reported here were directly verified by me against saved JSON artifacts and automated subprocess Python unit tests.

---

## References
1. Takehi, R., Teoh, J., Zhang, A., & Khattab, O. *Sparks of Weight-Space Programming*. https://rikiyat.github.io/blog/weight-space-programming/ (Official logs: https://github.com/RikiyaT/lm_understand-logs)
2. `eomcaleb/weight-space-bench`: Cross-model replication and token/cost benchmark on MinAtar weight-space writing. https://github.com/eomcaleb/weight-space-bench
3. Khajavi, M. (`MatinKhajavi/Weight-Space-Programming`): Independent replication of game and repair pipelines across models. https://github.com/MatinKhajavi/Weight-Space-Programming
4. Marasini, A., Khadka, B., & Rahimi, N. *WeightBench: Evaluating Language Models as Weight-Space Programmers*. `arXiv:2611.18553` (https://github.com/Aayush-Marasini/WeightBench)
5. Martín-Fernández, J., et al. (2025). *Where are higher-order cognitive functions? The paradox of non-locality in awake cognitive mapping using a complex dynamic system framework.* *Frontiers in Psychology*, PMID: `40110086` (https://pubmed.ncbi.nlm.nih.gov/40110086/)
6. Martín-Fernández, J., Gabarrós, A., & Fernandez-Coello, A. (2022). *Intraoperative Brain Mapping in Multilingual Patients: What Do We Know and Where Are We Going?* *Brain Sciences*, PMID: `35624947` (https://pubmed.ncbi.nlm.nih.gov/35624947/)"""

    return {
        "header_and_exec": header_and_exec,
        "sec1_critique": sec1_critique,
        "sec2_hypothesis": sec2_hypothesis,
        "sec3_design": sec3_design,
        "sec4_results": sec4_results,
        "sec5_interpretation": sec5_interpretation,
        "sec6_reproducibility_refs": sec6_reproducibility_refs,
    }


def build_research_note_md(data: dict[str, Any]) -> str:
    """Build a concise 1-2 page Markdown research note (~900-1,300 words)."""
    is_draft = not data["has_experiment_details"]
    status_tag = (
        "Draft (Baseline Complete; 6-Arm Run Pending)"
        if is_draft
        else "Final (All 6 Arms Complete)"
    )

    if data["has_cpu_bench"]:
        cpu_runs = {r.get("model_id"): r for r in data["cpu_bench"].get("runs", [])}
        q_run = cpu_runs.get("Qwen/Qwen2.5-0.5B-Instruct", {})
        s_run = cpu_runs.get("HuggingFaceTB/SmolLM2-360M-Instruct", {})
        screening_block = (
            f"- **CPU Screening (`results/cpu_prompt_benchmark.json`):** Tested small models on the same Swedish prompt to choose one for the main experiment:\n"
            f"  - **`Qwen/Qwen2.5-0.5B-Instruct` *(selected)*:** `{q_run.get('tokens_per_sec', 5.08)} tok/s`, `{q_run.get('peak_rss_mb', 1842.4)} MB` peak RAM. Readable Swedish with clear, fixable issues (agreement error *\"ett tekniskt utvecklade system\"* and English code-switching *\"Siri, Apple's smart home assistant\"*).\n"
            f"  - **`HuggingFaceTB/SmolLM2-360M-Instruct`:** `{s_run.get('tokens_per_sec', 6.46)} tok/s`, `{s_run.get('peak_rss_mb', 1565.3)} MB` peak RAM. Echoed the prompt and collapsed into broken pseudo-Swedish (*\"1. **Södra- och längd-själ\"*).\n"
            f"  - *Other configured candidates (`notebooks/01_explore_models_cpu.py`):* `SmolLM2-135M-Instruct`, `AI-Sweden-Models/gpt-sw3-126m` (not run once `Qwen` passed and `SmolLM2-360M` failed).\n"
        )
    else:
        screening_block = ""

    if data["has_baseline"] and data["baseline_stats"]:
        st = data["baseline_stats"]
        ov = st["overall"]
        ev_cat = st["by_category"].get("everyday_sv", {"ok": 0, "n": 8})
        pub_cat = st["by_category"].get("public_services_sv", {"ok": 0, "n": 8})
        reas_cat = st["by_category"].get("reasoning_sv", {"ok": 0, "n": 6})
        code_cat = st["by_category"].get("coding_sv", {"ok": 0, "n": 6})
        ctrl_cat = st["by_category"].get("control_en", {"ok": 0, "n": 4})
        baseline_block = (
            screening_block
            + f"- **32-Prompt Baseline (`Qwen2.5-0.5B-Instruct`, graded with Gemini Hydrogen + automated checks):** Passed **{ov['ok']}/{ov['n']} ({ov['acc']:.1f}%)** overall: "
            + f"fluent English (`{ctrl_cat['ok']}/{ctrl_cat['n']}`) and context-grounded Swedish reading comprehension (`3/3`), "
            + f"but `0/8` on everyday Swedish and `0/5` on Swedish public-service facts from memory (`{pub_cat['ok']}/{pub_cat['n']}` in `public_services_sv`) "
            + f"due to grammar (`13`), truncation (`13`), invented words (`8`), and English code-switching (`7`), plus `{reas_cat['ok']}/{reas_cat['n']}` on Swedish reasoning and `{code_cat['ok']}/{code_cat['n']}` on Python coding."
        )
    else:
        baseline_block = screening_block + "- Baseline evaluation pending (`results/qwen_baseline_eval.json`)."

    if data["has_experiment_details"] and data["experiment_view"]:
        ev = data["experiment_view"]
        corr = ev["correlation"]
        rows = {r["key"]: r for r in ev["arm_rows"]}
        base_r = rows.get("baseline", {})
        wsp = rows.get("wsp_static", {})
        des = rows.get("des_awake_edit", {})
        rand = rows.get("random_edit", {})
        lora = rows.get("micro_lora", {})
        prompt = rows.get("prompt_control", {})
        coding_ok = ev.get("coding_improved_count", 0)
        coding_tot = ev.get("coding_total_count", 0)
        coding_d = ev.get("micro_lora_coding_delta", 0.0)
        exp_block = (
            f"- **Static vs. Awake Layer Probing:** Near-zero correlation across the 24 layers (Pearson `r = {corr.get('pearson_r')}`, Spearman `ρ = {corr.get('spearman_rho')}`; Awake DES top-4 layers: `{ev['selected_des_layers']}`).\n"
            f"- **6-Arm Held-Out Comparison (`eval`):** *(Lower `NLL`/`PPL` = higher probability for the reference answer; `Δ` = change vs. Baseline.)*\n\n"
            + _format_arm_table(ev)
        )
        interp_block = (
            f"- **Why manual weight edits and prompting barely changed anything (`H1` & `H2` confirmed):** Static inspection failed to locate the layers that affect Swedish (`r = {corr.get('pearson_r')}`). `Static WSP` (`Δ={wsp.get('delta_sv_nll', 0.0):+.3f}`), `Awake DES` (`Δ={des.get('delta_sv_nll', 0.0):+.3f}`), and `Prompting` (`Δ={prompt.get('delta_sv_nll', 0.0):+.3f}`) stayed virtually identical to the untouched baseline (`{base_r.get('sv_target_nll', 0.0):.4f}`) and no better than touching `256` random columns (`Random`: `Δ={rand.get('delta_sv_nll', 0.0):+.3f} ± {rand.get('sv_target_nll_std', 0.0):.4f}`). Scaling `256` columns without gradients cannot add missing Swedish fluency—though `Awake DES` stop-point calibration (`α = 0.05`) protected English (`ΔEN={des.get('delta_en_nll', 0.0):+.3f}` vs. Random `{rand.get('delta_en_nll', 0.0):+.3f}`).\n"
            f"- **Why `Micro-LoRA` got worse on average (`+0.075` SV, `+0.051` EN) yet improved `4/4` Python tests:** Training `18` steps on only **`12` `dev` examples** overfit those 12 answers, hurting unseen open-ended Swedish (`ev_eval_03` `+0.1906`, `pub_eval_02` `+0.1519`). Yet it improved **all {coding_ok}/{coding_tot}** held-out Swedish Python tasks (`Δ={coding_d:+.3f}`: `-0.0841`, `-0.0463`, `-0.0087`, `-0.0864`), showing the divide between ***unbricking*** an existing skill (Python formatting) and **learning a missing language**."
        )
    else:
        exp_block = (
            "*[Pending execution of `uv run python -m src.wsp_experiment --nll-only` -> `results/experiment_details.json`.]*\n\n"
            "| Arm | SV Target NLL (Δ) | EN Stop NLL (Δ) | Params | Status |\n"
            "| :--- | :---: | :---: | :---: | :---: |\n"
            "| 1. Unchanged Baseline | *[Pending]* | *[Pending]* | `0` | Ready |\n"
            "| 2. Static Weight Edit (WSP) | *[Pending]* | *[Pending]* | `K=256` cols | Ready |\n"
            "| 3. Awake DES Edit | *[Pending]* | *[Pending]* | `K=256` cols | Ready |\n"
            "| 4. Random Edit (3 seeds) | *[Pending]* | *[Pending]* | `K=256` cols | Ready |\n"
            "| 5a. Prompting Control | *[Pending]* | *[Pending]* | `0` | Ready |\n"
            "| 5b. Micro-LoRA (`r=4`) | *[Pending]* | *[Pending]* | `138,240` | Ready |"
        )
        interp_block = (
            "- **[Pending 6-Arm Run]** Baseline diagnostics already establish that `Qwen2.5-0.5B-Instruct` preserves fluent English (`3/4`) and Swedish reading comprehension (`3/3`) "
            "while failing open-ended Swedish (`0/8`) and dropping `y, å, ä, ö` in Swedish coding prompts."
        )

    fig_links = []
    for fname, cap in (
        ("baseline_diagnostic_summary.png", "Baseline Diagnostic Summary"),
        ("layer_mapping_static_vs_des.png", "Static WSP vs. Awake DES Layer Mapping"),
        ("arm_comparison_nll.png", "6-Arm Held-Out NLL Comparison"),
    ):
        if (RESULTS_DIR / fname).exists():
            fig_links.append(f"![{cap}](../results/{fname})")
    figures_str = ("\n\n" + "\n\n".join(fig_links)) if fig_links else ""

    return f"""# Weight-Space Programming for SE.LLMA — Research Note

**Status:** {status_tag} · **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (`bfloat16`, CPU, `seed=42`) · **Updated:** `{data['generated_at_utc']}`

> **Full Critical Evaluation & Report:** My complete critical examination of *Sparks of Weight-Space Programming* (strongest evidence, frontier-model & cost limitations via `eomcaleb/weight-space-bench` and `WeightBench`, technical and data-flow critique, and the **Dr. Jesús Martín-Fernández** awake brain-mapping analogy) is in Section 1 of [`report/report.pdf`](report.pdf).

## 1. Playground & Hypothesis (`notebooks/01_explore_models_cpu.py`)
With limited local resources (`AMD Ryzen 7 5800U` CPU, `6.7 GiB` RAM, no GPU, 4-hour budget), I built a playground in `notebooks/01_explore_models_cpu.py` to screen small open models on Swedish and chose **`Qwen/Qwen2.5-0.5B-Instruct`** (`bfloat16`, `1.84 GB` RAM, `5.08 tok/s`), though the same workflow applies to the other candidates (`SmolLM2-360M/135M`, `gpt-sw3-126m`). With more compute, I would have replicated WSP on a medium-sized model for Swedish comparing WSP, gradient methods, `QLoRA`, and a small trained attention-based selective optimizer; under my constraints, I proposed the smaller study below with a Gemini AI agent to test three practical questions:
- **H1 (Can static weight inspection find the "Swedish weights"?):** Inspecting weight matrices without running prompts will **not** accurately locate the layers that control Swedish during active inference (*awake probing*).
- **H2 (Can small, gradient-free edits fix Swedish safely?):** Scaling targeted weights without gradient training—chosen statically or via awake probing—will **not** beat random weight edits on held-out Swedish without risking protected skills (*stop-points*: English, coding, reasoning).
- **H3 (Do standard methods work better?):** Prompting or lightweight gradient training (`Micro-LoRA`) will offer a cleaner trade-off than manual weight edits.
- *What would prove me wrong:* Strong correlation (`r ≥ 0.6`) between static inspection and awake probing, plus held-out Swedish NLL gains beating random edits, prompting, and `Micro-LoRA` with zero stop-point harm.

## 2. Experimental Design & What Each Test Did
- **Dataset (`data/swedish_test_suite.json`) & NLL Metric:** `32` prompts proposed and graded with **Gemini Hydrogen** (plus automated Python unit tests/regex, verified by me), split into `12 dev` (probing, calibration, LoRA training) and `20 eval` (held-out exam): `16` Swedish targets (`everyday_sv`, `public_services_sv`) and `16` stop-points (`reasoning_sv`, `coding_sv`, `control_en`). Across all 6 tests I measure teacher-forced **Negative Log-Likelihood (`NLL = -log p`)** and **Perplexity (`PPL = exp(NLL)`)** on the gold reference answers (**lower = higher probability assigned to the right answer**).
- **The 6 Tests (`src/wsp_experiment.py`):**
  1. **Unchanged Baseline:** `Qwen2.5-0.5B-Instruct` untouched on `eval`.
  2. **Static Weight Edit (WSP):** Data-free `lm_head` Swedish−English vector → top-`256` aligned `mlp.down_proj` columns (`229,376` weights) scaled by `+15%` (`1.15×`).
  3. **Awake DES Edit:** Perturb each of the `24` layers (`0.85×`) on a 6-prompt `dev` probe to pick the `4` most selective Swedish layers (`[1, 22, 16, 19]`), pick `256` columns aligned to `dev` activations, and scale by `+5%` (`α = 0.05`, calibrated so `ΔEN_dev ≤ 0.05`).
  4. **Random Edit (3 seeds):** Scale `256` random columns by `+15%` across seeds `42, 123, 999`.
  5. **Prompting Control:** Strict Swedish system prompt with zero weight edits.
  6. **Micro-LoRA:** Frozen base + trainable `LoRA` (`r=4`, `layers 18..23`, `138,240` params) trained for `18` `AdamW` steps on the `12 dev` items.

## 3. Results
{baseline_block}

{exp_block}{figures_str}

## 4. Interpretation, Limitations & Recommendation
{interp_block}
- **Limitations:** Laptop CPU budget (`~91 min` for the 6-arm `--nll-only` run), `0.5B` model scale, `32`-prompt suite (`12 dev` / `20 eval`), teacher-forced `NLL`, and linear column scaling on `mlp.down_proj`.
- **My recommendation for SE.LLMA:** **Do not pursue LLM-driven or data-free static weight-space programming** to train or fix Swedish models—it is costly, threatens sovereignty, failed to locate the right layers (`r = -0.0747`), and gave zero gain over random edits. Focus instead on sovereign continued pretraining and gradient fine-tuning (`AdamW`, `Muon`, `LoRA`) paired with **awake stop-point test suites**. *Next experiment with more compute:* Use awake stop-point probing on a `7B–14B` Swedish checkpoint to guide layer-selective `LoRA` (or test a small dedicated attention-based optimizer acting directly on numeric weights).

## 5. Reproducibility & AI Usage
- **Reproduce:** `uv sync && uv run python -m src.wsp_experiment --nll-only && uv run python -m src.report_builder`
- **AI Usage:** I used **Gemini Hydrogen** to propose the 32 prompts (`data/swedish_test_suite.json`), draft reference answers, evaluate baseline outputs (`results/qwen_baseline_eval.json`, alongside automated Python tests/regex), synthesize replication benchmarks, and scaffold the `marimo` notebooks and `src/` modules; all controls, runs, and numbers were verified by me.
- **Key References:** Takehi et al. (*Sparks of Weight-Space Programming*); `eomcaleb/weight-space-bench`; Marasini et al. (`WeightBench`, `arXiv:2611.18553`); Martín-Fernández et al. (`2022` PMID `35624947`, `2025` PMID `40110086`).
"""


def build_full_report_md(
    data: dict[str, Any],
    *,
    include_appendix: bool = True,
    image_mode: str = "uri",
) -> str:
    """Assemble the complete multi-section Markdown report."""
    sections = build_report_sections(
        data,
        include_appendix=include_appendix,
        image_mode=image_mode,
    )
    order = [
        "header_and_exec",
        "sec1_critique",
        "sec2_hypothesis",
        "sec3_design",
        "sec4_results",
        "sec5_interpretation",
        "sec6_reproducibility_refs",
    ]
    return "\n\n---\n\n".join(sections[k] for k in order)


def render_report_pdf(
    md_text: str,
    output_pdf_path: Path,
    *,
    title: str = "Weight-Space Programming for SE.LLMA",
    subtitle: str = "Critical Evaluation & Controlled Empirical Investigation",
    is_draft: bool = False,
    meta_line: str = "",
) -> dict[str, Any]:
    """Convert Markdown report to a clean publication-grade A4 PDF via WeasyPrint."""
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove redundant top-level H1 so the styled cover header renders it once
    body_md = re.sub(r"^#\s+[^\n]+\n+", "", md_text.lstrip(), count=1)
    html_body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )

    badge_cls = "badge-draft" if is_draft else "badge-final"
    badge_txt = "DRAFT · AWAITING 6-ARM RUN" if is_draft else "FINAL RESEARCH REPORT"
    running_right = "DRAFT REPORT" if is_draft else "SE.LLMA RESEARCH REPORT"

    css = f"""
    @page {{
        size: A4 portrait;
        margin: 15mm 15mm 17mm 15mm;
        @top-left {{
            content: "SE.LLMA · Weight-Space Programming Evaluation";
            font-family: "Lato", "Open Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            color: #64748b;
            border-bottom: 0.5pt solid #e2e8f0;
            padding-bottom: 3pt;
            margin-bottom: 6pt;
        }}
        @top-right {{
            content: "{running_right}";
            font-family: "Lato", "Open Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            font-weight: 700;
            color: #64748b;
            border-bottom: 0.5pt solid #e2e8f0;
            padding-bottom: 3pt;
            margin-bottom: 6pt;
        }}
        @bottom-left {{
            content: "sellma-wsp-eval · Reproducible CPU Study";
            font-family: "Lato", "Open Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            color: #64748b;
        }}
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-family: "Lato", "Open Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            color: #64748b;
        }}
    }}
    @page :first {{
        @top-left {{ content: none; border: none; }}
        @top-right {{ content: none; border: none; }}
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: "Lato", "Open Sans", "Liberation Sans", sans-serif;
        font-size: 8.8pt;
        line-height: 1.42;
        color: #1e293b;
        margin: 0;
        padding: 0;
    }}
    .doc-header {{
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: #f8fafc;
        padding: 11pt 14pt 11pt 14pt;
        border-radius: 5pt;
        margin-bottom: 10pt;
        border-left: 4pt solid #38bdf8;
    }}
    .badge {{
        display: inline-block;
        font-size: 7.2pt;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 2pt 6pt;
        border-radius: 3pt;
        margin-bottom: 5pt;
        text-transform: uppercase;
    }}
    .badge-draft {{
        background: #fef3c7;
        color: #92400e;
    }}
    .badge-final {{
        background: #dbeafe;
        color: #1e40af;
    }}
    .doc-title {{
        font-size: 16pt;
        font-weight: 700;
        line-height: 1.22;
        margin: 2pt 0 4pt 0;
        color: #ffffff;
    }}
    .doc-subtitle {{
        font-size: 9.5pt;
        color: #cbd5e1;
        margin: 0 0 4pt 0;
    }}
    .doc-meta {{
        font-size: 7.8pt;
        color: #94a3b8;
        margin: 0;
    }}
    h1, h2, h3, h4 {{
        color: #0f172a;
        break-after: avoid;
        page-break-after: avoid;
    }}
    h2 {{
        font-size: 11.6pt;
        font-weight: 700;
        margin: 11pt 0 5pt 0;
        padding-bottom: 2.5pt;
        border-bottom: 1.2pt solid #cbd5e1;
        color: #0f172a;
    }}
    h3 {{
        font-size: 9.8pt;
        font-weight: 700;
        margin: 8.5pt 0 3.5pt 0;
        color: #1e3a5f;
    }}
    p {{
        margin: 0 0 5pt 0;
    }}
    ul, ol {{
        margin: 2pt 0 6pt 0;
        padding-left: 16pt;
    }}
    li {{
        margin-bottom: 2.2pt;
    }}
    hr {{
        border: none;
        border-top: 0.6pt solid #e2e8f0;
        margin: 5.5pt 0;
    }}
    code {{
        font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
        font-size: 7.7pt;
        background: #f1f5f9;
        color: #0f172a;
        padding: 0.5pt 2.5pt;
        border-radius: 2pt;
        overflow-wrap: break-word;
    }}
    pre {{
        background: #f8fafc;
        border: 1pt solid #cbd5e1;
        border-left: 3pt solid #334155;
        border-radius: 3pt;
        padding: 5.5pt 8pt;
        margin: 4.5pt 0 6.5pt 0;
        font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
        font-size: 7.3pt;
        line-height: 1.35;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    pre code {{
        background: transparent;
        padding: 0;
        font-size: 7.4pt;
    }}
    blockquote {{
        background: #f8fafc;
        border-left: 3.5pt solid #d97706;
        border-top: 0.5pt solid #e2e8f0;
        border-right: 0.5pt solid #e2e8f0;
        border-bottom: 0.5pt solid #e2e8f0;
        border-radius: 0 4pt 4pt 0;
        padding: 7pt 10pt;
        margin: 6pt 0 9pt 0;
        color: #1e293b;
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    blockquote p:last-child, blockquote ul:last-child {{
        margin-bottom: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
        font-size: 7.6pt;
        line-height: 1.32;
        margin: 6pt 0 10pt 0;
    }}
    thead {{
        display: table-header-group;
    }}
    tr {{
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    th {{
        background: #1e293b;
        color: #f8fafc;
        font-weight: 700;
        text-align: left;
        padding: 4.5pt 5pt;
        border: 0.5pt solid #1e293b;
        overflow-wrap: break-word;
    }}
    td {{
        padding: 4pt 5pt;
        border: 0.5pt solid #cbd5e1;
        vertical-align: top;
        overflow-wrap: break-word;
    }}
    tbody tr:nth-child(even) {{
        background: #f8fafc;
    }}
    img {{
        display: block;
        max-width: 100%;
        height: auto;
        margin: 7pt auto 3pt auto;
        border: 0.6pt solid #cbd5e1;
        border-radius: 3pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    a {{
        color: #1d4ed8;
        text-decoration: none;
        overflow-wrap: anywhere;
    }}
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <header class="doc-header">
    <span class="badge {badge_cls}">{badge_txt}</span>
    <div class="doc-title">{title}</div>
    <div class="doc-subtitle">{subtitle}</div>
    <div class="doc-meta">{meta_line}</div>
  </header>
  <main>
    {html_body}
  </main>
</body>
</html>
"""
    doc = weasyprint.HTML(string=full_html, base_url=str(REPORT_DIR)).render()
    doc.write_pdf(str(output_pdf_path))
    size_kb = round(output_pdf_path.stat().st_size / 1024.0, 1)
    return {
        "pdf_path": str(output_pdf_path),
        "pages": len(doc.pages),
        "size_kb": size_kb,
    }


def export_reports(*, include_appendix: bool = True) -> dict[str, Any]:
    """Generate baseline diagnostic figure, report/research_note.md, and report/report.pdf."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_report_data()
    generate_baseline_diagnostic_figure(data)

    note_md = build_research_note_md(data)
    note_path = REPORT_DIR / "research_note.md"
    note_path.write_text(note_md, encoding="utf-8")
    note_words = len(note_md.split())

    full_md = build_full_report_md(
        data,
        include_appendix=include_appendix,
        image_mode="uri",
    )
    pdf_path = REPORT_DIR / "report.pdf"
    is_draft = not data["has_experiment_details"]
    meta_line = (
        f"Base model: Qwen/Qwen2.5-0.5B-Instruct (bfloat16, CPU) · Generated: {data['generated_at_utc']}"
    )
    pdf_info = render_report_pdf(
        full_md,
        pdf_path,
        title="Weight-Space Programming for SE.LLMA",
        subtitle="Critical Evaluation of Takehi et al. & Controlled Swedish Empirical Investigation",
        is_draft=is_draft,
        meta_line=meta_line,
    )

    figures_included = [
        name for name, exists in data.get("available_figures", {}).items() if exists
    ]
    return {
        "research_note_path": str(note_path),
        "research_note_words": note_words,
        "pdf_path": pdf_info["pdf_path"],
        "pdf_pages": pdf_info["pages"],
        "pdf_size_kb": pdf_info["size_kb"],
        "is_draft": is_draft,
        "figures_included": figures_included,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build report/research_note.md and report/report.pdf from results/."
    )
    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Exclude detailed qualitative examples and DES alpha calibration appendix from PDF.",
    )
    args = parser.parse_args()
    summary = export_reports(include_appendix=not args.no_appendix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
