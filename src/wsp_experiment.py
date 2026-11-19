"""Core experiment runner for Weight-Space Programming vs. Awake Mapping & LoRA on CPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "swedish_test_suite.json"
BASELINE_EVAL_PATH = ROOT_DIR / "results" / "qwen_baseline_eval.json"
BASELINE_RESP_PATH = ROOT_DIR / "results" / "qwen_baseline_responses.json"
RESULTS_DIR = ROOT_DIR / "results"

STRICT_SV_SYSTEM_PROMPT = (
    "Du är en expert på korrekt och idiomatisk svenska. "
    "Svara alltid helt på svenska utan engelska, danska eller påhittade ord. "
    "För Python-uppgifter: svara enbart med en kort, fungerande funktion i ett ```python```-kodblock "
    "utan extra testkod eller långa kommentarer."
)

SWEDISH_FUNCTION_WORDS = [
    "och",
    "att",
    "det",
    "som",
    "för",
    "med",
    "inte",
    "den",
    "till",
    "har",
    "av",
    "om",
    "på",
    "ett",
    "är",
    "var",
    "från",
    "men",
    "vid",
    "ska",
    "eller",
    "kan",
    "när",
    "ut",
    "så",
    "blir",
    "vara",
    "efter",
    "över",
    "under",
    "mycket",
    "också",
    "bara",
    "mellan",
    "svenska",
    "Sverige",
]

ENGLISH_FUNCTION_WORDS = [
    "the",
    "and",
    "that",
    "with",
    "for",
    "not",
    "this",
    "from",
    "have",
    "which",
    "will",
    "when",
    "between",
    "after",
    "should",
    "could",
    "about",
    "would",
    "there",
    "their",
    "other",
    "because",
    "into",
    "more",
    "only",
    "very",
    "also",
    "English",
    "answer",
    "example",
]


def _pearsonr(x: list[float], y: list[float]) -> float:
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    ax = ax - ax.mean()
    ay = ay - ay.mean()
    denom = float(np.sqrt(np.sum(ax * ax) * np.sum(ay * ay)))
    return float(np.sum(ax * ay) / denom) if denom > 1e-12 else 0.0


def _rankdata(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearmanr(x: list[float], y: list[float]) -> float:
    return _pearsonr(_rankdata(x).tolist(), _rankdata(y).tolist())


def _format_prompt(tokenizer, prompt: str, system_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{prompt}"
    return prompt


def _resolve_system_prompt(item: dict, system_override: str | None) -> str:
    if system_override is not None and item.get("category", "").endswith("_sv"):
        return system_override
    return item.get("system_prompt", "")


def _build_supervised_ids(
    tokenizer,
    item: dict,
    ref_text: str,
    system_override: str | None = None,
    max_ref_tokens: int = 96,
) -> tuple[torch.Tensor, int]:
    sys_prompt = _resolve_system_prompt(item, system_override)
    prompt_str = _format_prompt(tokenizer, item["prompt"], sys_prompt)
    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
    eos_tok = tokenizer.eos_token or ""
    ref_ids = tokenizer(
        ref_text + eos_tok,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][:, :max_ref_tokens]
    input_ids = torch.cat([prompt_ids, ref_ids], dim=1)
    return input_ids, int(prompt_ids.shape[1])


def _compute_target_nll(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    shift_hidden = hidden[:, prompt_len - 1 : -1, :]
    shift_labels = input_ids[:, prompt_len:]
    logits = model.lm_head(shift_hidden).float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))


def evaluate_nll(
    model,
    tokenizer,
    items: list[dict],
    ref_map: dict[str, str],
    system_override: str | None = None,
) -> dict:
    model.eval()
    per_item: dict[str, dict] = {}
    sv_target_losses: list[float] = []
    sv_non_neg_losses: list[float] = []
    en_ctrl_losses: list[float] = []
    all_losses: list[float] = []

    with torch.inference_mode():
        for item in items:
            pid = item["prompt_id"]
            ref_text = ref_map[pid]
            input_ids, prompt_len = _build_supervised_ids(
                tokenizer, item, ref_text, system_override=system_override
            )
            nll = float(_compute_target_nll(model, input_ids, prompt_len).item())
            cat = item["category"]
            per_item[pid] = {"category": cat, "priority": item["priority"], "nll": round(nll, 4)}
            all_losses.append(nll)
            if cat in ("everyday_sv", "public_services_sv"):
                sv_target_losses.append(nll)
            elif cat in ("reasoning_sv", "coding_sv"):
                sv_non_neg_losses.append(nll)
            elif cat == "control_en":
                en_ctrl_losses.append(nll)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    sv_t_nll = _mean(sv_target_losses)
    en_nll = _mean(en_ctrl_losses)
    return {
        "sv_target_nll": round(sv_t_nll, 4),
        "sv_target_ppl": round(math.exp(min(sv_t_nll, 20.0)), 2),
        "sv_non_neg_nll": round(_mean(sv_non_neg_losses), 4),
        "en_ctrl_nll": round(en_nll, 4),
        "en_ctrl_ppl": round(math.exp(min(en_nll, 20.0)), 2),
        "overall_nll": round(_mean(all_losses), 4),
        "per_item": per_item,
    }


def _make_response_id(run_name: str, prompt_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{run_name}:{prompt_id}:{seed}".encode("utf-8")).hexdigest()
    return f"resp_{digest[:12]}"


def _extract_python_code(response_text: str) -> str:
    py_block = re.search(r"```python\s*\n(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    if py_block:
        return py_block.group(1).strip()
    any_block = re.search(r"```\w*\s*\n(.*?)```", response_text, flags=re.DOTALL)
    if any_block:
        return any_block.group(1).strip()
    return response_text.strip()


def _run_auto_checks(item: dict, response_text: str) -> dict:
    contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", response_text))

    expected = item.get("expected_answer")
    expected_answer_match = None
    if expected is not None:
        pattern = rf"(?<!\w){re.escape(str(expected))}(?!\w)"
        expected_answer_match = bool(re.search(pattern, response_text, flags=re.IGNORECASE))

    test_cases = item.get("test_cases") or []
    code_exec_pass = None
    code_exec_error = None
    if test_cases:
        code_src = _extract_python_code(response_text)
        script = f"{code_src}\n\n" + "\n".join(test_cases) + "\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=3,
            )
            code_exec_pass = proc.returncode == 0
            if proc.returncode != 0:
                err_lines = (proc.stderr or proc.stdout or "Non-zero exit").strip().splitlines()
                code_exec_error = err_lines[-1] if err_lines else "Execution failed"
        except subprocess.TimeoutExpired:
            code_exec_pass = False
            code_exec_error = "TimeoutExpired (3s)"
        except Exception as exc:
            code_exec_pass = False
            code_exec_error = f"{type(exc).__name__}: {exc}"

    return {
        "contains_cjk": contains_cjk,
        "expected_answer_match": expected_answer_match,
        "code_exec_pass": code_exec_pass,
        "code_exec_error": code_exec_error,
    }


def generate_eval_responses(
    model,
    tokenizer,
    items: list[dict],
    run_name: str,
    model_id: str,
    model_revision: str,
    seed: int,
    max_new_tokens: int = 128,
    system_override: str | None = None,
    output_path: Path | None = None,
) -> dict:
    model.eval()
    process = psutil.Process()
    baseline_rss_mb = process.memory_info().rss / (1024**2)
    peak_rss_mb = baseline_rss_mb
    peak_sys_ram_pct = psutil.virtual_memory().percent
    total_gen_tokens = 0
    total_gen_time = 0.0
    responses: list[dict] = []

    for item in items:
        sys_prompt = _resolve_system_prompt(item, system_override)
        formatted = _format_prompt(tokenizer, item["prompt"], sys_prompt)
        inputs = tokenizer(formatted, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0
        new_ids = output_ids[0][input_tokens:]
        gen_tokens = int(new_ids.shape[-1])
        resp_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        total_gen_tokens += gen_tokens
        total_gen_time += dt
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
        peak_sys_ram_pct = max(peak_sys_ram_pct, psutil.virtual_memory().percent)

        responses.append(
            {
                "prompt_id": item["prompt_id"],
                "response_id": _make_response_id(run_name, item["prompt_id"], seed),
                "split": item["split"],
                "category": item["category"],
                "priority": item["priority"],
                "prompt": item["prompt"],
                "response_text": resp_text,
                "input_tokens": input_tokens,
                "generated_tokens": gen_tokens,
                "gen_time_s": round(dt, 2),
                "tokens_per_sec": round(gen_tokens / max(dt, 1e-6), 2),
                "auto_checks": _run_auto_checks(item, resp_text),
            }
        )

    payload = {
        "metadata": {
            "run_name": run_name,
            "model_id": model_id,
            "model_revision": model_revision,
            "split_filter": "eval",
            "dtype": "bfloat16",
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "timestamp_unix": round(time.time(), 2),
            "num_prompts": len(responses),
            "total_gen_time_s": round(total_gen_time, 2),
            "avg_tokens_per_sec": round(total_gen_tokens / max(total_gen_time, 1e-6), 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "peak_sys_ram_pct": round(peak_sys_ram_pct, 1),
        },
        "responses": responses,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _compute_static_language_direction(model, tokenizer) -> torch.Tensor:
    vocab = tokenizer.get_vocab()
    sv_ids: list[int] = []
    en_ids: list[int] = []

    for tok_str, tid in vocab.items():
        if any(ch in tok_str for ch in ("å", "ä", "ö", "Å", "Ä", "Ö")):
            sv_ids.append(tid)

    for word in SWEDISH_FUNCTION_WORDS:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            if len(ids) == 1:
                sv_ids.append(ids[0])

    for word in ENGLISH_FUNCTION_WORDS:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            if len(ids) == 1:
                en_ids.append(ids[0])

    sv_ids = sorted(set(sv_ids))
    en_ids = sorted(set(en_ids))
    lm_w = model.lm_head.weight.detach()
    sv_mean = lm_w[sv_ids].float().mean(dim=0)
    en_mean = lm_w[en_ids].float().mean(dim=0)
    return F.normalize(sv_mean - en_mean, dim=0)


def _score_down_proj_columns(model, direction: torch.Tensor) -> tuple[list[float], torch.Tensor]:
    """Return per-layer top-64 mean cosine score and full [num_layers, intermediate_size] matrix."""
    scores_matrix = []
    layer_scores = []
    dir_norm = F.normalize(direction.float(), dim=0)
    for layer in model.model.layers:
        w = layer.mlp.down_proj.weight.detach().float()  # [hidden_size, intermediate_size]
        col_norm = F.normalize(w, dim=0)
        cos = torch.matmul(dir_norm, col_norm)  # [intermediate_size]
        scores_matrix.append(cos)
        topk_mean = float(torch.topk(cos, k=64).values.mean().item())
        layer_scores.append(round(topk_mean, 5))
    return layer_scores, torch.stack(scores_matrix, dim=0)


def _apply_column_scaling(
    model,
    coords: list[tuple[int, int]],
    alpha: float,
) -> tuple[dict[int, torch.Tensor], int, float]:
    """Scale selected (layer_idx, col_idx) columns in mlp.down_proj.weight and return exact backups."""
    by_layer: dict[int, list[int]] = {}
    for layer_idx, col_idx in coords:
        by_layer.setdefault(layer_idx, []).append(col_idx)

    backups: dict[int, torch.Tensor] = {}
    sq_diff = 0.0
    edited_params = 0
    with torch.no_grad():
        for layer_idx, cols in by_layer.items():
            w = model.model.layers[layer_idx].mlp.down_proj.weight
            backups[layer_idx] = w.detach().clone()
            col_tensor = torch.tensor(cols, dtype=torch.long, device=w.device)
            orig_cols = w[:, col_tensor].float()
            delta_cols = orig_cols * alpha
            sq_diff += float((delta_cols * delta_cols).sum().item())
            edited_params += int(orig_cols.numel())
            w[:, col_tensor] = (orig_cols * (1.0 + alpha)).to(w.dtype)
    return backups, edited_params, round(math.sqrt(sq_diff), 4)


def _restore_backups(model, backups: dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for layer_idx, orig in backups.items():
            model.model.layers[layer_idx].mlp.down_proj.weight.copy_(orig)


class LoRALinear(nn.Module):
    """Minimal frozen-base LoRA linear wrapper in pure PyTorch."""

    def __init__(self, base_linear: nn.Linear, r: int = 4, lora_alpha: float = 8.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(
            torch.empty(r, base_linear.in_features, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_linear.out_features, r, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_f = x.float()
        lora_out = F.linear(F.linear(x_f, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def _compute_dev_activation_direction(
    model,
    tokenizer,
    dev_sv_target: list[dict],
    dev_en_ctrl: list[dict],
) -> torch.Tensor:
    sv_vecs = []
    en_vecs = []
    model.eval()
    with torch.inference_mode():
        for item in dev_sv_target:
            text = _format_prompt(tokenizer, item["prompt"], item.get("system_prompt", ""))
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            out = model.model(input_ids=ids, use_cache=False)
            sv_vecs.append(out.last_hidden_state[0, -1, :].float())
        for item in dev_en_ctrl:
            text = _format_prompt(tokenizer, item["prompt"], item.get("system_prompt", ""))
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            out = model.model(input_ids=ids, use_cache=False)
            en_vecs.append(out.last_hidden_state[0, -1, :].float())
    return F.normalize(torch.stack(sv_vecs).mean(dim=0) - torch.stack(en_vecs).mean(dim=0), dim=0)


def _generate_plots(
    static_scores: list[float],
    des_rows: list[dict],
    selected_layers: list[int],
    arm_summaries: dict[str, dict],
) -> None:
    layers = [r["layer"] for r in des_rows]
    delta_sv = [r["delta_sv_nll"] for r in des_rows]
    delta_stop = [r["delta_stop_nll"] for r in des_rows]
    selectivity = [r["selectivity"] for r in des_rows]
    pearson_r = _pearsonr(static_scores, delta_sv)
    spearman_r = _spearmanr(static_scores, delta_sv)

    # Figure 1: Layer mapping static vs DES
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1 = axes[0]
    static_arr = np.array(static_scores)
    static_norm = (static_arr - static_arr.min()) / max(static_arr.max() - static_arr.min(), 1e-8)
    ax1.plot(layers, static_norm, marker="o", ms=4, label="Static WSP alignment (norm.)", color="#64748b")
    ax1_r = ax1.twinx()
    ax1_r.plot(layers, delta_sv, marker="s", ms=4, label="Awake DES ΔNLL (Swedish target)", color="#2563eb")
    ax1_r.plot(layers, delta_stop, marker="^", ms=4, linestyle="--", label="Awake DES ΔNLL (Stop-point)", color="#dc2626")
    ax1.set_title(f"Static WSP vs. Awake DES Mapping\nPearson r={pearson_r:.2f}, Spearman ρ={spearman_r:.2f}")
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Normalized Static Score", color="#64748b")
    ax1_r.set_ylabel("ΔNLL under 0.85× Perturbation", color="#2563eb")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    ax2 = axes[1]
    bar_colors = ["#059669" if l in selected_layers else "#94a3b8" for l in layers]
    ax2.bar(layers, selectivity, color=bar_colors)
    ax2.set_title("DES Swedish-to-Stop-Point Selectivity by Layer\n(Green = Top-4 DES Selected Layers)")
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Selectivity: max(ΔSV,0) / (|ΔStop| + 0.01)")
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "layer_mapping_static_vs_des.png", dpi=160)
    plt.close(fig)

    # Figure 2: Arm comparison NLL + Trade-off
    arm_order = [
        ("baseline", "Baseline"),
        ("wsp_static", "WSP Static"),
        ("des_awake_edit", "DES Awake"),
        ("random_edit", "Random (3-seed)"),
        ("prompt_control", "Prompt Ctrl"),
        ("micro_lora", "Micro-LoRA"),
    ]
    labels = [lbl for _, lbl in arm_order]
    sv_vals = [arm_summaries[k]["eval_nll"]["sv_target_nll"] for k, _ in arm_order]
    sv_errs = [arm_summaries[k].get("sv_target_nll_std", 0.0) for k, _ in arm_order]
    en_vals = [arm_summaries[k]["eval_nll"]["en_ctrl_nll"] for k, _ in arm_order]
    base_sv = arm_summaries["baseline"]["eval_nll"]["sv_target_nll"]
    base_en = arm_summaries["baseline"]["eval_nll"]["en_ctrl_nll"]

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#475569", "#f59e0b", "#059669", "#ef4444", "#3b82f6", "#7c3aed"]
    x = np.arange(len(labels))
    axes2[0].bar(x, sv_vals, yerr=sv_errs, capsize=4, color=colors)
    axes2[0].axhline(base_sv, color="#475569", linestyle=":", linewidth=1.2)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes2[0].set_ylabel("Held-Out Swedish Target NLL (lower = better)")
    axes2[0].set_title("Held-Out Swedish Target NLL Across All 6 Arms")
    axes2[0].grid(axis="y", alpha=0.25)

    offsets = {
        "baseline": (-42, -16),
        "wsp_static": (8, 12),
        "prompt_control": (10, -14),
        "des_awake_edit": (8, -4),
        "random_edit": (8, 4),
        "micro_lora": (-62, -14),
    }
    for idx, (k, lbl) in enumerate(arm_order):
        dx = sv_vals[idx] - base_sv
        dy = en_vals[idx] - base_en
        axes2[1].scatter(dx, dy, color=colors[idx], s=85, zorder=3)
        axes2[1].annotate(
            lbl,
            (dx, dy),
            textcoords="offset points",
            xytext=offsets.get(k, (6, 4)),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.6),
        )
    axes2[1].axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
    axes2[1].axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
    axes2[1].margins(x=0.14, y=0.16)
    axes2[1].set_xlabel("Δ Swedish Target NLL vs. Baseline (left = better)")
    axes2[1].set_ylabel("Δ English Stop-Point NLL vs. Baseline (up = regression)")
    axes2[1].set_title("Target Improvement vs. Stop-Point Regression")
    axes2[1].grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "arm_comparison_nll.png", dpi=160)
    plt.close(fig2)


def run_experiment(
    nll_only: bool = False,
    num_threads: int = 8,
    seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)
        else:
            print(msg, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suite = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    prompts = suite["prompts"]
    dev_items = [p for p in prompts if p["split"] == "dev"]
    eval_items = [p for p in prompts if p["split"] == "eval"]

    eval_records = json.loads(BASELINE_EVAL_PATH.read_text(encoding="utf-8"))["evaluations"]
    ref_map = {e["prompt_id"]: e["reference_answer"] for e in eval_records}

    dev_sv_target = [p for p in dev_items if p["priority"] == "target"]
    dev_non_neg = [p for p in dev_items if p["priority"] == "non_negotiable"]
    dev_en_ctrl = [p for p in dev_items if p["category"] == "control_en"]

    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    process = psutil.Process()
    baseline_rss_mb = process.memory_info().rss / (1024**2)

    try:
        snap_dir = snapshot_download(model_id, local_files_only=True)
        model_revision = Path(snap_dir).name
        local_only = True
    except Exception:
        model_revision = "unknown"
        local_only = False

    _log(f"Loading {model_id} (rev={model_revision}, bfloat16, threads={num_threads})...")
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    model.eval()
    if model_revision == "unknown":
        model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
    load_time_s = round(time.perf_counter() - t_load0, 2)
    peak_rss_mb = process.memory_info().rss / (1024**2)

    arm_summaries: dict[str, dict] = {}

    try:
        # 1. Baseline
        _log("[1/6] Evaluating baseline...")
        t0 = time.perf_counter()
        base_eval_nll = evaluate_nll(model, tokenizer, eval_items, ref_map)
        base_dev_nll = evaluate_nll(model, tokenizer, dev_items, ref_map)
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
        arm_summaries["baseline"] = {
            "name": "1. Unchanged Baseline",
            "eval_nll": base_eval_nll,
            "dev_nll": base_dev_nll,
            "edited_params": 0,
            "edit_l2_norm": 0.0,
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
        }

        # 2. WSP Static (zero forward passes for localization/selection)
        _log("[2/6] Running WSP Static (data-free weight inspection)...")
        t0 = time.perf_counter()
        v_sv_en = _compute_static_language_direction(model, tokenizer)
        static_layer_scores, static_cos_matrix = _score_down_proj_columns(model, v_sv_en)
        flat_vals, flat_idx = torch.topk(static_cos_matrix.reshape(-1), k=256)
        inter_size = static_cos_matrix.shape[1]
        static_coords = [(int(idx // inter_size), int(idx % inter_size)) for idx in flat_idx.tolist()]
        static_alpha = 0.15
        backups, edited_params, edit_l2 = _apply_column_scaling(model, static_coords, static_alpha)
        wsp_eval_nll = evaluate_nll(model, tokenizer, eval_items, ref_map)
        wsp_gen_info = None
        if not nll_only:
            _log("  Generating 20 held-out responses for wsp_static...")
            wsp_gen_info = generate_eval_responses(
                model,
                tokenizer,
                eval_items,
                run_name="qwen_wsp_static",
                model_id=model_id,
                model_revision=model_revision,
                seed=seed,
                output_path=RESULTS_DIR / "qwen_wsp_static_responses.json",
            )["metadata"]
        _restore_backups(model, backups)
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
        arm_summaries["wsp_static"] = {
            "name": "2. Static Weight Edit (WSP)",
            "alpha": static_alpha,
            "eval_nll": wsp_eval_nll,
            "edited_params": edited_params,
            "edit_l2_norm": edit_l2,
            "top_layers": sorted({l for l, _ in static_coords}),
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "gen_metadata": wsp_gen_info,
        }

        # 3. DES Awake Perturbation Mapping (Martín-Fernández analogy on balanced dev subset)
        _log("[3/6] Running Awake DES layer probing on dev (24 layers)...")
        t0 = time.perf_counter()
        des_probe_sv = dev_sv_target[:3]
        des_probe_stop = [
            next(p for p in dev_non_neg if p["category"] == cat)
            for cat in ("reasoning_sv", "coding_sv", "control_en")
        ]
        base_dev_sv = evaluate_nll(model, tokenizer, des_probe_sv, ref_map)["sv_target_nll"]
        base_dev_stop = evaluate_nll(model, tokenizer, des_probe_stop, ref_map)["overall_nll"]

        des_rows: list[dict] = []
        num_layers_total = len(model.model.layers)
        for layer_idx in range(num_layers_total):
            w = model.model.layers[layer_idx].mlp.down_proj.weight
            orig_w = w.detach().clone()
            with torch.no_grad():
                w.mul_(0.85)
            sv_pert = evaluate_nll(model, tokenizer, des_probe_sv, ref_map)["sv_target_nll"]
            stop_pert = evaluate_nll(model, tokenizer, des_probe_stop, ref_map)["overall_nll"]
            with torch.no_grad():
                w.copy_(orig_w)
            del orig_w
            delta_sv = sv_pert - base_dev_sv
            delta_stop = stop_pert - base_dev_stop
            sel = max(delta_sv, 0.0) / (abs(delta_stop) + 0.01)
            des_rows.append(
                {
                    "layer": layer_idx,
                    "static_score": static_layer_scores[layer_idx],
                    "delta_sv_nll": round(delta_sv, 5),
                    "delta_stop_nll": round(delta_stop, 5),
                    "selectivity": round(sel, 5),
                }
            )
            if (layer_idx + 1) % 4 == 0 or (layer_idx + 1) == num_layers_total:
                cur_rss = process.memory_info().rss / (1024**2)
                peak_rss_mb = max(peak_rss_mb, cur_rss)
                _log(f"  DES probing: layer {layer_idx + 1}/{num_layers_total} · RSS={cur_rss:.0f} MB")
        gc.collect()

        selected_layers = [
            r["layer"]
            for r in sorted(des_rows, key=lambda r: r["selectivity"], reverse=True)[:4]
        ]
        dev_dir = _compute_dev_activation_direction(model, tokenizer, dev_sv_target, dev_en_ctrl)
        _, awake_cos_matrix = _score_down_proj_columns(model, dev_dir)
        sub_scores = awake_cos_matrix[selected_layers]
        _, sub_flat_idx = torch.topk(sub_scores.reshape(-1), k=256)
        des_coords = [
            (selected_layers[int(idx // inter_size)], int(idx % inter_size))
            for idx in sub_flat_idx.tolist()
        ]

        # Calibrate alpha on dev subject to stop-point constraint
        base_dev_en = base_dev_nll["en_ctrl_nll"]
        best_alpha = 0.05
        best_dev_sv = float("inf")
        alpha_trials = []
        for cand_alpha in (0.05, 0.10, 0.15, -0.10):
            bkp, _, _ = _apply_column_scaling(model, des_coords, cand_alpha)
            dev_res = evaluate_nll(model, tokenizer, dev_items, ref_map)
            _restore_backups(model, bkp)
            delta_en_dev = dev_res["en_ctrl_nll"] - base_dev_en
            feasible = delta_en_dev <= 0.05
            alpha_trials.append(
                {
                    "alpha": cand_alpha,
                    "dev_sv_target_nll": dev_res["sv_target_nll"],
                    "delta_en_ctrl_dev_nll": round(delta_en_dev, 4),
                    "feasible": feasible,
                }
            )
            if feasible and dev_res["sv_target_nll"] < best_dev_sv:
                best_dev_sv = dev_res["sv_target_nll"]
                best_alpha = cand_alpha

        backups, edited_params, edit_l2 = _apply_column_scaling(model, des_coords, best_alpha)
        des_eval_nll = evaluate_nll(model, tokenizer, eval_items, ref_map)
        des_gen_info = None
        if not nll_only:
            _log("  Generating 20 held-out responses for des_awake_edit...")
            des_gen_info = generate_eval_responses(
                model,
                tokenizer,
                eval_items,
                run_name="qwen_des_awake",
                model_id=model_id,
                model_revision=model_revision,
                seed=seed,
                output_path=RESULTS_DIR / "qwen_des_awake_responses.json",
            )["metadata"]
        _restore_backups(model, backups)
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))

        delta_sv_list = [r["delta_sv_nll"] for r in des_rows]
        layer_corr = {
            "pearson_r": round(_pearsonr(static_layer_scores, delta_sv_list), 4),
            "spearman_rho": round(_spearmanr(static_layer_scores, delta_sv_list), 4),
        }
        arm_summaries["des_awake_edit"] = {
            "name": "3. Awake DES Edit (Martín-Fernández)",
            "selected_layers": selected_layers,
            "best_alpha": best_alpha,
            "alpha_trials": alpha_trials,
            "eval_nll": des_eval_nll,
            "edited_params": edited_params,
            "edit_l2_norm": edit_l2,
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "gen_metadata": des_gen_info,
        }

        # 4. Random / Untargeted Edit (3 seeds)
        _log("[4/6] Running Random Edit control (3 seeds: 42, 123, 999)...")
        t0 = time.perf_counter()
        num_layers = len(model.model.layers)
        rand_seed_runs = []
        rand_gen_info = None
        for r_seed in (42, 123, 999):
            rng = random.Random(r_seed)
            rand_coords = rng.sample(
                [(l, c) for l in range(num_layers) for c in range(inter_size)],
                k=256,
            )
            bkp, r_params, r_l2 = _apply_column_scaling(model, rand_coords, 0.15)
            r_eval = evaluate_nll(model, tokenizer, eval_items, ref_map)
            if not nll_only and r_seed == 42:
                _log("  Generating 20 held-out responses for random_edit (seed=42)...")
                rand_gen_info = generate_eval_responses(
                    model,
                    tokenizer,
                    eval_items,
                    run_name="qwen_random_edit",
                    model_id=model_id,
                    model_revision=model_revision,
                    seed=r_seed,
                    output_path=RESULTS_DIR / "qwen_random_edit_responses.json",
                )["metadata"]
            _restore_backups(model, bkp)
            rand_seed_runs.append(
                {
                    "seed": r_seed,
                    "edited_params": r_params,
                    "edit_l2_norm": r_l2,
                    "eval_nll": r_eval,
                }
            )
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            arr = np.asarray(vals, dtype=np.float64)
            return round(float(arr.mean()), 4), round(float(arr.std(ddof=0)), 4)

        sv_m, sv_s = _mean_std([x["eval_nll"]["sv_target_nll"] for x in rand_seed_runs])
        nn_m, _ = _mean_std([x["eval_nll"]["sv_non_neg_nll"] for x in rand_seed_runs])
        en_m, en_s = _mean_std([x["eval_nll"]["en_ctrl_nll"] for x in rand_seed_runs])
        ov_m, _ = _mean_std([x["eval_nll"]["overall_nll"] for x in rand_seed_runs])
        l2_m, _ = _mean_std([x["edit_l2_norm"] for x in rand_seed_runs])
        arm_summaries["random_edit"] = {
            "name": "4. Random / Untargeted Edit (3 seeds)",
            "seeds": [42, 123, 999],
            "seed_runs": rand_seed_runs,
            "eval_nll": {
                "sv_target_nll": sv_m,
                "sv_target_ppl": round(math.exp(min(sv_m, 20.0)), 2),
                "sv_non_neg_nll": nn_m,
                "en_ctrl_nll": en_m,
                "en_ctrl_ppl": round(math.exp(min(en_m, 20.0)), 2),
                "overall_nll": ov_m,
            },
            "sv_target_nll_std": sv_s,
            "en_ctrl_nll_std": en_s,
            "edited_params": rand_seed_runs[0]["edited_params"],
            "edit_l2_norm": l2_m,
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "gen_metadata": rand_gen_info,
        }

        # 5a. Prompt / Inference Control
        _log("[5/6] Running Prompting / Inference control...")
        t0 = time.perf_counter()
        prompt_eval_nll = evaluate_nll(
            model,
            tokenizer,
            eval_items,
            ref_map,
            system_override=STRICT_SV_SYSTEM_PROMPT,
        )
        prompt_gen_info = None
        if not nll_only:
            _log("  Generating 20 held-out responses for prompt_control...")
            prompt_gen_info = generate_eval_responses(
                model,
                tokenizer,
                eval_items,
                run_name="qwen_prompt_control",
                model_id=model_id,
                model_revision=model_revision,
                seed=seed,
                system_override=STRICT_SV_SYSTEM_PROMPT,
                output_path=RESULTS_DIR / "qwen_prompt_control_responses.json",
            )["metadata"]
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
        arm_summaries["prompt_control"] = {
            "name": "5a. Prompting / Inference Control",
            "eval_nll": prompt_eval_nll,
            "edited_params": 0,
            "edit_l2_norm": 0.0,
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "gen_metadata": prompt_gen_info,
        }

        # 5b. Micro-LoRA (layers 18..23 mlp.down_proj, r=4, frozen bfloat16 base)
        _log("[6/6] Training Micro-LoRA on dev (layers 18..23, r=4, 18 steps)...")
        t0 = time.perf_counter()
        for p in model.parameters():
            p.requires_grad_(False)

        lora_layers = list(range(18, 24))
        orig_linears: dict[int, nn.Linear] = {}
        lora_params: list[nn.Parameter] = []
        for idx in lora_layers:
            orig = model.model.layers[idx].mlp.down_proj
            orig_linears[idx] = orig
            wrapped = LoRALinear(orig, r=4, lora_alpha=8.0)
            model.model.layers[idx].mlp.down_proj = wrapped
            lora_params.extend([wrapped.lora_A, wrapped.lora_B])

        trainable_params = sum(p.numel() for p in lora_params)
        optimizer = torch.optim.AdamW(lora_params, lr=5e-4)
        rng_lora = random.Random(seed)
        train_order = []
        while len(train_order) < 18 * 4:
            batch_items = list(dev_items)
            rng_lora.shuffle(batch_items)
            train_order.extend(batch_items)

        lora_losses: list[float] = []
        model.train()
        for step in range(18):
            optimizer.zero_grad()
            step_loss = 0.0
            for micro in range(4):
                item = train_order[step * 4 + micro]
                input_ids, prompt_len = _build_supervised_ids(tokenizer, item, ref_map[item["prompt_id"]])
                loss = _compute_target_nll(model, input_ids, prompt_len) / 4.0
                loss.backward()
                step_loss += float(loss.item())
                peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            lora_losses.append(round(step_loss, 4))
            if (step + 1) % 6 == 0:
                cur_rss = process.memory_info().rss / (1024**2)
                _log(f"  Micro-LoRA step {step + 1}/18 · loss={step_loss:.4f} · RSS={cur_rss:.0f} MB")

        model.eval()
        lora_sq = 0.0
        with torch.no_grad():
            for idx in lora_layers:
                wr = model.model.layers[idx].mlp.down_proj
                dw = (wr.lora_B @ wr.lora_A) * wr.scaling
                lora_sq += float((dw * dw).sum().item())
        lora_l2 = round(math.sqrt(lora_sq), 4)

        lora_eval_nll = evaluate_nll(model, tokenizer, eval_items, ref_map)
        lora_gen_info = None
        if not nll_only:
            _log("  Generating 20 held-out responses for micro_lora...")
            lora_gen_info = generate_eval_responses(
                model,
                tokenizer,
                eval_items,
                run_name="qwen_micro_lora",
                model_id=model_id,
                model_revision=model_revision,
                seed=seed,
                output_path=RESULTS_DIR / "qwen_micro_lora_responses.json",
            )["metadata"]

        for idx, orig in orig_linears.items():
            model.model.layers[idx].mlp.down_proj = orig
        del optimizer, lora_params
        gc.collect()
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))

        arm_summaries["micro_lora"] = {
            "name": "5b. Micro-LoRA (layers 18..23, r=4)",
            "lora_layers": lora_layers,
            "steps": 18,
            "grad_accum": 4,
            "loss_curve": lora_losses,
            "eval_nll": lora_eval_nll,
            "edited_params": trainable_params,
            "edit_l2_norm": lora_l2,
            "runtime_s": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "gen_metadata": lora_gen_info,
        }

        _log("Generating Matplotlib figures...")
        _generate_plots(static_layer_scores, des_rows, selected_layers, arm_summaries)

        details = {
            "metadata": {
                "model_id": model_id,
                "revision": model_revision,
                "dtype": "bfloat16",
                "seed": seed,
                "random_control_seeds": [42, 123, 999],
                "torch_threads": num_threads,
                "nll_only": nll_only,
                "load_time_s": load_time_s,
                "baseline_rss_mb": round(baseline_rss_mb, 1),
                "peak_rss_mb": round(peak_rss_mb, 1),
                "timestamp_unix": round(time.time(), 2),
            },
            "layer_mapping": {
                "static_vs_des_correlation": layer_corr,
                "selected_des_layers": selected_layers,
                "per_layer": des_rows,
            },
            "arms": arm_summaries,
        }
        (RESULTS_DIR / "experiment_details.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Build summary.json compatible with notebooks/report.py
        base_sv_nll = arm_summaries["baseline"]["eval_nll"]["sv_target_nll"]
        base_en_nll = arm_summaries["baseline"]["eval_nll"]["en_ctrl_nll"]
        summary_conditions = []
        for k in (
            "baseline",
            "wsp_static",
            "des_awake_edit",
            "random_edit",
            "prompt_control",
            "micro_lora",
        ):
            arm = arm_summaries[k]
            ev = arm["eval_nll"]
            d_sv = ev["sv_target_nll"] - base_sv_nll
            d_en = ev["en_ctrl_nll"] - base_en_nll
            std_txt = f" ± {arm['sv_target_nll_std']}" if "sv_target_nll_std" in arm else ""
            summary_conditions.append(
                {
                    "name": arm["name"],
                    "target_score": f"NLL {ev['sv_target_nll']}{std_txt} (PPL {ev['sv_target_ppl']}, Δ {d_sv:+.3f})",
                    "regression_score": f"EN NLL {ev['en_ctrl_nll']} (Δ {d_en:+.3f}) | Non-Neg {ev['sv_non_neg_nll']}",
                    "cost": f"{arm['runtime_s']}s CPU | {arm['edited_params']:,} params (L2 {arm['edit_l2_norm']})",
                    "notes": (
                        "Data-free lm_head->down_proj top-256 edit"
                        if k == "wsp_static"
                        else f"DES top-4 layers {arm['selected_layers']}, alpha={arm['best_alpha']}"
                        if k == "des_awake_edit"
                        else "3-seed mean ± std"
                        if k == "random_edit"
                        else "Strict Swedish system prompt"
                        if k == "prompt_control"
                        else f"18 steps on dev (loss {arm['loss_curve'][0]}->{arm['loss_curve'][-1]})"
                        if k == "micro_lora"
                        else "Reference model"
                    ),
                }
            )

        summary_payload = {
            "metadata": {
                "model_id": model_id,
                "revision": model_revision,
                "seeds": "42 (main), [42, 123, 999] (random control)",
            },
            "conditions": summary_conditions,
            "examples": [],
        }
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _log("Experiment complete. Saved results/experiment_details.json, results/summary.json, and PNG plots.")
        return details
    finally:
        del model, tokenizer
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6-arm CPU WSP experiment on Qwen2.5-0.5B-Instruct.")
    parser.add_argument("--nll-only", action="store_true", help="Skip 20-prompt held-out generation (fast ~2-3 min pass).")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_experiment(nll_only=args.nll_only, num_threads=args.threads, seed=args.seed)


if __name__ == "__main__":
    main()
