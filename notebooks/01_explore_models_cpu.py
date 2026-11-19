import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="CPU Prompt & Resource Benchmark — SE.LLMA",
)


@app.cell
def _():
    import gc
    import json
    import os
    import threading
    import time
    from pathlib import Path

    import marimo as mo
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        gc,
        json,
        mo,
        os,
        psutil,
        results_dir,
        threading,
        time,
        torch,
    )


@app.cell
def _(mo, os, psutil, torch):
    _vm = psutil.virtual_memory()
    _cpu_count = os.cpu_count() or 4
    mo.md(
        f"""
        # CPU Model Comparison & Resource Tracking

        This notebook runs **the same prompt** sequentially across the selected models
        (explicitly freeing memory between models to stay within RAM limits) while sampling
        **CPU (%)**, **process RSS RAM**, **system RAM**, **load time**, and **tokens/s** in the background.

        - **Detected logical CPUs:** `{_cpu_count}` threads (current `torch` threads: `{torch.get_num_threads()}`)
        - **System RAM:** `{_vm.total / (1024**3):.2f} GiB` total · `{_vm.available / (1024**3):.2f} GiB` available (`{_vm.percent:.1f}%` used)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, os):
    _cpu_count = os.cpu_count() or 4
    _model_options = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "AI-Sweden-Models/gpt-sw3-126m",
    ]

    prompt_input = mo.ui.text_area(
        value="Förklara kortfattat vad artificiell intelligens är och ge två konkreta exempel från vardagen.",
        label="Prompt (same prompt sent to all selected models)",
        rows=3,
        full_width=True,
    )
    system_input = mo.ui.text_area(
        value="Du är en hjälpsam assistent som alltid svarar på korrekt och naturlig svenska.",
        label="System prompt (optional, used for Instruct/Chat models)",
        rows=2,
        full_width=True,
    )
    model_selector = mo.ui.multiselect(
        options=_model_options,
        value=["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"],
        label="Models to compare",
    )
    custom_model_input = mo.ui.text(
        value="",
        label="Additional Hugging Face model ID (optional)",
        placeholder="e.g. Qwen/Qwen3-0.6B",
        full_width=True,
    )
    dtype_select = mo.ui.dropdown(
        options={
            "bfloat16 (recommended for ~3 GB free RAM)": "bfloat16",
            "float32 (higher RAM usage, maximum compatibility)": "float32",
        },
        value="bfloat16 (recommended for ~3 GB free RAM)",
        label="Precision (dtype)",
    )
    max_tokens_slider = mo.ui.slider(
        start=16,
        stop=256,
        step=16,
        value=80,
        label="max_new_tokens",
    )
    threads_slider = mo.ui.slider(
        start=1,
        stop=_cpu_count,
        step=1,
        value=min(8, _cpu_count),
        label="PyTorch CPU threads",
    )
    save_json_checkbox = mo.ui.checkbox(
        value=True,
        label="Save results and metrics to results/cpu_prompt_benchmark.json",
    )
    run_btn = mo.ui.run_button(label="Run prompt across selected models")

    mo.vstack(
        [
            prompt_input,
            system_input,
            mo.hstack([model_selector, dtype_select], widths="equal"),
            custom_model_input,
            mo.hstack([max_tokens_slider, threads_slider, save_json_checkbox], widths="equal"),
            run_btn,
        ]
    )
    return (
        custom_model_input,
        dtype_select,
        max_tokens_slider,
        model_selector,
        prompt_input,
        run_btn,
        save_json_checkbox,
        system_input,
        threads_slider,
    )


@app.cell(hide_code=True)
def _(AutoModelForCausalLM, AutoTokenizer, gc, psutil, threading, time, torch):
    class ResourceMonitor:
        """Sample process RSS RAM and CPU usage in the background during execution."""

        def __init__(self, interval_s: float = 0.1):
            self.interval_s = interval_s
            self.process = psutil.Process()
            self.samples: list[dict] = []
            self.phase = "init"
            self._stop_event = threading.Event()
            self._thread: threading.Thread | None = None
            self._t0 = 0.0

        def start(self) -> None:
            self._t0 = time.perf_counter()
            self.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
            self._record_sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def set_phase(self, phase: str) -> None:
            self.phase = phase

        def _record_sample(self) -> None:
            vm = psutil.virtual_memory()
            rss_mb = self.process.memory_info().rss / (1024**2)
            proc_cpu = self.process.cpu_percent(interval=None)
            sys_cpu = psutil.cpu_percent(interval=None)
            self.samples.append(
                {
                    "t_s": round(time.perf_counter() - self._t0, 3),
                    "phase": self.phase,
                    "rss_mb": round(rss_mb, 1),
                    "proc_cpu_pct": round(proc_cpu, 1),
                    "sys_cpu_pct": round(sys_cpu, 1),
                    "sys_ram_used_mb": round((vm.total - vm.available) / (1024**2), 1),
                    "sys_ram_avail_mb": round(vm.available / (1024**2), 1),
                    "sys_ram_pct": round(vm.percent, 1),
                }
            )

        def _loop(self) -> None:
            while not self._stop_event.wait(self.interval_s):
                self._record_sample()

        def stop(self) -> list[dict]:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._record_sample()
            return self.samples


    def _format_prompt(tokenizer, prompt: str, system_prompt: str) -> str:
        has_template = bool(getattr(tokenizer, "chat_template", None))
        if has_template:
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


    def benchmark_model(
        model_id: str,
        prompt: str,
        system_prompt: str,
        dtype_name: str,
        max_new_tokens: int,
        num_threads: int,
    ) -> dict:
        gc.collect()
        torch.set_num_threads(num_threads)
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

        monitor = ResourceMonitor(interval_s=0.1)
        baseline_rss_mb = psutil.Process().memory_info().rss / (1024**2)
        model = None
        tokenizer = None

        monitor.set_phase("load")
        monitor.start()
        t_load_start = time.perf_counter()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.eval()
            load_time_s = time.perf_counter() - t_load_start

            formatted_prompt = _format_prompt(tokenizer, prompt, system_prompt)
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            input_tokens = int(inputs["input_ids"].shape[-1])

            monitor.set_phase("generate")
            t_gen_start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_time_s = time.perf_counter() - t_gen_start

            new_tokens_ids = output_ids[0][input_tokens:]
            generated_tokens = int(new_tokens_ids.shape[-1])
            response_text = tokenizer.decode(new_tokens_ids, skip_special_tokens=True).strip()
            samples = monitor.stop()

            rss_values = [s["rss_mb"] for s in samples]
            sys_cpu_values = [s["sys_cpu_pct"] for s in samples[1:]] or [0.0]
            proc_cpu_values = [s["proc_cpu_pct"] for s in samples[1:]] or [0.0]
            sys_ram_pcts = [s["sys_ram_pct"] for s in samples]

            return {
                "model_id": model_id,
                "status": "ok",
                "dtype": dtype_name,
                "load_time_s": round(load_time_s, 2),
                "gen_time_s": round(gen_time_s, 2),
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "tokens_per_sec": round(generated_tokens / max(gen_time_s, 1e-6), 2),
                "baseline_rss_mb": round(baseline_rss_mb, 1),
                "peak_rss_mb": round(max(rss_values), 1),
                "delta_rss_mb": round(max(rss_values) - baseline_rss_mb, 1),
                "avg_sys_cpu_pct": round(sum(sys_cpu_values) / len(sys_cpu_values), 1),
                "peak_sys_cpu_pct": round(max(sys_cpu_values), 1),
                "avg_proc_cpu_pct": round(sum(proc_cpu_values) / len(proc_cpu_values), 1),
                "peak_proc_cpu_pct": round(max(proc_cpu_values), 1),
                "peak_sys_ram_pct": round(max(sys_ram_pcts), 1),
                "response_text": response_text,
                "samples": samples,
            }
        except Exception as exc:
            samples = monitor.stop()
            return {
                "model_id": model_id,
                "status": "error",
                "dtype": dtype_name,
                "error": f"{type(exc).__name__}: {exc}",
                "samples": samples,
            }
        finally:
            del model
            del tokenizer
            gc.collect()

    return (benchmark_model,)


@app.cell(hide_code=True)
def _(
    benchmark_model,
    custom_model_input,
    dtype_select,
    json,
    max_tokens_slider,
    mo,
    model_selector,
    prompt_input,
    psutil,
    results_dir,
    run_btn,
    save_json_checkbox,
    system_input,
    threads_slider,
    time,
):
    def _build_svg_timeline(
        runs: list[dict],
        metric_key: str,
        peak_key: str,
        y_label: str,
        unit: str,
        skip_first: bool = False,
    ) -> str:
        ok_runs = [r for r in runs if r.get("status") == "ok" and r.get("samples")]
        if not ok_runs:
            return ""
        colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
        width, height, pad = 760, 220, 42
        series_list = [
            (r, r["samples"][1:] if skip_first and len(r["samples"]) > 1 else r["samples"])
            for r in ok_runs
        ]
        max_t = max(s["t_s"] for _, ss in series_list for s in ss) or 1.0
        max_val = max(s[metric_key] for _, ss in series_list for s in ss) or 1.0
        min_val = min(s[metric_key] for _, ss in series_list for s in ss)
        span = max(max_val - min_val, 10.0)

        polylines = []
        legend_items = []
        for idx, (run, ss) in enumerate(series_list):
            color = colors[idx % len(colors)]
            pts = []
            for s in ss:
                x = pad + (s["t_s"] / max_t) * (width - 2 * pad)
                y = (height - pad) - ((s[metric_key] - min_val) / span) * (height - 2 * pad)
                pts.append(f"{x:.1f},{y:.1f}")
            polylines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}" />'
            )
            short_name = run["model_id"].split("/")[-1]
            legend_items.append(
                f'<span style="color:{color};font-weight:600;margin-right:14px;">■ {short_name} '
                f'(peak {run[peak_key]}{unit})</span>'
            )

        return (
            f'<div style="margin-top:8px;">'
            f'<div style="font-size:13px;margin-bottom:6px;">{"".join(legend_items)}</div>'
            f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;border:1px solid #e5e7eb;border-radius:6px;background:#fafafa;">'
            f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9ca3af" />'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9ca3af" />'
            f'<text x="{pad}" y="20" font-size="11" fill="#374151">{y_label}: {min_val:.0f} → {max_val:.0f}{unit}</text>'
            f'<text x="{width - pad - 110}" y="{height - 12}" font-size="11" fill="#374151">Time (0 → {max_t:.1f}s)</text>'
            f'{"".join(polylines)}'
            f'</svg></div>'
        )


    if not run_btn.value:
        _vm = psutil.virtual_memory()
        view = mo.callout(
            mo.md(
                f"Configure the prompt and models above, then click **Run prompt across selected models**.\n\n"
                f"Current memory status: **{_vm.available / (1024**3):.2f} GiB free** "
                f"out of {_vm.total / (1024**3):.2f} GiB ({_vm.percent:.1f}% used)."
            ),
            kind="info",
        )
    else:
        _models = list(model_selector.value)
        _custom = custom_model_input.value.strip()
        if _custom and _custom not in _models:
            _models.append(_custom)

        if not _models:
            view = mo.callout(mo.md("Select at least one model to run."), kind="warn")
        else:
            _runs = []
            for _m in _models:
                _res = benchmark_model(
                    model_id=_m,
                    prompt=prompt_input.value,
                    system_prompt=system_input.value,
                    dtype_name=dtype_select.value,
                    max_new_tokens=int(max_tokens_slider.value),
                    num_threads=int(threads_slider.value),
                )
                _runs.append(_res)

            if save_json_checkbox.value:
                _out_path = results_dir / "cpu_prompt_benchmark.json"
                _payload = {
                    "timestamp_unix": round(time.time(), 2),
                    "prompt": prompt_input.value,
                    "system_prompt": system_input.value,
                    "dtype": dtype_select.value,
                    "max_new_tokens": int(max_tokens_slider.value),
                    "torch_threads": int(threads_slider.value),
                    "runs": _runs,
                }
                _out_path.write_text(json.dumps(_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            _rows_md = [
                "| Model | Status | Load (s) | Gen (s) | Tokens | Speed (tok/s) | Peak RSS (MB) | Δ RAM (MB) | Proc CPU avg/peak (%) | Sys CPU avg/peak (%) | Peak Sys RAM (%) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            _resp_sections = []
            for _r in _runs:
                if _r["status"] == "ok":
                    _rows_md.append(
                        f"| `{_r['model_id']}` | OK | {_r['load_time_s']} | {_r['gen_time_s']} | "
                        f"{_r['generated_tokens']} | {_r['tokens_per_sec']} | {_r['peak_rss_mb']} | "
                        f"+{_r['delta_rss_mb']} | {_r['avg_proc_cpu_pct']}% / {_r['peak_proc_cpu_pct']}% | "
                        f"{_r['avg_sys_cpu_pct']}% / {_r['peak_sys_cpu_pct']}% | {_r['peak_sys_ram_pct']}% |"
                    )
                    _resp_sections.append(
                        f"### `{_r['model_id']}`\n"
                        f"*Load: {_r['load_time_s']}s · Generation: {_r['gen_time_s']}s ({_r['tokens_per_sec']} tok/s) · "
                        f"Peak RSS: {_r['peak_rss_mb']} MB (+{_r['delta_rss_mb']} MB) · "
                        f"CPU (proc avg/peak): {_r['avg_proc_cpu_pct']}% / {_r['peak_proc_cpu_pct']}%*\n\n"
                        f"> {_r['response_text']}"
                    )
                else:
                    _rows_md.append(
                        f"| `{_r['model_id']}` | ERROR | - | - | - | - | - | - | - | - | - |"
                    )
                    _resp_sections.append(
                        f"### `{_r['model_id']}`\n**Error:** `{_r.get('error', 'Unknown error')}`"
                    )

            _ram_svg = _build_svg_timeline(
                _runs,
                metric_key="rss_mb",
                peak_key="peak_rss_mb",
                y_label="RSS RAM (MB)",
                unit=" MB",
            )
            _cpu_svg = _build_svg_timeline(
                _runs,
                metric_key="sys_cpu_pct",
                peak_key="peak_sys_cpu_pct",
                y_label="System CPU (%)",
                unit="%",
                skip_first=True,
            )
            _blocks = [
                mo.md("## Results & Resource Usage\n\n" + "\n".join(_rows_md)),
            ]
            if _ram_svg:
                _blocks.append(mo.md("### Process RAM (RSS) Timeline During Load & Generation"))
                _blocks.append(mo.Html(_ram_svg))
            if _cpu_svg:
                _blocks.append(mo.md("### System CPU (%) Timeline During Load & Generation"))
                _blocks.append(mo.Html(_cpu_svg))
            _blocks.append(mo.md("## Generated Responses\n\n" + "\n\n---\n\n".join(_resp_sections)))
            if save_json_checkbox.value:
                _blocks.append(
                    mo.callout(
                        mo.md("Metrics saved to `results/cpu_prompt_benchmark.json`."),
                        kind="success",
                    )
                )
            view = mo.vstack(_blocks)

    view
    return


if __name__ == "__main__":
    app.run()
