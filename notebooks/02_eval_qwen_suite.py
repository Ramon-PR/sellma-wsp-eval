import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Swedish Diagnostic Suite Evaluation — SE.LLMA",
)


@app.cell
def _():
    import gc
    import hashlib
    import json
    import os
    import re
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root_dir = Path(__file__).resolve().parent.parent
    suite_path = root_dir / "data" / "swedish_test_suite.json"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        gc,
        hashlib,
        json,
        mo,
        os,
        psutil,
        re,
        results_dir,
        subprocess,
        suite_path,
        sys,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(json, mo, suite_path):
    if not suite_path.exists():
        suite_data = {"prompts": []}
        _header_md = mo.callout(
            mo.md(f"Test suite not found at `{suite_path}`."),
            kind="danger",
        )
    else:
        suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
        _prompts = suite_data.get("prompts", [])
        _categories: dict[str, dict] = {}
        for _p in _prompts:
            _cat = _p["category"]
            if _cat not in _categories:
                _categories[_cat] = {
                    "priority": _p.get("priority", "-"),
                    "dev": 0,
                    "eval": 0,
                    "total": 0,
                }
            _split = _p.get("split", "eval")
            if _split in ("dev", "eval"):
                _categories[_cat][_split] += 1
            _categories[_cat]["total"] += 1

        _rows = [
            "| Category | Priority | `dev` | `eval` | Total |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for _cat, _counts in _categories.items():
            _rows.append(
                f"| `{_cat}` | `{_counts['priority']}` | {_counts['dev']} | {_counts['eval']} | {_counts['total']} |"
            )

        _header_md = mo.md(
            "\n".join(
                [
                    "# Swedish Diagnostic Suite Evaluation — SE.LLMA",
                    "",
                    "This notebook runs an open-weight model across a structured Swedish evaluation and regression suite",
                    "(`data/swedish_test_suite.json`), saves responses with opaque `response_id` hashes and resource metrics",
                    "to `results/<run_name>_responses.json`, and conditionally loads external human/LLM grading from",
                    "`results/<run_name>_eval.json` if present.",
                    "",
                    f"**Suite version:** `{suite_data.get('version', 'unknown')}` · **Total prompts:** `{len(_prompts)}`",
                    "",
                    *_rows,
                ]
            )
        )

    _header_md
    return (suite_data,)


@app.cell(hide_code=True)
def _(mo, os):
    _cpu_count = os.cpu_count() or 4

    run_name_input = mo.ui.text(
        value="qwen_baseline",
        label="Run name (controls output & eval filenames)",
    )
    model_id_input = mo.ui.text(
        value="Qwen/Qwen2.5-0.5B-Instruct",
        label="Hugging Face model ID or local path",
        full_width=True,
    )
    split_select = mo.ui.dropdown(
        options={
            "All (dev + eval)": "all",
            "dev only": "dev",
            "eval only": "eval",
        },
        value="All (dev + eval)",
        label="Split to run",
    )
    dtype_select = mo.ui.dropdown(
        options={
            "bfloat16 (recommended for CPU RAM)": "bfloat16",
            "float32": "float32",
        },
        value="bfloat16 (recommended for CPU RAM)",
        label="Precision (dtype)",
    )
    max_tokens_slider = mo.ui.slider(
        start=32,
        stop=256,
        step=16,
        value=128,
        label="max_new_tokens",
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
        label="Random seed",
    )
    run_btn = mo.ui.run_button(label="Run model on test suite and save responses")

    mo.vstack(
        [
            mo.hstack([run_name_input, model_id_input], widths=[1, 2]),
            mo.hstack([split_select, dtype_select, seed_input], widths="equal"),
            mo.hstack([max_tokens_slider, threads_slider], widths="equal"),
            run_btn,
        ]
    )
    return (
        dtype_select,
        max_tokens_slider,
        model_id_input,
        run_btn,
        run_name_input,
        seed_input,
        split_select,
        threads_slider,
    )


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    gc,
    hashlib,
    json,
    mo,
    psutil,
    re,
    subprocess,
    sys,
    time,
    torch,
):
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


    def run_suite(
        prompts: list[dict],
        run_name: str,
        model_id: str,
        split_filter: str,
        dtype_name: str,
        max_new_tokens: int,
        num_threads: int,
        seed: int,
        output_path=None,
    ) -> dict:
        selected = [
            p for p in prompts if split_filter == "all" or p.get("split") == split_filter
        ]

        gc.collect()
        torch.manual_seed(seed)
        torch.set_num_threads(num_threads)
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

        process = psutil.Process()
        baseline_rss_mb = process.memory_info().rss / (1024**2)
        peak_rss_mb = baseline_rss_mb
        peak_sys_ram_pct = psutil.virtual_memory().percent

        model = None
        tokenizer = None
        results: list[dict] = []

        t_load_start = time.perf_counter()
        try:
            with mo.status.spinner(title=f"Loading {model_id} ({dtype_name})..."):
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                model.eval()
            load_time_s = round(time.perf_counter() - t_load_start, 2)
            model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

            vm_after_load = psutil.virtual_memory()
            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
            peak_sys_ram_pct = max(peak_sys_ram_pct, vm_after_load.percent)

            total_gen_tokens = 0
            total_gen_time = 0.0

            def _build_payload() -> dict:
                return {
                    "metadata": {
                        "run_name": run_name,
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "split_filter": split_filter,
                        "dtype": dtype_name,
                        "max_new_tokens": max_new_tokens,
                        "torch_threads": num_threads,
                        "seed": seed,
                        "timestamp_unix": round(time.time(), 2),
                        "num_prompts": len(results),
                        "total_selected": len(selected),
                        "load_time_s": load_time_s,
                        "total_gen_time_s": round(total_gen_time, 2),
                        "avg_tokens_per_sec": round(
                            total_gen_tokens / max(total_gen_time, 1e-6), 2
                        ),
                        "baseline_rss_mb": round(baseline_rss_mb, 1),
                        "peak_rss_mb": round(peak_rss_mb, 1),
                        "delta_rss_mb": round(peak_rss_mb - baseline_rss_mb, 1),
                        "peak_sys_ram_pct": round(peak_sys_ram_pct, 1),
                    },
                    "responses": results,
                }

            with mo.status.progress_bar(
                total=len(selected),
                title="Running model on test suite",
                subtitle=f"0/{len(selected)} prompts",
                completion_title="Suite run complete",
                completion_subtitle=f"{len(selected)}/{len(selected)} prompts saved",
            ) as _bar:
                for _idx, item in enumerate(selected, start=1):
                    formatted = _format_prompt(
                        tokenizer,
                        item["prompt"],
                        item.get("system_prompt", ""),
                    )
                    inputs = tokenizer(formatted, return_tensors="pt")
                    input_tokens = int(inputs["input_ids"].shape[-1])

                    t_gen_start = time.perf_counter()
                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen_time_s = time.perf_counter() - t_gen_start
                    new_ids = output_ids[0][input_tokens:]
                    generated_tokens = int(new_ids.shape[-1])
                    response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                    total_gen_tokens += generated_tokens
                    total_gen_time += gen_time_s

                    peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
                    peak_sys_ram_pct = max(peak_sys_ram_pct, psutil.virtual_memory().percent)

                    _tok_s = round(generated_tokens / max(gen_time_s, 1e-6), 2)
                    auto_checks = _run_auto_checks(item, response_text)
                    results.append(
                        {
                            "prompt_id": item["prompt_id"],
                            "response_id": _make_response_id(run_name, item["prompt_id"], seed),
                            "split": item["split"],
                            "category": item["category"],
                            "priority": item["priority"],
                            "prompt": item["prompt"],
                            "response_text": response_text,
                            "input_tokens": input_tokens,
                            "generated_tokens": generated_tokens,
                            "gen_time_s": round(gen_time_s, 2),
                            "tokens_per_sec": _tok_s,
                            "auto_checks": auto_checks,
                        }
                    )
                    if output_path is not None:
                        output_path.write_text(
                            json.dumps(_build_payload(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    _bar.update(
                        increment=1,
                        subtitle=f"{_idx}/{len(selected)} · {item['prompt_id']} ({_tok_s:.1f} tok/s)",
                    )

            return _build_payload()
        finally:
            del model
            del tokenizer
            gc.collect()
    return (_run_auto_checks, run_suite)


@app.cell(hide_code=True)
def _(
    dtype_select,
    json,
    max_tokens_slider,
    mo,
    model_id_input,
    results_dir,
    run_btn,
    run_name_input,
    run_suite,
    seed_input,
    split_select,
    suite_data,
    threads_slider,
):
    _clean_run_name = run_name_input.value.strip() or "qwen_baseline"
    responses_path = results_dir / f"{_clean_run_name}_responses.json"

    _run_error = None
    if run_btn.value:
        _prompts = suite_data.get("prompts", [])
        if not _prompts:
            _run_error = "Test suite is empty; check `data/swedish_test_suite.json`."
        else:
            try:
                _payload = run_suite(
                    prompts=_prompts,
                    run_name=_clean_run_name,
                    model_id=model_id_input.value.strip(),
                    split_filter=split_select.value,
                    dtype_name=dtype_select.value,
                    max_new_tokens=int(max_tokens_slider.value),
                    num_threads=int(threads_slider.value),
                    seed=int(seed_input.value),
                    output_path=responses_path,
                )
                responses_path.write_text(
                    json.dumps(_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as _exc:
                _run_error = f"{type(_exc).__name__}: {_exc}"

    if _run_error:
        responses_view = mo.callout(mo.md(f"**Run failed:** `{_run_error}`"), kind="danger")
    elif not responses_path.exists():
        responses_view = mo.callout(
            mo.md(
                f"No saved responses found yet at `results/{responses_path.name}`.\n\n"
                "Click **Run model on test suite and save responses** above to generate them."
            ),
            kind="info",
        )
    else:
        _saved = json.loads(responses_path.read_text(encoding="utf-8"))
        _meta = _saved.get("metadata", {})
        _resps = _saved.get("responses", [])

        _cjk_count = sum(1 for _r in _resps if _r.get("auto_checks", {}).get("contains_cjk"))
        _reas_items = [
            _r for _r in _resps
            if _r.get("auto_checks", {}).get("expected_answer_match") is not None
        ]
        _reas_pass = sum(
            1 for _r in _reas_items if _r["auto_checks"]["expected_answer_match"]
        )
        _code_items = [
            _r for _r in _resps
            if _r.get("auto_checks", {}).get("code_exec_pass") is not None
        ]
        _code_pass = sum(
            1 for _r in _code_items if _r["auto_checks"]["code_exec_pass"]
        )

        _table_rows = [
            "| `prompt_id` | `response_id` | Split | Category | Tokens | Speed (tok/s) | Auto Checks |",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :--- |",
        ]
        _detail_blocks = []
        for _r in _resps:
            _ac = _r.get("auto_checks", {})
            _flags = []
            if _ac.get("contains_cjk"):
                _flags.append("CJK_LEAK")
            if _ac.get("expected_answer_match") is True:
                _flags.append("ANS_OK")
            elif _ac.get("expected_answer_match") is False:
                _flags.append("ANS_FAIL")
            if _ac.get("code_exec_pass") is True:
                _flags.append("CODE_PASS")
            elif _ac.get("code_exec_pass") is False:
                _flags.append(f"CODE_FAIL ({_ac.get('code_exec_error', '')})")
            _flag_str = ", ".join(_flags) if _flags else "manual/LLM eval"

            _table_rows.append(
                f"| `{_r['prompt_id']}` | `{_r['response_id']}` | `{_r['split']}` | "
                f"`{_r['category']}` | {_r['generated_tokens']} | {_r['tokens_per_sec']} | {_flag_str} |"
            )
            _detail_blocks.append(
                "\n".join(
                    [
                        f"#### `{_r['prompt_id']}` (`{_r['response_id']}` · `{_r['split']}` · `{_r['category']}`)",
                        f"**Prompt:** {_r['prompt']}",
                        "",
                        f"**Model output** *({_r['gen_time_s']}s, {_r['tokens_per_sec']} tok/s, checks: {_flag_str})*:",
                        "",
                        f"> {_r['response_text'].replace(chr(10), chr(10) + '> ')}",
                    ]
                )
            )

        _summary_md = "\n".join(
            [
                f"## Saved Model Responses (`results/{responses_path.name}`)",
                "",
                f"- **Model:** `{_meta.get('model_id')}` (rev `{_meta.get('model_revision')}`)",
                f"- **Config:** split=`{_meta.get('split_filter')}`, dtype=`{_meta.get('dtype')}`, "
                f"seed=`{_meta.get('seed')}`, threads=`{_meta.get('torch_threads')}`, "
                f"max_new_tokens=`{_meta.get('max_new_tokens')}`",
                f"- **Resources:** load=`{_meta.get('load_time_s')}s`, total_gen=`{_meta.get('total_gen_time_s')}s`, "
                f"avg_speed=`{_meta.get('avg_tokens_per_sec')} tok/s`, peak_RSS=`{_meta.get('peak_rss_mb')} MB` "
                f"(+{_meta.get('delta_rss_mb')} MB), peak_sys_RAM=`{_meta.get('peak_sys_ram_pct')}%`",
                f"- **Automatic Checks:** CJK leaks=`{_cjk_count}/{len(_resps)}` · "
                f"Expected-answer matches=`{_reas_pass}/{len(_reas_items)}` · "
                f"Python unit tests passed=`{_code_pass}/{len(_code_items)}`",
                "",
                *_table_rows,
            ]
        )
        _details_md = "\n\n---\n\n".join(_detail_blocks)
        responses_view = mo.vstack(
            [
                mo.md(_summary_md),
                mo.accordion({"Show all generated responses": mo.md(_details_md)}),
            ]
        )

    responses_view
    return (responses_path,)


@app.cell(hide_code=True)
def _(json, mo, responses_path, results_dir, run_name_input):
    _clean_run_name = run_name_input.value.strip() or "qwen_baseline"
    eval_path = results_dir / f"{_clean_run_name}_eval.json"

    if not eval_path.exists():
        _schema_example = """{
      "run_name": "qwen_baseline",
      "evaluator": "blind_llm_plus_auto",
      "evaluations": [
        {
          "prompt_id": "ev_dev_01",
          "response_id": "resp_...",
          "is_correct": false,
          "swedish_quality": 1,
          "task_correctness": 1,
          "failure_modes": ["code_switching", "grammar"],
          "notes": "Brief explanation of what succeeded or failed.",
          "reference_answer": "Correct Swedish example answer."
        }
      ]
    }"""
        eval_view = mo.callout(
            mo.md(
                "\n".join(
                    [
                        f"**External evaluation file not found** (`results/{eval_path.name}`): skipped loading.",
                        "",
                        "Once responses are generated, run the external evaluation step to create this file with schema:",
                        "```json",
                        _schema_example,
                        "```",
                    ]
                )
            ),
            kind="info",
        )
    else:
        _eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        _evals = _eval_data.get("evaluations", [])

        _resp_map: dict[str, dict] = {}
        if responses_path.exists():
            _saved = json.loads(responses_path.read_text(encoding="utf-8"))
            _resp_map = {r["prompt_id"]: r for r in _saved.get("responses", [])}

        _joined = []
        _id_mismatches = []
        for _ev in _evals:
            _pid = _ev["prompt_id"]
            _resp_rec = _resp_map.get(_pid, {})
            if _resp_rec and _ev.get("response_id") and _resp_rec.get("response_id") != _ev.get("response_id"):
                _id_mismatches.append(_pid)
            _joined.append(
                {
                    "prompt_id": _pid,
                    "response_id": _ev.get("response_id") or _resp_rec.get("response_id", "-"),
                    "split": _resp_rec.get("split", _ev.get("split", "-")),
                    "category": _resp_rec.get("category", _ev.get("category", "-")),
                    "priority": _resp_rec.get("priority", _ev.get("priority", "-")),
                    "prompt": _resp_rec.get("prompt", _ev.get("prompt", "")),
                    "response_text": _resp_rec.get("response_text", ""),
                    "is_correct": bool(_ev.get("is_correct", False)),
                    "swedish_quality": _ev.get("swedish_quality", 0),
                    "task_correctness": _ev.get("task_correctness", 0),
                    "failure_modes": _ev.get("failure_modes", []),
                    "notes": _ev.get("notes", ""),
                    "reference_answer": _ev.get("reference_answer", ""),
                }
            )

        _total = len(_joined)
        _passed = sum(1 for _x in _joined if _x["is_correct"])
        _acc = (100.0 * _passed / _total) if _total else 0.0
        _mean_sv = sum(_x["swedish_quality"] for _x in _joined) / _total if _total else 0.0
        _mean_task = sum(_x["task_correctness"] for _x in _joined) / _total if _total else 0.0

        def _group_table(group_key: str, title: str) -> str:
            _groups: dict[str, list[dict]] = {}
            for _row in _joined:
                _k = str(_row.get(group_key, "-"))
                _groups.setdefault(_k, []).append(_row)
            _lines = [
                f"### {title}",
                f"| {title} | Correct | Accuracy | Mean Swedish (0–2) | Mean Task (0–2) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
            for _k, _items in _groups.items():
                _n = len(_items)
                _ok = sum(1 for _i in _items if _i["is_correct"])
                _sv = sum(_i["swedish_quality"] for _i in _items) / _n
                _tk = sum(_i["task_correctness"] for _i in _items) / _n
                _lines.append(
                    f"| `{_k}` | {_ok}/{_n} | {100.0 * _ok / _n:.1f}% | {_sv:.2f} | {_tk:.2f} |"
                )
            return "\n".join(_lines)

        _fm_counts: dict[str, int] = {}
        for _row in _joined:
            for _fm in _row.get("failure_modes") or []:
                _fm_counts[_fm] = _fm_counts.get(_fm, 0) + 1
        _fm_lines = [
            "### Failure Modes Breakdown",
            "| Failure Mode | Count |",
            "| :--- | :---: |",
        ]
        if _fm_counts:
            for _fm, _cnt in sorted(_fm_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                _fm_lines.append(f"| `{_fm}` | {_cnt} |")
        else:
            _fm_lines.append("| *(none recorded)* | 0 |")

        _failed_items = [_x for _x in _joined if not _x["is_correct"]]
        _passed_items = [_x for _x in _joined if _x["is_correct"]]

        def _format_cards(items: list[dict]) -> str:
            _cards = []
            for _it in items:
                _verdict = "PASS" if _it["is_correct"] else "FAIL"
                _fms = ", ".join(_it["failure_modes"]) if _it["failure_modes"] else "none"
                _cards.append(
                    "\n".join(
                        [
                            f"#### [{_verdict}] `{_it['prompt_id']}` (`{_it['response_id']}` · `{_it['split']}` · `{_it['category']}` · `{_it['priority']}`)",
                            f"- **Scores:** Swedish=`{_it['swedish_quality']}/2` · Task=`{_it['task_correctness']}/2` · Failure modes: `{_fms}`",
                            f"- **Prompt:** {_it['prompt']}",
                            f"- **Model output:** {_it['response_text']}",
                            f"- **Evaluator notes:** {_it['notes']}",
                            f"- **Reference correct answer:** {_it['reference_answer']}",
                        ]
                    )
                )
            return "\n\n---\n\n".join(_cards) if _cards else "*None.*"

        _mismatch_notice = (
            f"\n\n> **Warning:** `response_id` mismatch detected for: `{', '.join(_id_mismatches)}`"
            if _id_mismatches
            else ""
        )

        _eval_md = "\n\n".join(
            [
                f"## External Evaluation Summary (`results/{eval_path.name}`)",
                f"**Evaluator:** `{_eval_data.get('evaluator', 'unknown')}` · "
                f"**Overall accuracy:** `{_passed}/{_total}` ({_acc:.1f}%) · "
                f"**Mean Swedish quality:** `{_mean_sv:.2f}/2` · "
                f"**Mean Task correctness:** `{_mean_task:.2f}/2`{_mismatch_notice}",
                _group_table("category", "Breakdown by Category"),
                _group_table("split", "Breakdown by Split (`dev` vs `eval`)"),
                _group_table("priority", "Breakdown by Priority (`target` vs `non_negotiable`)"),
                "\n".join(_fm_lines),
            ]
        )

        eval_view = mo.vstack(
            [
                mo.md(_eval_md),
                mo.accordion(
                    {
                        f"Failed Cases ({len(_failed_items)})": mo.md(_format_cards(_failed_items)),
                        f"Passed Cases ({len(_passed_items)})": mo.md(_format_cards(_passed_items)),
                    }
                ),
            ]
        )

    eval_view
    return


if __name__ == "__main__":
    app.run()
