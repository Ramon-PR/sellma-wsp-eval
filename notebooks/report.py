import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Weight-Space Programming for SE.LLMA — Report",
)


@app.cell
def _():
    from pathlib import Path
    import sys

    import marimo as mo

    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.report_builder import (
        REPORT_DIR,
        RESULTS_DIR,
        build_report_sections,
        export_reports,
        generate_baseline_diagnostic_figure,
        load_report_data,
    )

    return (
        REPORT_DIR,
        RESULTS_DIR,
        build_report_sections,
        export_reports,
        generate_baseline_diagnostic_figure,
        load_report_data,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    reload_btn = mo.ui.run_button(label="Reload results")
    include_appendix_chk = mo.ui.checkbox(
        value=True,
        label="Include detailed examples & appendix in PDF",
    )
    export_btn = mo.ui.run_button(
        label="Generate report/research_note.md & report/report.pdf",
        kind="success",
    )

    mo.vstack(
        [
            mo.hstack(
                [reload_btn, export_btn, include_appendix_chk],
                justify="start",
                align="center",
                gap=1.0,
            ),
        ]
    )
    return export_btn, include_appendix_chk, reload_btn


@app.cell(hide_code=True)
def _(
    REPORT_DIR,
    build_report_sections,
    export_btn,
    export_reports,
    generate_baseline_diagnostic_figure,
    include_appendix_chk,
    load_report_data,
    mo,
    reload_btn,
):
    _ = reload_btn.value
    report_data = load_report_data()
    baseline_fig_path = generate_baseline_diagnostic_figure(report_data)
    sections = build_report_sections(
        report_data,
        include_appendix=include_appendix_chk.value,
        image_mode="none",
    )

    _export_summary = None
    _export_err = None
    if export_btn.value:
        try:
            with mo.status.spinner(title="Generating report/research_note.md & report/report.pdf..."):
                _export_summary = export_reports(include_appendix=include_appendix_chk.value)
        except Exception as _exc:
            _export_err = f"{type(_exc).__name__}: {_exc}"

    _note_file = REPORT_DIR / "research_note.md"
    _pdf_file = REPORT_DIR / "report.pdf"
    _note_status = (
        f"Ready ({len(_note_file.read_text(encoding='utf-8').split())} words)"
        if _note_file.exists()
        else "Not yet generated"
    )
    _pdf_status = (
        f"Ready ({round(_pdf_file.stat().st_size / 1024.0, 1)} KB)"
        if _pdf_file.exists()
        else "Not yet generated"
    )

    _checks = [
        ("CPU screening (`cpu_prompt_benchmark.json`)", report_data["has_cpu_bench"]),
        ("Baseline responses & eval (`qwen_baseline_*.json`)", report_data["has_baseline"]),
        ("6-arm experiment (`experiment_details.json`)", report_data["has_experiment_details"]),
        ("Held-out arm generations (`qwen_*_responses.json`)", report_data["has_arm_generations"]),
    ]
    _check_lines = " · ".join(
        f"{'✅' if _ok else '⏳'} {_lbl}" for _lbl, _ok in _checks
    )

    if _export_err:
        _status_callout = mo.callout(
            mo.md(f"**Export failed:** `{_export_err}`"),
            kind="danger",
        )
    elif _export_summary is not None:
        _status_callout = mo.callout(
            mo.md(
                f"**Exported deliverables:** `report/research_note.md` (`{_export_summary['research_note_words']}` words) "
                f"and `report/report.pdf` (`{_export_summary['pdf_pages']}` pages, `{_export_summary['pdf_size_kb']} KB`)."
            ),
            kind="success",
        )
    else:
        _kind = "success" if report_data["has_experiment_details"] else "warn"
        _status_callout = mo.callout(
            mo.md(
                f"**Artifacts:** {_check_lines}  \n"
                f"**Deliverables on disk:** `report/research_note.md`: `{_note_status}` · "
                f"`report/report.pdf`: `{_pdf_status}`"
            ),
            kind=_kind,
        )

    _status_callout
    return baseline_fig_path, report_data, sections


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["header_and_exec"])
    return


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["sec1_critique"])
    return


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["sec2_hypothesis"])
    return


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["sec3_design"])
    return


@app.cell(hide_code=True)
def _(RESULTS_DIR, baseline_fig_path, mo, report_data, sections):
    _fig_items = []
    _fig_specs = [
        (
            "baseline_diagnostic_summary.png",
            "Baseline diagnostic accuracy by category (left) and failure modes across 32 prompts (right).",
        ),
        (
            "layer_mapping_static_vs_des.png",
            "Layer-wise alignment between Static WSP scores and Awake DES causal sensitivity.",
        ),
        (
            "arm_comparison_nll.png",
            "Held-out Swedish target NLL across all 6 arms and stop-point trade-off.",
        ),
    ]
    _ = baseline_fig_path, report_data
    for _fname, _cap in _fig_specs:
        _fpath = RESULTS_DIR / _fname
        if _fpath.exists():
            _fig_items.append(mo.image(src=str(_fpath), caption=_cap))

    if _fig_items:
        _sec4_view = mo.vstack(
            [
                mo.md(sections["sec4_results"]),
                mo.md("### 4.5 Figures"),
                *_fig_items,
            ]
        )
    else:
        _sec4_view = mo.md(sections["sec4_results"])

    _sec4_view
    return


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["sec5_interpretation"])
    return


@app.cell(hide_code=True)
def _(mo, sections):
    mo.md(sections["sec6_reproducibility_refs"])
    return


if __name__ == "__main__":
    app.run()
