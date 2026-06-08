"""Chart rendering via matplotlib.

Replaces the previous hand-rolled PNG writer + 5x7 bitmap font (clean but
illegible when shrunk into Markdown viewers). matplotlib uses real TrueType
fonts with antialiasing and handles tick spacing / label rotation for us.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    import matplotlib

    matplotlib.use("Agg")  # Headless: no GUI, write PNGs directly.
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL_AVAILABLE = True
    _MPL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    # matplotlib is a hard dependency in pyproject.toml but we tolerate its
    # absence in stripped environments (CI without GUI deps, minimal Docker
    # images, etc.). The report keeps generating; only the PNGs are skipped.
    plt = None  # type: ignore[assignment]
    mticker = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False
    _MPL_IMPORT_ERROR = exc
    _ALREADY_WARNED = False


def _warn_once_no_matplotlib() -> None:
    global _ALREADY_WARNED
    if _MPL_AVAILABLE:
        return
    try:
        already = _ALREADY_WARNED  # noqa: F821
    except NameError:
        already = False
    if already:
        return
    print(
        "[charts] matplotlib not installed; skipping PNG generation. "
        "Install with `pip install matplotlib` to render report images. "
        f"(import error: {_MPL_IMPORT_ERROR})",
        flush=True,
    )
    globals()["_ALREADY_WARNED"] = True


# A muted, web-friendly palette. Kept in sync with the previous hand-rolled
# charts so historical run_id reports look consistent.
THROUGHPUT_COLOR = (49 / 255, 120 / 255, 196 / 255)
TTFT_COLOR = (56 / 255, 142 / 255, 60 / 255)
E2E_COLOR = (211 / 255, 84 / 255, 0 / 255)
QPS_COLOR = (255 / 255, 152 / 255, 0 / 255)
TPOT_COLOR = (156 / 255, 39 / 255, 176 / 255)
P50_COLOR = (66 / 255, 165 / 255, 245 / 255)
P90_COLOR = (255 / 255, 183 / 255, 77 / 255)
P99_COLOR = (239 / 255, 83 / 255, 80 / 255)
TREND_COLOR = (0 / 255, 150 / 255, 136 / 255)
GPU_UTIL_COLOR = (93 / 255, 64 / 255, 160 / 255)
GPU_MEM_COLOR = (0 / 255, 121 / 255, 107 / 255)
NCCL_BUS_COLOR = (25 / 255, 118 / 255, 210 / 255)
NCCL_ALG_COLOR = (123 / 255, 31 / 255, 162 / 255)
COMPARE_UP_COLOR = (46 / 255, 125 / 255, 50 / 255)
COMPARE_DOWN_COLOR = (198 / 255, 40 / 255, 40 / 255)

# 16:9 inches at 120 DPI -> 1920x1080 px PNG. Readable on most displays even
# when Markdown viewers shrink-to-fit page width.
FIGSIZE = (16, 9)
DPI = 120


def render_summary_charts(
    summary: dict[str, object],
    output_dir: Path,
    gpu_metrics: list[dict[str, object]] | None = None,
) -> list[Path]:
    if not _MPL_AVAILABLE:
        _warn_once_no_matplotlib()
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for item in summary.get("backend_results", []) or []:
        workload = item["workload"]
        metrics = item["metrics"]
        label = f"i{workload['input_tokens']}/o{workload['output_tokens']}/c{workload['concurrency']}"
        rows.append(
            {
                "label": label,
                "output_tokens_per_sec": float(metrics["output_tokens_per_sec"]),
                "qps": float(metrics.get("qps", 0)),
                "tpot_p99_ms": float(metrics.get("tpot_p99_ms", 0)),
                "e2e_p50_ms": float(metrics.get("e2e_p50_ms", 0)),
                "e2e_p90_ms": float(metrics.get("e2e_p90_ms", 0)),
                "e2e_p99_ms": float(metrics["e2e_p99_ms"]),
                "ttft_p99_ms": float(metrics["ttft_p99_ms"]),
                "concurrency": int(workload["concurrency"]),
                "input_tokens": int(workload["input_tokens"]),
                "output_tokens_wl": int(workload["output_tokens"]),
            }
        )

    chart_paths: list[Path] = []
    if not rows:
        return _render_gpu_charts(output_dir, gpu_metrics or [])

    labels = [r["label"] for r in rows]

    chart_paths.append(_bar(
        output_dir / "throughput_output_tokens_per_sec.png",
        labels,
        [r["output_tokens_per_sec"] for r in rows],
        color=THROUGHPUT_COLOR,
        title="Throughput (output tokens/s)",
        y_label="tokens/s",
        x_label="workload (i=input / o=output / c=concurrency)",
    ))
    chart_paths.append(_grouped_bar(
        output_dir / "latency_p99_ms.png",
        labels,
        [[r["ttft_p99_ms"] for r in rows], [r["e2e_p99_ms"] for r in rows]],
        series_labels=["TTFT p99", "E2E p99"],
        colors=[TTFT_COLOR, E2E_COLOR],
        title="Latency p99 (ms)",
        y_label="ms",
        x_label="workload",
    ))
    chart_paths.append(_bar(
        output_dir / "qps.png",
        labels,
        [r["qps"] for r in rows],
        color=QPS_COLOR,
        title="QPS",
        y_label="req/s",
        x_label="workload",
    ))
    chart_paths.append(_bar(
        output_dir / "tpot_p99_ms.png",
        labels,
        [r["tpot_p99_ms"] for r in rows],
        color=TPOT_COLOR,
        title="TPOT p99 (ms)",
        y_label="ms",
        x_label="workload",
    ))
    chart_paths.append(_grouped_bar(
        output_dir / "latency_percentiles_ms.png",
        labels,
        [
            [r["e2e_p50_ms"] for r in rows],
            [r["e2e_p90_ms"] for r in rows],
            [r["e2e_p99_ms"] for r in rows],
        ],
        series_labels=["p50", "p90", "p99"],
        colors=[P50_COLOR, P90_COLOR, P99_COLOR],
        title="E2E latency percentiles (ms)",
        y_label="ms",
        x_label="workload",
    ))

    # Two complementary "facet" views to replace the unreadable 16-bar chart:
    # - Per-input: x=concurrency, series=output_tokens. Answers "how does
    #   each output length scale as I add concurrent users?"
    # - Per-output: x=input_tokens, series=concurrency. Answers "for a fixed
    #   answer length, how does input length affect throughput / latency at
    #   different concurrency levels?"
    chart_paths.extend(_per_input_charts(rows, output_dir))
    chart_paths.extend(_per_output_charts(rows, output_dir))

    concurrency_values = sorted({r["concurrency"] for r in rows})
    if len(concurrency_values) > 1:
        system_tps_values = []
        per_req_tps_values = []
        for c in concurrency_values:
            c_rows = [r for r in rows if r["concurrency"] == c]
            if c_rows:
                # System throughput across all workloads at this concurrency.
                system_tps_values.append(
                    sum(r["output_tokens_per_sec"] for r in c_rows) / len(c_rows)
                )
                # Per-request "worst-case decode speed" at this concurrency:
                # take the slowest workload's TPOT p99 (= largest p99) and
                # invert it. Averaging across workloads with different (i, o)
                # is meaningless because their TPOT distributions don't share
                # a common scale, so we surface the worst single-user
                # experience instead — that's what the user actually notices.
                worst_tpot_p99 = max(r["tpot_p99_ms"] for r in c_rows if r["tpot_p99_ms"] > 0)
                per_req_tps_values.append(1000.0 / worst_tpot_p99 if worst_tpot_p99 > 0 else 0.0)
            else:
                system_tps_values.append(0.0)
                per_req_tps_values.append(0.0)

        # System total Output TPS vs concurrency (the headline metric).
        chart_paths.append(_line(
            output_dir / "concurrency_trend.png",
            [str(c) for c in concurrency_values],
            system_tps_values,
            color=TREND_COLOR,
            title="System Output TPS vs concurrency",
            y_label="tokens/s",
            x_label="concurrency",
        ))
        # Two-line comparison: system throughput climbs, worst per-request
        # decode slows. Shows the "more users = each one waits more" tradeoff.
        chart_paths.append(_two_line(
            output_dir / "concurrency_dual.png",
            [str(c) for c in concurrency_values],
            system_tps_values,
            per_req_tps_values,
            colors=(TREND_COLOR, P99_COLOR),
            series_labels=("System total Output TPS", "Worst per-req Decode TPS (1000/max TPOT p99)"),
            title="System throughput vs worst per-request decode speed",
            y_label="tokens/s",
            x_label="concurrency",
        ))

    chart_paths.extend(_render_gpu_charts(output_dir, gpu_metrics or []))
    return chart_paths


def render_compare_chart(comparison: dict[str, object], output_dir: Path) -> Path | None:
    if not _MPL_AVAILABLE:
        _warn_once_no_matplotlib()
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float]] = []
    for item in comparison.get("workload_deltas", []) or []:
        workload = item.get("workload") or {}
        label = f"i{workload.get('input_tokens')}/o{workload.get('output_tokens')}/c{workload.get('concurrency')}"
        for metric in item.get("metrics", []) or []:
            if metric.get("metric") in {"output_tokens_per_sec", "e2e_p99_ms"} and metric.get("delta_pct") is not None:
                rows.append((f"{label}:{metric['metric']}", float(metric["delta_pct"])))
    if not rows:
        return None
    path = output_dir / "compare_delta_pct.png"
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    colors = [COMPARE_UP_COLOR if v >= 0 else COMPARE_DOWN_COLOR for v in values]

    fig, ax = _new_fig()
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#404040", linewidth=1.0)
    ax.set_title("Candidate vs baseline (delta %)")
    ax.set_ylabel("%")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:+.0f}%"))
    _annotate_bars(ax, bars, lambda v: f"{v:+.1f}%")
    _finalize(fig, ax, path, labels)
    return path


def render_nccl_charts(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    if not _MPL_AVAILABLE:
        _warn_once_no_matplotlib()
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    busbw = [_to_float(row.get("busbw_gbps")) for row in rows]
    busbw_ip = [_to_float(row.get("busbw_gbps_out")) for row in rows]
    algbw = [_to_float(row.get("algbw_gbps")) for row in rows]
    algbw_ip = [_to_float(row.get("algbw_gbps_out")) for row in rows]
    sizes = [_format_bytes(row.get("size_bytes")) for row in rows]
    paths: list[Path] = []
    if any(v > 0 for v in busbw):
        paths.append(_line(
            output_dir / "nccl_busbw_gbps.png",
            sizes, busbw, color=NCCL_BUS_COLOR,
            title="NCCL busbw out-of-place (GB/s)", y_label="GB/s", x_label="message size",
        ))
    if any(v > 0 for v in algbw):
        paths.append(_line(
            output_dir / "nccl_algbw_gbps.png",
            sizes, algbw, color=NCCL_ALG_COLOR,
            title="NCCL algbw out-of-place (GB/s)", y_label="GB/s", x_label="message size",
        ))
    # In-place vs out-of-place on the same axes lets the user spot whether
    # one mode dramatically outperforms the other (large gap = NCCL pass
    # selection issue or memory-copy overhead).
    if any(v > 0 for v in busbw) and any(v > 0 for v in busbw_ip):
        paths.append(_multi_line(
            output_dir / "nccl_busbw_cmp.png",
            sizes,
            [("out-of-place", busbw), ("in-place", busbw_ip)],
            title="NCCL busbw: out-of-place vs in-place (GB/s)",
            y_label="GB/s",
            x_label="message size",
        ))
    if any(v > 0 for v in algbw) and any(v > 0 for v in algbw_ip):
        paths.append(_multi_line(
            output_dir / "nccl_algbw_cmp.png",
            sizes,
            [("out-of-place", algbw), ("in-place", algbw_ip)],
            title="NCCL algbw: out-of-place vs in-place (GB/s)",
            y_label="GB/s",
            x_label="message size",
        ))
    return paths


def _per_input_charts(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    """For each input_tokens value, emit charts of Output TPS / Decode TPS /
    TTFT p99 / E2E p99 vs concurrency, with one line per output_tokens.

    Returns the list of generated PNG paths. Skips the per-input variant when
    only a single concurrency was tested (then it would just be a dot).
    """
    if not _MPL_AVAILABLE:
        return []

    paths: list[Path] = []
    inputs = sorted({int(r["input_tokens"]) for r in rows})
    concurrencies = sorted({int(r["concurrency"]) for r in rows})
    if len(concurrencies) < 2:
        return []

    # (chart suffix, title, y label, row key, lower-is-better)
    plots = [
        ("output_tps", "Output TPS vs concurrency", "tokens/s", "output_tokens_per_sec", False),
        ("decode_tps_p50", "Decode TPS (p50) vs concurrency", "tokens/s", "_decode_tps_p50", False),
        ("ttft_p99", "TTFT p99 vs concurrency", "ms", "ttft_p99_ms", True),
        ("e2e_p99", "E2E p99 vs concurrency", "ms", "e2e_p99_ms", True),
    ]
    for in_tok in inputs:
        in_rows = [r for r in rows if int(r["input_tokens"]) == in_tok]
        outputs = sorted({int(r["output_tokens_wl"]) for r in in_rows})
        if not outputs:
            continue
        for suffix, title_base, ylabel, key, _lower_better in plots:
            # Build series: {output_tokens: [value@c1, value@c4, ...]}
            series: list[tuple[str, list[float]]] = []
            for out in outputs:
                out_rows = [r for r in in_rows if int(r["output_tokens_wl"]) == out]
                ys: list[float] = []
                for c in concurrencies:
                    matches = [r for r in out_rows if int(r["concurrency"]) == c]
                    if not matches:
                        ys.append(0.0)
                        continue
                    m = matches[0]
                    if key == "_decode_tps_p50":
                        tpot = float(m.get("tpot_p99_ms", 0)) or 0
                        ys.append(round(1000.0 / tpot, 2) if tpot > 0 else 0.0)
                    else:
                        ys.append(float(m.get(key, 0)))
                series.append((f"o={out}", ys))
            path = output_dir / f"per_input_{in_tok}_{suffix}.png"
            _multi_line(
                path,
                [str(c) for c in concurrencies],
                series,
                title=f"{title_base} (input={in_tok})",
                y_label=ylabel,
                x_label="concurrency",
            )
            paths.append(path)
    return paths


def _per_output_charts(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    """For each output_tokens value, emit charts of Output TPS / Decode TPS /
    TTFT p99 / E2E p99 vs input_tokens, with one line per concurrency.

    Skips the view when only a single input_tokens was tested (then it would
    just be a dot per series).
    """
    if not _MPL_AVAILABLE:
        return []

    paths: list[Path] = []
    outputs = sorted({int(r["output_tokens_wl"]) for r in rows})
    inputs = sorted({int(r["input_tokens"]) for r in rows})
    concurrencies = sorted({int(r["concurrency"]) for r in rows})
    if len(inputs) < 2:
        return []

    plots = [
        ("output_tps", "Output TPS vs input length", "tokens/s", "output_tokens_per_sec"),
        ("decode_tps_p50", "Decode TPS (p50) vs input length", "tokens/s", "_decode_tps_p50"),
        ("ttft_p99", "TTFT p99 vs input length", "ms", "ttft_p99_ms"),
        ("e2e_p99", "E2E p99 vs input length", "ms", "e2e_p99_ms"),
    ]
    for out_tok in outputs:
        out_rows = [r for r in rows if int(r["output_tokens_wl"]) == out_tok]
        for suffix, title_base, ylabel, key in plots:
            # Build series: {concurrency: [value@i512, value@i2048, ...]}
            series: list[tuple[str, list[float]]] = []
            for c in concurrencies:
                c_rows = [r for r in out_rows if int(r["concurrency"]) == c]
                ys: list[float] = []
                for i in inputs:
                    matches = [r for r in c_rows if int(r["input_tokens"]) == i]
                    if not matches:
                        ys.append(0.0)
                        continue
                    m = matches[0]
                    if key == "_decode_tps_p50":
                        tpot = float(m.get("tpot_p99_ms", 0)) or 0
                        ys.append(round(1000.0 / tpot, 2) if tpot > 0 else 0.0)
                    else:
                        ys.append(float(m.get(key, 0)))
                series.append((f"c={c}", ys))
            path = output_dir / f"per_output_{out_tok}_{suffix}.png"
            _multi_line(
                path,
                [str(i) for i in inputs],
                series,
                title=f"{title_base} (output={out_tok})",
                y_label=ylabel,
                x_label="input tokens",
            )
            paths.append(path)
    return paths


def _multi_line(
    path: Path,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    title: str,
    y_label: str,
    x_label: str = "",
) -> Path:
    """Multiple lines sharing one y-axis. Use for "metric vs concurrency,
    one line per output_tokens" style plots.
    """
    fig, ax = _new_fig()
    palette = [THROUGHPUT_COLOR, TREND_COLOR, E2E_COLOR, P50_COLOR, P99_COLOR, GPU_UTIL_COLOR, NCCL_BUS_COLOR]
    for idx, (label, values) in enumerate(series):
        color = palette[idx % len(palette)]
        ax.plot(labels, values, color=color, marker="o", linewidth=2.5, markersize=8, label=label)
        for x, y in zip(labels, values):
            ax.annotate(_format_number(y), (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=color)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    ax.legend(loc="best", frameon=False)
    _finalize(fig, ax, path, labels)
    return path


def _render_gpu_charts(output_dir: Path, gpu_metrics: list[dict[str, object]]) -> list[Path]:
    if not gpu_metrics:
        return []
    util_values = [_to_float(row.get("utilization", row.get("utilization.gpu"))) for row in gpu_metrics]
    mem_values = [_to_float(row.get("memory_used_mb", row.get("memory.used"))) for row in gpu_metrics]
    paths: list[Path] = []
    if any(v > 0 for v in util_values):
        paths.append(_line(
            output_dir / "gpu_utilization.png",
            [str(i) for i in range(len(util_values))],
            util_values,
            color=GPU_UTIL_COLOR,
            title="GPU utilization (%)", y_label="%", x_label="sample",
        ))
    if any(v > 0 for v in mem_values):
        paths.append(_line(
            output_dir / "gpu_memory.png",
            [str(i) for i in range(len(mem_values))],
            mem_values,
            color=GPU_MEM_COLOR,
            title="GPU memory used (MiB)", y_label="MiB", x_label="sample",
        ))
    return paths


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _bar(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    color,
    title: str,
    y_label: str,
    x_label: str = "",
) -> Path:
    fig, ax = _new_fig()
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    _annotate_bars(ax, bars, _format_number)
    _finalize(fig, ax, path, labels)
    return path


def _grouped_bar(
    path: Path,
    labels: Sequence[str],
    series: Sequence[Sequence[float]],
    *,
    series_labels: Sequence[str],
    colors,
    title: str,
    y_label: str,
    x_label: str = "",
) -> Path:
    fig, ax = _new_fig()
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    import numpy as _np
    x = _np.arange(n_groups)
    for i, (vals, slabel, color) in enumerate(zip(series, series_labels, colors)):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=slabel, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    ax.legend(loc="upper left", frameon=False)
    _finalize(fig, ax, path, labels)
    return path


def _line(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    color,
    title: str,
    y_label: str,
    x_label: str = "",
) -> Path:
    fig, ax = _new_fig()
    ax.plot(labels, values, color=color, marker="o", linewidth=2.5, markersize=8)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    # Annotate every point so users do not have to eyeball the y-axis.
    for x, y in zip(labels, values):
        ax.annotate(_format_number(y), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color="#202020")
    _finalize(fig, ax, path, labels)
    return path


def _two_line(
    path: Path,
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    colors,
    series_labels,
    title: str,
    y_label: str,
    x_label: str = "",
) -> Path:
    """Two lines on twin y-axes: system throughput (left) vs per-req speed (right).

    The two metrics typically move in opposite directions as concurrency grows
    so plotting them together on shared y-axis would squash one. Twin axes let
    each series use its own scale while sharing the x-axis.
    """
    fig, ax = _new_fig()
    color_left, color_right = colors
    label_left, label_right = series_labels
    ax.plot(labels, left_values, color=color_left, marker="o", linewidth=2.5,
            markersize=8, label=label_left)
    ax.set_xlabel(x_label or "")
    ax.set_ylabel(f"{label_left} ({y_label})", color=color_left)
    ax.tick_params(axis="y", labelcolor=color_left)
    ax.set_title(title)
    for x, y in zip(labels, left_values):
        ax.annotate(_format_number(y), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color=color_left)
    ax2 = ax.twinx()
    ax2.plot(labels, right_values, color=color_right, marker="s", linewidth=2.5,
             markersize=8, linestyle="--", label=label_right)
    ax2.set_ylabel(f"{label_right} ({y_label})", color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)
    for x, y in zip(labels, right_values):
        ax2.annotate(_format_number(y), (x, y), textcoords="offset points", xytext=(0, -16),
                     ha="center", fontsize=10, color=color_right)
    # Combined legend in the upper-left corner.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    _finalize(fig, ax, path, labels)
    return path


def _new_fig():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.grid(axis="y", color="#E1E6EB", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#404040")
    return fig, ax


def _annotate_bars(ax, bars, formatter):
    """Draw the value just above each bar (or below for negative bars)."""
    for bar in bars:
        height = bar.get_height()
        offset = 6 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            formatter(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va=va, fontsize=11, color="#202020",
        )


def _finalize(fig, ax, path: Path, labels: Sequence[str]) -> None:
    # Rotate long labels so they don't overlap.
    if labels and max(len(s) for s in labels) > 8:
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_v >= 10_000:
        return f"{value / 1_000:.1f}K"
    if abs_v >= 100:
        return f"{value:.0f}"
    if abs_v >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_bytes(value: object) -> str:
    """Render NCCL message sizes like '64KB', '256MB' instead of raw byte counts."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n //= 1024
        idx += 1
    return f"{n}{units[idx]}"


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
