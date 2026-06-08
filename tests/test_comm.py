import pytest

from llm_bench.comm import NcclConfig, _docker_cmd, parse_all_reduce_output, render_nccl_report, summarize_nccl
from llm_bench.commands.comm import DEFAULT_NCCL_IMAGE, _resolve_image


def test_parse_all_reduce_output():
    output = "1024 256 float sum -1 20.0 2.1 3.2 0 19.0 2.2 3.4 0\n"
    rows = parse_all_reduce_output(output)
    assert rows[0]["size_bytes"] == 1024
    assert rows[0]["busbw_gbps"] == 3.2


def test_summarize_includes_inplace_and_error_metrics():
    rows = parse_all_reduce_output(
        "8 2 float sum -1 22.1 0.5 1.0 0 19.4 0.8 1.6 0\n"
        "1048576 262144 float sum -1 100.0 10.0 19.0 0 95.0 11.0 20.5 0\n"
        # second column has non-zero numerical error -> flagged.
        "16777216 4194304 float sum -1 800.0 20.0 38.0 2.5e-6 780.0 21.0 39.5 5.0e-6\n"
    )
    s = summarize_nccl(rows, returncode=0)
    assert s["max_busbw_gbps"] == 38.0
    assert s["max_busbw_inplace_gbps"] == 39.5  # in-place column WAS parsed but used to be dropped
    assert s["max_algbw_gbps"] == 20.0
    assert s["max_algbw_inplace_gbps"] == 21.0
    assert s["non_zero_error_rows"] == 2          # 1 row x 2 cols both > 0
    assert s["max_error_inplace"] == pytest.approx(5e-6)


def test_format_bytes_renders_human_units():
    from llm_bench.comm import _format_bytes
    assert _format_bytes(8) == "8B"
    assert _format_bytes(65536) == "64KB"
    assert _format_bytes(268435456) == "256MB"
    assert _format_bytes(1073741824) == "1GB"
    assert _format_bytes(0) == "0B"


def test_theoretical_bandwidth_prefers_strongest_link():
    from llm_bench.comm import _theoretical_bandwidth
    # NVLink beats PCIe when both are present in the topology.
    assert _theoretical_bandwidth(["NV4", "PIX"])["gbps"] == 200.0
    assert _theoretical_bandwidth(["PIX"])["gbps"] == 32.0
    # SYS-only topology means the link crosses NUMA: no clean theoretical value.
    assert _theoretical_bandwidth(["SYS"]) is None


def test_extract_g_flag_handles_separate_and_glued_forms():
    from llm_bench.comm import _extract_g_flag
    assert _extract_g_flag(["-b", "8", "-g", "2", "-n", "100"]) == 2
    assert _extract_g_flag(["-b", "8", "-g4"]) == 4
    assert _extract_g_flag(["-b", "8"]) == 0


def test_render_nccl_report_exposes_inplace_and_topology():
    rows = parse_all_reduce_output(
        "8 2 float sum -1 22.1 0.5 1.0 0 19.4 0.8 1.6 0\n"
        "1073741824 268435456 float sum -1 83876 12.80 12.80 0 83452 12.87 12.87 0\n"
    )
    summary = summarize_nccl(rows, returncode=0)
    manifest = {
        "run_id": "test-run",
        "config": {"image": "nccl-tests:latest"},
        "container_command": ["/opt/nccl-tests/build/all_reduce_perf", "-b", "8", "-e", "1G", "-g", "2"],
        "returncode": 0,
        "plan_script": "/tmp/plan.sh",
        "summary": summary,
        "gpu_count": 2,
        "gpu_topology": {"available": True, "links": ["NV1"]},
        "stderr_log": "/tmp/missing",
    }
    md = render_nccl_report(manifest, rows, image_paths=[])
    # Key indicators include BOTH out-of-place AND in-place now.
    assert "max busbw (out-of-place)" in md
    assert "max busbw (in-place)" in md
    assert "12.87" in md  # in-place value reached the report
    # Topology + theoretical ceiling section.
    assert "## 拓扑 / 环境" in md
    assert "NV1" in md
    assert "NVLink" in md  # human-readable description
    # Largest size shown in both raw bytes and human form.
    assert "1GB" in md
    assert "1073741824" in md


def test_summarize_nccl():
    rows = parse_all_reduce_output(
        "8 2 float sum -1 12.3 0.01 0.02 0 11.1 0.02 0.03 0\n"
        "1024 256 float sum -1 20.0 2.1 3.2 0 19.0 2.2 3.4 0\n"
    )
    summary = summarize_nccl(rows, 0)
    assert summary["success"] is True
    assert summary["largest_size_busbw_gbps"] == 3.2


def test_docker_cmd_passes_user_command_unchanged():
    config = NcclConfig(
        image="nccl:test",
        command=["/opt/nccl-tests/build/all_reduce_perf", "-b", "8", "-e", "1G", "-f", "2", "-g", "8", "-n", "100", "-w", "20"],
        docker_args=["--gpus", "all", "--shm-size", "16g"],
    )
    cmd = _docker_cmd(config, "llm-bench-nccl-test")
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--name" in cmd and "llm-bench-nccl-test" in cmd
    assert "--gpus" in cmd and "all" in cmd
    assert "--shm-size" in cmd and "16g" in cmd
    image_idx = cmd.index("nccl:test")
    assert cmd[image_idx + 1:] == config.command


def test_docker_cmd_no_extra_docker_args():
    config = NcclConfig(image="nccl:test", command=["echo", "hello"])
    cmd = _docker_cmd(config, "nccl-name")
    assert cmd == ["docker", "run", "--rm", "--name", "nccl-name", "nccl:test", "echo", "hello"]


def test_resolve_image_uses_discovered_when_empty(monkeypatch):
    monkeypatch.setattr(
        "llm_bench.commands.comm.discover_docker_images",
        lambda backend: [{"name": "ghcr.io/local/nccl:1", "id": "x", "size": ""}],
    )
    assert _resolve_image("") == "ghcr.io/local/nccl:1"


def test_resolve_image_keeps_explicit(monkeypatch):
    monkeypatch.setattr("llm_bench.commands.comm.discover_docker_images", lambda backend: [])
    assert _resolve_image("custom/nccl:tag") == "custom/nccl:tag"


def test_resolve_image_falls_back_to_default(monkeypatch):
    monkeypatch.setattr("llm_bench.commands.comm.discover_docker_images", lambda backend: [])
    assert _resolve_image("") == DEFAULT_NCCL_IMAGE


def test_missing_command_raises():
    from llm_bench.comm import run_nccl_all_reduce

    with pytest.raises(ValueError, match="missing NCCL command"):
        run_nccl_all_reduce(NcclConfig(image="nccl:test"))
