from llm_bench.config import BenchConfig
from llm_bench.http_client import HttpBenchTarget, _payload, wait_for_openai_server
from llm_bench.workload import WorkloadRequest


def test_payload_uses_sampling_parameters():
    config = BenchConfig()
    config.workload.temperature = 0.2
    config.workload.top_p = 0.9
    workload = WorkloadRequest(prompt="hello", input_tokens=1, output_tokens=3)

    payload = _payload(config, HttpBenchTarget("http://127.0.0.1:8000", "model", "vllm"), workload)

    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9


def test_wait_for_server_can_stop_from_callback():
    calls = []

    ok = wait_for_openai_server("http://127.0.0.1:1", 30, on_wait=lambda elapsed: calls.append(elapsed) or False)

    assert ok is False
    assert calls


def _run_local_server(handler_cls):
    import http.server
    import threading
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


def test_smoke_ping_succeeds_on_completions_response():
    import http.server
    import json
    from llm_bench.http_client import HttpBenchTarget, smoke_ping_server

    received: dict = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received.update(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}).encode())

        def log_message(self, *_a, **_kw):
            pass

    srv, _t = _run_local_server(H)
    try:
        target = HttpBenchTarget(url=f"http://127.0.0.1:{srv.server_port}", model="m", backend="vllm")
        err = smoke_ping_server(target, api="completions", timeout_seconds=5)
        assert err is None
        # Verify the minimal payload was sent (max_tokens=1, non-stream).
        assert received["max_tokens"] == 1
        assert received["stream"] is False
        assert received["prompt"] == "ping"
        assert received["model"] == "m"
    finally:
        srv.shutdown()
        srv.server_close()


def test_smoke_ping_uses_chat_endpoint():
    import http.server
    import json
    from llm_bench.http_client import HttpBenchTarget, smoke_ping_server

    captured_path: dict = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            captured_path["path"] = self.path
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "x"}}]}).encode())

        def log_message(self, *_a, **_kw):
            pass

    srv, _t = _run_local_server(H)
    try:
        target = HttpBenchTarget(url=f"http://127.0.0.1:{srv.server_port}", model="m", backend="vllm")
        assert smoke_ping_server(target, api="chat", timeout_seconds=5) is None
        assert captured_path["path"] == "/v1/chat/completions"
    finally:
        srv.shutdown()
        srv.server_close()


def test_smoke_ping_reports_http_error():
    import http.server
    from llm_bench.http_client import HttpBenchTarget, smoke_ping_server

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"model not found")

        def log_message(self, *_a, **_kw):
            pass

    srv, _t = _run_local_server(H)
    try:
        target = HttpBenchTarget(url=f"http://127.0.0.1:{srv.server_port}", model="missing", backend="vllm")
        err = smoke_ping_server(target, api="completions", timeout_seconds=5)
        assert err and "404" in err
    finally:
        srv.shutdown()
        srv.server_close()


def test_smoke_ping_returns_error_when_no_choices():
    import http.server
    import json
    from llm_bench.http_client import HttpBenchTarget, smoke_ping_server

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"unexpected": "body"}).encode())

        def log_message(self, *_a, **_kw):
            pass

    srv, _t = _run_local_server(H)
    try:
        target = HttpBenchTarget(url=f"http://127.0.0.1:{srv.server_port}", model="m", backend="vllm")
        err = smoke_ping_server(target, api="completions", timeout_seconds=5)
        assert err and "unexpected body" in err
    finally:
        srv.shutdown()
        srv.server_close()


def test_smoke_ping_returns_error_when_server_unreachable():
    from llm_bench.http_client import HttpBenchTarget, smoke_ping_server

    target = HttpBenchTarget(url="http://127.0.0.1:1", model="m", backend="vllm")
    err = smoke_ping_server(target, api="completions", timeout_seconds=2)
    assert err is not None
