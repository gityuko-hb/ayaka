"""``benchmarks/e2e/compare_llama_cpp.py`` against two fake OpenAI-compatible servers."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load() -> Any:
    sys.path.insert(0, str(_ROOT / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "compare_llama_cpp", _ROOT / "benchmarks" / "e2e" / "compare_llama_cpp.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_llama_cpp"] = module
    spec.loader.exec_module(module)
    return module


compare = _load()

#: A fake engine: answers /health, streams /v1/completions word by word, and logs when it
#: starts and stops so a test can see that two engines never overlap.
_FAKE = textwrap.dedent(
    """
    import http.server, json, signal, sys, time
    port, log, name, delay = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])

    def note(event):
        with open(log, "a") as f:
            f.write(json.dumps({"name": name, "event": event, "t": time.time()}) + "\\n")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            time.sleep(delay)  # the prompt read
            for i in range(body["max_tokens"]):
                chunk = {"choices": [{"index": 0, "text": f" w{i}"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\\n\\n".encode()); self.wfile.flush()
                time.sleep(delay / 10)
            usage = {"prompt_tokens": len(body["prompt"]) // 4,
                     "completion_tokens": body["max_tokens"]}
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\\n\\n".encode())
            self.wfile.write(b"data: [DONE]\\n\\n")

    server = http.server.HTTPServer(("127.0.0.1", port), H)
    def stop(*_):
        note("stop"); sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    note("start")
    try:
        server.serve_forever()
    finally:
        note("stop")
    """
)


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def fakes(tmp_path: Path) -> dict[str, Any]:
    script = tmp_path / "fake.py"
    script.write_text(_FAKE, encoding="utf-8")
    log = tmp_path / "events.jsonl"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    gguf = tmp_path / "model-f16.gguf"
    gguf.write_bytes(b"gguf")
    ports = {"ayaka": _free_port(), "llama.cpp": _free_port()}

    def command(name: str, delay: float) -> str:
        return f'"{sys.executable}" "{script}" {ports[name]} "{log}" {name} {delay}'

    return {"command": command, "ports": ports, "log": log, "model": model_dir, "gguf": gguf}


def _argv(fakes: dict[str, Any], out: Path, *, precision: str = "f16") -> list[str]:
    return [
        "--ayaka-cmd",
        fakes["command"]("ayaka", 0.05),
        "--ayaka-url",
        f"http://127.0.0.1:{fakes['ports']['ayaka']}",
        "--ayaka-model",
        str(fakes["model"]),
        "--llama-cmd",
        fakes["command"]("llama.cpp", 0.1),
        "--llama-url",
        f"http://127.0.0.1:{fakes['ports']['llama.cpp']}",
        "--llama-model",
        str(fakes["gguf"]),
        "--llama-precision",
        precision,
        "--lengths",
        "64",
        "--rounds",
        "2",
        "--max-tokens",
        "8",
        "--timeout",
        "30",
        "--out",
        str(out),
    ]


class TestRun:
    def test_a_run_alternates_never_overlaps_and_validates(
        self, fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")  # the machine record reads torch
        out = tmp_path / "result.json"
        # The fake model has no tokenizer: bytes stand in for tokens.
        monkeypatch.setattr(
            compare,
            "_tokenizer",
            lambda model: (lambda text: list(text.encode()), lambda ids: bytes(ids).decode()),
        )

        code = compare.main(_argv(fakes, out))

        result = json.loads(out.read_text(encoding="utf-8"))
        assert code == 0, compare.validate(result)
        assert [slot["engine"] for slot in result["slots"]] == [
            "ayaka",
            "llama.cpp",
            "llama.cpp",
            "ayaka",
        ]
        spans = [slot["up"] for slot in result["slots"]]
        assert all(end <= start for (_, end), (start, _) in itertools.pairwise(spans))
        starts = [json.loads(x) for x in fakes["log"].read_text(encoding="utf-8").splitlines()]
        assert [event["name"] for event in starts if event["event"] == "start"] == [
            "ayaka",
            "llama.cpp",
            "llama.cpp",
            "ayaka",
        ]
        assert result["opponent"]["precision"] == "f16"
        assert result["opponent"]["hash"]
        assert any("not the same file" in note for note in result["differences"])
        assert not any("precisions differ" in note for note in result["differences"])  # f16 = fp16
        assert all(len(slot["gpu_temperature_c"]) == 2 for slot in result["slots"])
        assert len(result["power"]) == 2
        ratios = result["summary"]["ratios"]["median"]["prose-64"]
        assert ratios["ttft_cold_s"] < 1  # the fake ayaka reads its prompt faster
        assert compare.validate(result) == []

    def test_an_engine_that_never_gets_healthy_times_out(self) -> None:
        with pytest.raises(compare.HealthTimeoutError):
            compare.wait_healthy(f"http://127.0.0.1:{_free_port()}", 1.0)

    def test_an_engine_that_exits_is_reported(self) -> None:
        with pytest.raises(RuntimeError, match="exited before it was healthy"):
            compare.wait_healthy("http://127.0.0.1:9", 5.0, alive=lambda: False)

    def test_an_engine_that_exits_says_why(self, tmp_path: Path) -> None:
        engine = compare.Engine(
            "ayaka",
            f"http://127.0.0.1:{_free_port()}",
            [
                sys.executable,
                "-c",
                "import sys; print('KV floor: short 54556263', file=sys.stderr)",
            ],
            "m",
            None,
            "fp16",
        )
        plan = compare.Plan(prompts=[], rounds=1, max_tokens=1, timeout=30.0, key=None)

        with pytest.raises(RuntimeError, match="short 54556263"):
            compare.run_slot(engine, plan, 0, running=[])

    def test_a_missing_model_file_stops_the_run(
        self, fakes: dict[str, Any], tmp_path: Path
    ) -> None:
        argv = _argv(fakes, tmp_path / "r.json")
        argv[argv.index("--llama-model") + 1] = str(tmp_path / "missing.gguf")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            compare.main(argv)


def _slot(engine: str, round_index: int, ttft: float, rate: float) -> dict[str, Any]:
    return {
        "engine": engine,
        "round": round_index,
        "rows": [
            {
                "prompt": "p",
                "ttft_cold_s": ttft,
                "ttft_warm_s": ttft / 2,
                "prompt_tok_s": 1000 / ttft,
                "generation_tok_s": rate,
            }
        ],
    }


class TestSummary:
    def test_ratios_pair_each_round_and_medians_use_every_run(self) -> None:
        engines = [
            compare.Engine("ayaka", "u", None, "m", None, "fp16"),
            compare.Engine("llama.cpp", "u", None, "m", None, "f16"),
        ]
        slots = [
            _slot("ayaka", 0, 0.2, 50.0),
            _slot("llama.cpp", 0, 0.4, 100.0),
            _slot("llama.cpp", 1, 0.5, 80.0),
            _slot("ayaka", 1, 0.25, 40.0),
        ]

        summary = compare.summarize(slots, engines)

        assert summary["ratios"]["of"] == "ayaka/llama.cpp"
        assert summary["ratios"]["per_round"]["p"]["generation_tok_s"] == [0.5, 0.5]
        assert summary["ratios"]["per_round"]["p"]["ttft_cold_s"] == [0.5, 0.5]
        assert summary["medians"]["ayaka"]["p"]["generation_tok_s"] == 45.0
        assert compare.round_order(1, engines) == list(reversed(engines))

    def test_a_quantized_side_is_never_merged_with_an_f16_one(self) -> None:
        result = {
            "schema": compare.SCHEMA,
            "machine": {},
            "opponent": {},
            "differences": [],
            "slots": [],
            "summary": {"medians": {}},
            "engines": [{"precision": "fp16"}, {"precision": "Q4_K_M"}],
        }

        assert compare.validate(result) == [
            "a quantized engine is compared with an unquantized one in one result"
        ]
        engines = [
            compare.Engine("ayaka", "u", None, "m", None, "fp16"),
            compare.Engine("llama.cpp", "u", None, "m", None, "q4_k_m"),
        ]
        assert any("precisions differ" in note for note in compare.differences(engines))
        assert compare.precision_name("fp16") == compare.precision_name("F16") == "f16"

    def test_one_engine_at_a_time_is_enforced(self) -> None:
        engine = compare.Engine("ayaka", "u", None, "m", None, "fp16")
        plan = compare.Plan([], 1, 1, 1.0, None)

        with pytest.raises(RuntimeError, match="never run at the same time"):
            compare.run_slot(engine, plan, 0, running=["llama.cpp"])
