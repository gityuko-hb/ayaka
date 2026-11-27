"""Tests for InferenceGrpcServer + GrpcEngineClient seam contract, HTTP serving, and CLI flags."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.server import ServerConfig
from ayaka.core.errors import InvalidRequestError, OverloadedError, UnsupportedFeatureError
from ayaka.core.outputs import CompletionOutput, RequestOutput
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.seam import EngineClient, EngineInfo
from ayaka.entrypoints.cli_parsers import build_cli_parser
from ayaka.entrypoints.grpc.client import GrpcEngineClient
from ayaka.entrypoints.grpc.inference import InferenceGrpcServer
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.settings import describe_remote_engine, describe_server
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.utils.dtypes import DType
from fake_engine import FakeEngine, script
from tokenizer_fakes import FakeTokenizer, loaded_for, make_service

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _hanging_script() -> list[RequestOutput]:
    return [
        RequestOutput(
            request_id=RequestId("unused"),
            outputs=(CompletionOutput(index=0, text="part", token_ids=(1,)),),
            finished=False,
        )
    ]


def test_grpc_engine_client_seam_contract_and_cancellation(tmp_path: Path) -> None:
    pytest.importorskip("grpc")
    pytest.importorskip("google.protobuf")

    async def exercise(target: str) -> None:
        info = EngineInfo(max_model_len=512, max_context_tokens=256, prefix_caching=True)
        fake = FakeEngine(
            info,
            [
                script("hello", " world"),
                OverloadedError("busy now"),
                InvalidRequestError("bad req"),
                UnsupportedFeatureError("nope"),
                _hanging_script(),
                _hanging_script(),
            ],
        )
        server = InferenceGrpcServer(fake, target=target)
        bound = await server.start()
        remote = await GrpcEngineClient(bound, watch_interval_seconds=0.05).connect()
        try:
            assert isinstance(remote, EngineClient)
            assert remote.info == info
            assert remote.health().state == fake.health().state

            # 1. Normal stream round-trip
            s1 = await remote.add(
                GenerateRequest(
                    request_id=RequestId("r-ok"),
                    prompt=(1, 2),
                    sampling=SamplingParams(max_tokens=8),
                )
            )
            chunks = [out async for out in s1]
            assert "".join(c.outputs[0].text for c in chunks) == "hello world"
            assert chunks[-1].finished is True

            # 2. Admission errors raise at add()
            with pytest.raises(OverloadedError, match="busy now"):
                await remote.add(
                    GenerateRequest(
                        request_id=RequestId("r-ov"), prompt=(1,), sampling=SamplingParams()
                    )
                )
            with pytest.raises(InvalidRequestError, match="bad req"):
                await remote.add(
                    GenerateRequest(
                        request_id=RequestId("r-inv"), prompt=(1,), sampling=SamplingParams()
                    )
                )
            with pytest.raises(UnsupportedFeatureError, match="nope"):
                await remote.add(
                    GenerateRequest(
                        request_id=RequestId("r-unsup"), prompt=(1,), sampling=SamplingParams()
                    )
                )

            # 3. abort() closes local stream and propagates cancellation remotely
            s_abort = await remote.add(
                GenerateRequest(
                    request_id=RequestId("r-abort"), prompt=(1,), sampling=SamplingParams()
                )
            )
            first = await anext(aiter(s_abort))
            assert first.outputs[0].text == "part"
            remote.abort(RequestId("r-abort"))
            for _ in range(30):
                if RequestId("r-abort") in fake.aborted:
                    break
                await asyncio.sleep(0.01)
            assert RequestId("r-abort") in fake.aborted

            # 4. abort_all() finishes in-flight request with CANCELLED
            s_all = await remote.add(
                GenerateRequest(
                    request_id=RequestId("r-all"), prompt=(1,), sampling=SamplingParams()
                )
            )
            _ = await anext(aiter(s_all))
            remote.abort_all()
            tail = [out async for out in s_all]
            assert tail[-1].finished is True
            assert tail[-1].outputs[0].finish_reason is FinishReason.CANCELLED
        finally:
            await remote.shutdown()
            await server.shutdown()

    asyncio.run(exercise("127.0.0.1:0"))
    if sys.platform != "win32":
        asyncio.run(exercise(f"unix://{tmp_path / 'ayaka-engine.sock'}"))


def test_http_openai_and_anthropic_serve_over_grpc_engine_client() -> None:
    pytest.importorskip("grpc")
    pytest.importorskip("google.protobuf")
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    from ayaka.entrypoints.http import create_app
    from ayaka.entrypoints.serving import Serving

    async def main() -> None:
        info = EngineInfo(max_model_len=1024, max_context_tokens=1024, prefix_caching=True)
        fake = FakeEngine(
            info,
            [
                script("<think>", "thought", "</think>", "chat ok"),
                script("responses ok"),
                script("<think>", "thought", "</think>", "anthropic ok"),
            ],
        )
        server = InferenceGrpcServer(fake, target="127.0.0.1:0")
        bound = await server.start()
        remote = await GrpcEngineClient(bound).connect()
        service = make_service()
        serving = Serving(
            ServerConfig(reasoning_parser="think", tool_call_parser="hermes"),
            model_name="tiny-remote",
        )
        serving.attach(
            remote,
            service,
            output_format=OutputFormat.resolve(reasoning="think", tool_calls="hermes"),
        )
        http_server = TestServer(create_app(serving), handler_cancellation=True)
        try:
            async with TestClient(http_server) as client:
                r1 = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert r1.status == 200
                b1 = await r1.json()
                assert b1["choices"][0]["message"]["content"] == "chat ok"
                assert b1["choices"][0]["message"]["reasoning_content"] == "thought"

                r2 = await client.post("/v1/responses", json={"input": "hi"})
                assert r2.status == 200
                b2: dict[str, Any] = await r2.json()
                assert b2["status"] == "completed"

                r3 = await client.post(
                    "/v1/messages",
                    json={
                        "model": "tiny-remote",
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert r3.status == 200
                b3: dict[str, Any] = await r3.json()
                assert any(
                    block["type"] == "text" and block["text"] == "anthropic ok"
                    for block in b3["content"]
                )
        finally:
            await remote.shutdown()
            await server.shutdown()
            service.close()

    asyncio.run(main())


def test_grpc_cli_subcommands_and_config_flags() -> None:
    cfg = ServerConfig(engine_grpc="127.0.0.1:50051")
    assert cfg.engine_grpc == "127.0.0.1:50051"

    parser = build_cli_parser()
    a1 = parser.parse_args(["serve", "m", "--engine-grpc", "unix:/tmp/ayaka.sock"])
    assert a1.command == "serve"
    assert a1.engine_grpc == "unix:/tmp/ayaka.sock"

    a2 = parser.parse_args(["engine-server", "m", "--listen", "127.0.0.1:50051"])
    assert a2.command == "engine-server"
    assert a2.listen == "127.0.0.1:50051"

    a3 = parser.parse_args(["render-server", "m", "--listen", "127.0.0.1:50052"])
    assert a3.command == "render-server"
    assert a3.listen == "127.0.0.1:50052"


def test_engine_grpc_startup_logs_remote_device_and_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("grpc")
    pytest.importorskip("google.protobuf")
    pytest.importorskip("aiohttp")
    from ayaka.entrypoints.serve import serve

    loaded = loaded_for(FakeTokenizer())
    monkeypatch.setattr(DefaultTokenizerFactory, "from_config", staticmethod(lambda _cfg: loaded))
    local_defaults = EngineConfig(
        source=ModelSourceConfig(str(tmp_path)),
        device=DeviceConfig(device="cpu", dtype=DType.FP32),
    )
    server_cfg = ServerConfig(port=0, engine_grpc="127.0.0.1:50051")
    pre_lines = describe_server(server_cfg, local_defaults, "Qwen/Qwen3-8B")
    assert "cpu" not in pre_lines[0]
    assert "fp32" not in pre_lines[0]
    assert "remote engine 127.0.0.1:50051" in pre_lines[0]

    remote_info = EngineInfo(
        max_model_len=8192,
        max_context_tokens=4096,
        prefix_caching=True,
        device="cuda",
        dtype="bf16",
        backends=("flashinfer", "torch"),
        cuda_graph=True,
    )
    remote_line = describe_remote_engine("127.0.0.1:50051", remote_info, "Qwen/Qwen3-8B")
    assert "on cuda in bf16" in remote_line
    assert "flashinfer, torch" in remote_line
    assert "CUDA graphs" in remote_line

    async def main() -> None:
        fake = FakeEngine(remote_info, [script("ok")])
        grpc_server = InferenceGrpcServer(fake, target="127.0.0.1:0")
        bound = await grpc_server.start()
        stop = asyncio.Event()
        said: list[str] = []
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def on_ready(_host: str, _port: int) -> None:
            if not ready.done():
                ready.set_result(None)

        task = asyncio.create_task(
            serve(
                ServerConfig(port=0, engine_grpc=bound),
                local_defaults,
                model_name="Qwen/Qwen3-8B",
                say=said.append,
                stop=stop,
                handle_signals=False,
                on_ready=on_ready,
            )
        )
        try:
            await asyncio.wait_for(ready, timeout=15)
            joined = "\n".join(said)
            assert "on cuda in bf16" in joined
            assert "flashinfer, torch" in joined
            assert "CUDA graphs" in joined
            assert "cpu" not in joined
            assert "fp32" not in joined
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=15)
            await grpc_server.shutdown()

    asyncio.run(main())
