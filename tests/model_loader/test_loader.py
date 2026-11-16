"""Loading a model from a safetensors checkpoint the test writes itself, and every way it can fail.

The checkpoint is a tiny GPT-2 in Hugging Face naming and layout (``Conv1D`` weights, optional
``transformer.`` prefix, mask buffers, a tied ``lm_head``), written by ``write_checkpoint``
without the ``safetensors`` package. The model it loads must equal the model that was filled
from the same state in memory: same parameters bit for bit, same hidden states through a paged
KV cache. Failures must say everything that is wrong at once and must not read a tensor first.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.configs.model_source import ModelSourceConfig, RequestFormat
from ayaka.core.errors import CapabilityError, ModelLoadError
from ayaka.utils import torch_utils
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.model_loader.loader import load_model, load_weights  # noqa: E402
from ayaka.models.gpt2 import GPT2LMHeadModel  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.weights.safetensors import SafetensorsFile  # noqa: E402
from gpt2_support import (  # noqa: E402  # needs the importorskip above
    hf_config_dict,
    hf_state,
    load_hf_state,
    run_paged,
    tensor_data,
    tiny_config,
    write_checkpoint,
)
from safetensors_files import write_safetensors  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

CPU_FP32 = DeviceConfig()


def _source(directory: Path, **fields: Any) -> ModelSourceConfig:
    return ModelSourceConfig(str(directory), **fields)


def _in_memory(state: dict[str, Any], dtype: DType = DType.FP32, **overrides: object) -> Any:
    config = tiny_config(**overrides)
    ops = load_backend(["torch"], device="cpu", dtype=dtype)
    model = GPT2LMHeadModel(config, ops=ops, dtype=dtype)
    load_hf_state(model, state)
    return model


def _tokens(length: int, seed: int) -> Any:
    return torch.randint(0, 50, (length,), generator=torch.Generator().manual_seed(seed))


def _same_parameters(a: Any, b: Any) -> None:
    left, right = dict(a.named_parameters()), dict(b.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


class TestLoadsTheSameModel:
    @pytest.mark.parametrize("prefix", ["transformer.", ""])
    @pytest.mark.parametrize("extras", [False, True], ids=["plain", "head-and-masks"])
    def test_parameters_and_outputs_equal_the_in_memory_model(
        self, tmp_path: Path, prefix: str, extras: bool
    ) -> None:
        config = tiny_config()
        state = hf_state(config, seed=2, prefix=prefix, with_masks=extras, with_head=extras)
        write_checkpoint(tmp_path, state, config)
        expected = _in_memory(state)

        model = load_model(_source(tmp_path), CPU_FP32)

        _same_parameters(model, expected)
        sequences = [_tokens(14, 1), _tokens(9, 2)]
        for got, want in zip(
            run_paged(model, sequences, prompt_lengths=[8, 9], chunk=4),
            run_paged(expected, sequences, prompt_lengths=[8, 9], chunk=4),
            strict=True,
        ):
            assert torch.equal(got, want)

    def test_the_result_is_a_ready_model(self, tmp_path: Path) -> None:
        config = tiny_config()
        state = hf_state(config, seed=3)
        write_checkpoint(tmp_path, state, config)

        model = load_model(_source(tmp_path), CPU_FP32)

        assert isinstance(model, GPT2LMHeadModel)
        assert model.config == config
        assert model.training is False
        assert {p.device.type for p in model.parameters()} == {"cpu"}
        assert not any(p.requires_grad for p in model.parameters())
        assert model.lm_head.weight is model.wte.weight  # tied again after materialising
        assert torch.equal(model.lm_head.weight, state["transformer.wte.weight"])

    def test_an_untied_head_is_loaded_from_its_own_tensor(self, tmp_path: Path) -> None:
        config = tiny_config(tie_word_embeddings=False)
        state = hf_state(config, seed=4, with_head=True)
        state["lm_head.weight"] = torch.randn(50, 32)
        write_checkpoint(tmp_path, state, config)

        model = load_model(_source(tmp_path), CPU_FP32)

        assert isinstance(model, GPT2LMHeadModel)
        assert model.lm_head.weight is not model.wte.weight
        assert torch.equal(model.lm_head.weight, state["lm_head.weight"])
        _same_parameters(model, _in_memory(state, tie_word_embeddings=False))

    def test_a_sharded_checkpoint_with_an_index(self, tmp_path: Path) -> None:
        config = tiny_config()
        state = hf_state(config, seed=5)
        write_checkpoint(tmp_path, state, config, shards=3)

        model = load_model(_source(tmp_path), CPU_FP32)

        _same_parameters(model, _in_memory(state))

    def test_the_model_is_built_with_the_ops_it_is_given(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=6), config)
        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

        model = load_model(_source(tmp_path), CPU_FP32, ops=ops)

        assert model.ops is ops

    def test_without_ops_the_device_config_chooses_them(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=6), config)

        model = load_model(_source(tmp_path), CPU_FP32)

        assert set(model.ops.providers.values()) == {"torch"}

    def test_load_weights_fills_a_model_that_already_exists(self, tmp_path: Path) -> None:
        config = tiny_config()
        state = hf_state(config, seed=7)
        write_checkpoint(tmp_path, state, config)
        other = _in_memory(hf_state(config, seed=99))

        load_weights(other, tmp_path)

        _same_parameters(other, _in_memory(state))


class TestConvertsTheDtype:
    @pytest.mark.parametrize(
        ("stored", "compute", "torch_compute"),
        [
            (torch.float16, DType.FP32, torch.float32),
            (torch.bfloat16, DType.FP32, torch.float32),
            (torch.float32, DType.BF16, torch.bfloat16),
            (torch.float32, DType.FP16, torch.float16),
            (torch.float64, DType.FP32, torch.float32),
            (torch.bfloat16, DType.BF16, torch.bfloat16),
        ],
        ids=["f16-to-f32", "bf16-to-f32", "f32-to-bf16", "f32-to-f16", "f64-to-f32", "bf16-same"],
    )
    def test_the_copy_converts_to_the_compute_dtype(
        self, tmp_path: Path, stored: Any, compute: DType, torch_compute: Any
    ) -> None:
        config = tiny_config()
        state = hf_state(config, seed=8)
        write_checkpoint(tmp_path, state, config, dtype=stored)
        # What a correct loader holds: the stored values, then the compute dtype.
        expected = _in_memory(
            {n: t.to(stored).to(torch_compute) for n, t in state.items()}, compute
        )

        model = load_model(_source(tmp_path), DeviceConfig(dtype=compute))

        assert {p.dtype for p in model.parameters()} == {torch_compute}
        _same_parameters(model, expected)

    def test_an_integer_tensor_is_refused_with_every_such_tensor_listed(
        self, tmp_path: Path
    ) -> None:
        config = tiny_config()
        state = hf_state(config, seed=9)
        write_checkpoint(tmp_path, state, config)
        tensors = {n: tensor_data(t) for n, t in state.items()}
        tensors["transformer.wpe.weight"] = tensor_data(state["transformer.wpe.weight"], torch.int8)
        tensors["transformer.ln_f.bias"] = tensor_data(state["transformer.ln_f.bias"], torch.int8)
        write_safetensors(tmp_path / "model.safetensors", tensors)

        with pytest.raises(ModelLoadError, match=r"2 tensors in a dtype") as error:
            load_model(_source(tmp_path), CPU_FP32)

        assert error.value.context["unsupported_dtype"] == {
            "wpe.weight": "int8",
            "ln_f.bias": "int8",
        }


class TestTheCheckpointIsCheckedBeforeAnythingIsRead:
    @pytest.fixture
    def no_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(self: SafetensorsFile, name: str) -> Any:
            raise AssertionError(f"read {name} of a checkpoint that does not match")

        monkeypatch.setattr(SafetensorsFile, "read", refuse)

    @pytest.mark.usefixtures("no_reads")
    def test_missing_extra_and_wrong_shape_are_all_listed_at_once(self, tmp_path: Path) -> None:
        config = tiny_config()
        state = hf_state(config, seed=10)
        del state["transformer.h.1.mlp.c_fc.weight"]  # missing
        del state["transformer.ln_f.bias"]  # missing
        state["transformer.h.0.attn.rotary_emb.inv_freq"] = torch.zeros(4)  # unexpected
        state["score.weight"] = torch.zeros(2, 32)  # unexpected
        state["transformer.wpe.weight"] = torch.zeros(63, 32)  # wrong shape
        state["transformer.h.0.attn.c_attn.weight"] = torch.zeros(96, 32)  # untransposed
        write_checkpoint(tmp_path, state, config)

        with pytest.raises(ModelLoadError) as error:
            load_model(_source(tmp_path), CPU_FP32)

        context = error.value.context
        assert context["missing"] == ["layers.1.mlp.up_proj.weight", "ln_f.bias"]
        assert context["unexpected"] == [
            "score.weight",
            "transformer.h.0.attn.rotary_emb.inv_freq",
        ]
        assert context["wrong_shape"] == {
            "wpe.weight": ((64, 32), (63, 32)),
            "layers.0.attn.qkv_proj.weight": ((32, 96), (96, 32)),
        }
        message = str(error.value)
        for fragment in (
            "missing 2 parameters",
            "ln_f.bias",
            "unexpected 2 tensors",
            "score.weight",
            "2 tensors with the wrong shape",
            "wpe.weight is [63, 32], expected [64, 32]",
        ):
            assert fragment in message, fragment

    @pytest.mark.usefixtures("no_reads")
    def test_a_parameter_claimed_by_two_tensors_is_reported(self, tmp_path: Path) -> None:
        config = tiny_config()
        state = hf_state(config, seed=11)
        state["wte.weight"] = state["transformer.wte.weight"].clone()  # both spellings present
        write_checkpoint(tmp_path, state, config)

        with pytest.raises(ModelLoadError, match="claimed by several tensors") as error:
            load_model(_source(tmp_path), CPU_FP32)

        assert error.value.context["ambiguous"] == {
            "wte.weight": ["transformer.wte.weight", "wte.weight"]
        }

    def test_nothing_is_allocated_on_the_device_for_a_wrong_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config()
        state = hf_state(config, seed=12)
        del state["transformer.ln_f.weight"]
        write_checkpoint(tmp_path, state, config)

        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("memory was allocated before the checkpoint was checked")

        monkeypatch.setattr(torch.nn.Module, "to_empty", refuse)

        with pytest.raises(ModelLoadError, match=re.escape("ln_f.weight")):
            load_model(_source(tmp_path), CPU_FP32)


class TestCorruptCheckpoints:
    def test_a_truncated_file_is_a_model_load_error_naming_it(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=13), config)
        file = tmp_path / "model.safetensors"
        file.write_bytes(file.read_bytes()[:-100])

        with pytest.raises(ModelLoadError, match=r"model\.safetensors.*(ends at|truncated)"):
            load_model(_source(tmp_path), CPU_FP32)

    def test_a_header_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=13), config)
        (tmp_path / "model.safetensors").write_bytes((8).to_bytes(8, "little") + b"not json")

        with pytest.raises(ModelLoadError, match="not valid JSON"):
            load_model(_source(tmp_path), CPU_FP32)

    def test_an_index_that_names_a_missing_shard_is_refused(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=13), config, shards=2)
        (tmp_path / "model-00002-of-00002.safetensors").unlink()

        with pytest.raises(ModelLoadError, match="names shards that are not in"):
            load_model(_source(tmp_path), CPU_FP32)

    def test_a_directory_without_tensors_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps(hf_config_dict(tiny_config())), "utf-8")

        with pytest.raises(ModelLoadError, match="no tensors found"):
            load_model(_source(tmp_path), CPU_FP32)

    def test_a_file_that_changes_after_its_header_was_read_is_caught_when_it_is_read(
        self, tmp_path: Path
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=14), config)
        model = _in_memory(hf_state(config, seed=14))
        original = SafetensorsFile.read
        file = tmp_path / "model.safetensors"

        def grow_then_read(self: SafetensorsFile, name: str) -> Any:
            file.write_bytes(file.read_bytes() + b"\0")
            return original(self, name)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(SafetensorsFile, "read", grow_then_read)
            with pytest.raises(ModelLoadError, match="changed after it was opened"):
                load_weights(model, tmp_path)


class TestConfigAndArchitecture:
    def test_a_missing_config_is_a_model_load_error(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=15))

        with pytest.raises(ModelLoadError, match=r"cannot read config\.json"):
            load_model(_source(tmp_path), CPU_FP32)

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [("{not json", "not valid JSON"), ("[1, 2]", "expected a JSON object, got list")],
    )
    def test_an_invalid_config_is_refused(self, tmp_path: Path, text: str, fragment: str) -> None:
        (tmp_path / "config.json").write_text(text, encoding="utf-8")

        with pytest.raises(ModelLoadError, match=re.escape(fragment)):
            load_model(_source(tmp_path), CPU_FP32)

    def test_a_config_that_lacks_keys_lists_them_all(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text('{"architectures": ["GPT2LMHeadModel"]}', "utf-8")

        with pytest.raises(ModelLoadError, match="vocab_size") as error:
            load_model(_source(tmp_path), CPU_FP32)

        assert "hidden_size" in error.value.context["missing"]  # type: ignore[operator]  # a tuple

    def test_an_unknown_architecture_is_refused_before_the_weights_are_looked_at(
        self, tmp_path: Path
    ) -> None:
        config = tiny_config(architecture="MysteryForCausalLM")
        write_checkpoint(tmp_path / "ok", hf_state(config), config)
        (tmp_path / "ok" / "model.safetensors").unlink()  # no weights at all

        with pytest.raises(ModelLoadError, match=r"'MysteryForCausalLM' is not supported"):
            load_model(_source(tmp_path / "ok"), CPU_FP32)

    @pytest.mark.parametrize(
        ("fmt", "fragment"),
        [(RequestFormat.PT, "pickle checkpoints are never read"), (RequestFormat.DUMMY, "random")],
    )
    def test_only_safetensors_are_read(
        self, tmp_path: Path, fmt: RequestFormat, fragment: str
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config), config)

        with pytest.raises(ModelLoadError, match=fragment):
            load_model(_source(tmp_path, format=fmt), CPU_FP32)

    def test_an_explicit_safetensors_format_is_accepted(self, tmp_path: Path) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config), config)

        model = load_model(_source(tmp_path, format=RequestFormat.SAFETENSORS), CPU_FP32)

        assert isinstance(model, GPT2LMHeadModel)

    def test_a_missing_gpu_is_a_capability_error_and_not_a_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config), config)
        monkeypatch.setattr(torch_utils, "cuda_available", lambda: False)

        with pytest.raises(CapabilityError, match="requires CUDA"):
            load_model(_source(tmp_path), DeviceConfig(device="cuda"))


class TestStreaming:
    def test_tensors_are_read_in_file_order_whatever_the_model_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=16), config, reverse=True)
        original = SafetensorsFile.read
        offsets: list[int] = []

        def record(self: SafetensorsFile, name: str) -> Any:
            offsets.append(self.tensors[name].offset)
            return original(self, name)

        monkeypatch.setattr(SafetensorsFile, "read", record)

        load_model(_source(tmp_path), CPU_FP32)

        assert len(offsets) == 28
        assert offsets == sorted(offsets)

    def test_shards_are_read_one_after_another(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=17), config, shards=3)
        original = SafetensorsFile.read
        files: list[str] = []

        def record(self: SafetensorsFile, name: str) -> Any:
            files.append(self.path.name)
            return original(self, name)

        monkeypatch.setattr(SafetensorsFile, "read", record)

        load_model(_source(tmp_path), CPU_FP32)

        assert files == sorted(files)
        assert len(set(files)) == 3

    def test_no_file_stays_mapped_after_loading(self, tmp_path: Path) -> None:
        # Windows refuses to delete a file that is still mapped.
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=18), config)

        model = load_model(_source(tmp_path), CPU_FP32)
        (tmp_path / "model.safetensors").unlink()

        assert not (tmp_path / "model.safetensors").exists()
        # The weights were copied, not mapped.
        assert torch.isfinite(model.get_parameter("wte.weight")).all()

    def test_a_failure_while_copying_surfaces_unmasked_and_leaves_no_file_locked(
        self, tmp_path: Path
    ) -> None:
        config = tiny_config()
        write_checkpoint(tmp_path, hf_state(config, seed=19), config)
        model = _in_memory(hf_state(config, seed=19))
        calls = 0

        def fail_on_the_third(name: str, tensor: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("device out of memory")

        model.load_weight = fail_on_the_third  # type: ignore[method-assign]  # a stub, on purpose

        with pytest.raises(RuntimeError, match="device out of memory"):
            load_weights(model, tmp_path)

        assert calls == 3
