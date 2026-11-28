"""Internal dict codecs for gRPC sampling, logprob, and structured-output payloads."""

from __future__ import annotations

import json
from typing import cast

from ayaka.core.constraints import (
    ConstraintKind,
    StructuralTagItem,
    StructuralTagSpec,
    StructuredParams,
    ToolCallSyntax,
    ToolSchema,
)
from ayaka.core.outputs import TokenLogprob
from ayaka.core.params import LogprobMode, SamplingParams

__all__ = [
    "as_float",
    "as_int",
    "as_opt_float",
    "as_opt_int",
    "decode_logprob",
    "decode_sampling",
    "decode_structured",
    "encode_logprob",
    "encode_sampling",
    "encode_structured",
]


def as_int(value: object, default: int) -> int:
    return (
        int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def as_opt_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value) if isinstance(value, (int, float)) else None


def as_float(value: object, default: float) -> float:
    return (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def as_opt_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def encode_sampling(params: SamplingParams) -> dict[str, object]:
    """Serialize :class:`~ayaka.core.params.SamplingParams`."""
    bias_pairs: list[tuple[int, float]] = (
        list(params.logit_bias.items())
        if isinstance(params.logit_bias, dict)
        else list(cast("tuple[tuple[int, float], ...]", params.logit_bias or ()))
    )
    return {
        "n": params.n,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "max_tokens": params.max_tokens,
        "min_tokens": params.min_tokens,
        "ignore_eos": params.ignore_eos,
        "stop": list(params.stop),
        "stop_token_ids": sorted(params.stop_token_ids),
        "include_stop_str_in_output": params.include_stop_str_in_output,
        "seed": params.seed,
        "allowed_token_ids": (
            sorted(params.allowed_token_ids) if params.allowed_token_ids is not None else None
        ),
        "logit_bias": [[int(t), float(b)] for t, b in bias_pairs],
        "logprobs": params.logprobs,
        "prompt_logprobs": params.prompt_logprobs,
        "logprob_token_ids": (
            list(params.logprob_token_ids) if params.logprob_token_ids is not None else None
        ),
        "logprob_mode": str(params.logprob_mode),
        "return_sampling_support": params.return_sampling_support,
    }


def decode_sampling(data: dict[str, object]) -> SamplingParams:
    """Deserialize :class:`~ayaka.core.params.SamplingParams`."""
    allowed_raw = data.get("allowed_token_ids")
    logprob_raw = data.get("logprob_token_ids")
    bias_raw = data.get("logit_bias")
    logit_bias: tuple[tuple[int, float], ...] | None = None
    if isinstance(bias_raw, list) and bias_raw:
        logit_bias = tuple(
            (int(pair[0]), float(pair[1])) for pair in cast("list[list[float]]", bias_raw)
        )
    return SamplingParams(
        n=as_int(data.get("n"), 1),
        temperature=as_float(data.get("temperature"), 1.0),
        top_p=as_float(data.get("top_p"), 1.0),
        top_k=as_int(data.get("top_k"), -1),
        min_p=as_float(data.get("min_p"), 0.0),
        repetition_penalty=as_float(data.get("repetition_penalty"), 1.0),
        frequency_penalty=as_float(data.get("frequency_penalty"), 0.0),
        presence_penalty=as_float(data.get("presence_penalty"), 0.0),
        max_tokens=as_opt_int(data.get("max_tokens")),
        min_tokens=as_int(data.get("min_tokens"), 0),
        ignore_eos=bool(data.get("ignore_eos", False)),
        stop=tuple(cast("list[str]", data.get("stop") or [])),
        stop_token_ids=frozenset(
            int(t) for t in cast("list[int]", data.get("stop_token_ids") or [])
        ),
        include_stop_str_in_output=bool(data.get("include_stop_str_in_output", False)),
        seed=as_opt_int(data.get("seed")),
        allowed_token_ids=(
            frozenset(int(t) for t in cast("list[int]", allowed_raw))
            if isinstance(allowed_raw, list)
            else None
        ),
        logit_bias=logit_bias,
        logprobs=as_opt_int(data.get("logprobs")),
        prompt_logprobs=as_opt_int(data.get("prompt_logprobs")),
        logprob_token_ids=(
            tuple(int(t) for t in cast("list[int]", logprob_raw))
            if isinstance(logprob_raw, list)
            else None
        ),
        logprob_mode=LogprobMode(str(data.get("logprob_mode", LogprobMode.RAW))),
        return_sampling_support=bool(data.get("return_sampling_support", False)),
    )


def encode_structured(params: StructuredParams) -> dict[str, object]:
    """Serialize all seven :class:`~ayaka.core.constraints.ConstraintKind` specs."""
    syntax = (
        {
            "open": params.syntax.open,
            "close": params.syntax.close,
            "open_token": params.syntax.open_token,
            "close_token": params.syntax.close_token,
            "open_ids": list(params.syntax.open_ids),
            "arguments_key": params.syntax.arguments_key,
            "separator": params.syntax.separator,
        }
        if params.syntax is not None
        else None
    )
    st = (
        {
            "triggers": list(params.structural_tag.triggers),
            "at_least_one": params.structural_tag.at_least_one,
            "stop_after_first": params.structural_tag.stop_after_first,
            "tags": [
                {
                    "begin": item.begin,
                    "end": item.end,
                    "schema_json": (
                        json.dumps(item.schema, separators=(",", ":"), ensure_ascii=False)
                        if item.schema is not None
                        else None
                    ),
                    "grammar": item.grammar,
                }
                for item in params.structural_tag.tags
            ],
        }
        if params.structural_tag is not None
        else None
    )
    return {
        "kind": str(params.kind),
        "schema_json": (
            json.dumps(params.schema, separators=(",", ":"), ensure_ascii=False)
            if params.schema is not None
            else None
        ),
        "tools": [
            {
                "name": t.name,
                "parameters_json": json.dumps(
                    t.parameters, separators=(",", ":"), ensure_ascii=False
                ),
            }
            for t in params.tools
        ],
        "syntax": syntax,
        "tool_choice": params.tool_choice,
        "strict": params.strict,
        "parallel": params.parallel,
        "start_after": params.start_after,
        "reasoning_open": params.reasoning_open,
        "pattern": params.pattern,
        "choices": list(params.choices),
        "grammar": params.grammar,
        "structural_tag": st,
    }


def decode_structured(data: dict[str, object]) -> StructuredParams:
    """Deserialize :class:`~ayaka.core.constraints.StructuredParams` preserving JSON key order."""
    kind = ConstraintKind(str(data["kind"]))
    schema_raw = data.get("schema_json")
    schema = (
        cast("dict[str, object]", json.loads(schema_raw)) if isinstance(schema_raw, str) else None
    )
    tools = tuple(
        ToolSchema(
            name=str(item["name"]),
            parameters=cast("dict[str, object]", json.loads(str(item["parameters_json"]))),
        )
        for item in cast("list[dict[str, object]]", data.get("tools") or [])
    )
    syn_raw = data.get("syntax")
    syntax: ToolCallSyntax | None = None
    if isinstance(syn_raw, dict):
        smap = cast("dict[str, object]", syn_raw)
        syntax = ToolCallSyntax(
            open=str(smap.get("open", "")),
            close=str(smap.get("close", "")),
            open_token=as_opt_int(smap.get("open_token")),
            close_token=as_opt_int(smap.get("close_token")),
            open_ids=tuple(int(t) for t in cast("list[int]", smap.get("open_ids") or [])),
            arguments_key=str(smap.get("arguments_key", "arguments")),
            separator=str(smap.get("separator", "")),
        )
    st_raw = data.get("structural_tag")
    structural_tag: StructuralTagSpec | None = None
    if isinstance(st_raw, dict):
        st_map = cast("dict[str, object]", st_raw)
        items: list[StructuralTagItem] = []
        for entry in cast("list[dict[str, object]]", st_map.get("tags") or []):
            entry_schema = entry.get("schema_json")
            items.append(
                StructuralTagItem(
                    begin=str(entry["begin"]),
                    end=str(entry.get("end", "")),
                    schema=(
                        cast("dict[str, object]", json.loads(entry_schema))
                        if isinstance(entry_schema, str)
                        else None
                    ),
                    grammar=(
                        str(entry["grammar"]) if isinstance(entry.get("grammar"), str) else None
                    ),
                )
            )
        structural_tag = StructuralTagSpec(
            tags=tuple(items),
            triggers=tuple(cast("list[str]", st_map.get("triggers") or [])),
            at_least_one=bool(st_map.get("at_least_one", False)),
            stop_after_first=bool(st_map.get("stop_after_first", False)),
        )
    return StructuredParams(
        kind=kind,
        schema=schema,
        tools=tools,
        syntax=syntax,
        tool_choice=str(data.get("tool_choice", "auto")),
        strict=bool(data.get("strict", False)),
        parallel=bool(data.get("parallel", True)),
        start_after=as_opt_int(data.get("start_after")),
        reasoning_open=as_opt_int(data.get("reasoning_open")),
        pattern=str(data["pattern"]) if isinstance(data.get("pattern"), str) else None,
        choices=tuple(cast("list[str]", data.get("choices") or [])),
        grammar=str(data["grammar"]) if isinstance(data.get("grammar"), str) else None,
        structural_tag=structural_tag,
    )


def encode_logprob(lp: TokenLogprob) -> dict[str, object]:
    return {
        "token_id": lp.token_id,
        "logprob": lp.logprob,
        "rank": lp.rank,
        "decoded": lp.decoded,
        "top": [[tid, val] for tid, val in lp.top],
        "top_decoded": list(lp.top_decoded),
    }


def decode_logprob(data: dict[str, object]) -> TokenLogprob:
    decoded_raw = data.get("decoded")
    top_raw = cast("list[list[float]]", data.get("top") or [])
    return TokenLogprob(
        token_id=as_int(data["token_id"], 0),
        logprob=as_float(data["logprob"], 0.0),
        rank=as_opt_int(data.get("rank")),
        decoded=str(decoded_raw) if isinstance(decoded_raw, str) else None,
        top=tuple((int(pair[0]), float(pair[1])) for pair in top_raw),
        top_decoded=tuple(cast("list[str]", data.get("top_decoded") or [])),
    )
