"""How the tool-call parser's time grows with the length of the arguments: linear or not.

A call whose ``arguments`` hold one string of ``N`` characters is fed to the parser cut every
``--cut`` characters (one fragment per decoded token is the usual case: a token is a few
characters), and the time is taken for ``N`` and for ``4 N``. A linear parser takes about four
times as long for ``4 N``; a quadratic one about sixteen times. The script prints the ratio and,
with ``--out``, writes every run.

    uv run python benchmarks/parsers/tool_body.py --chars 50000 --cut 1 7 32 --out r.json

It measures the parser alone, on the CPU, with no model and no server; the numbers say how the
parser scales, not what a request costs. No threshold is checked here (no timing assertion lives
in CI); the issue that asks for the measurement says what ratio it accepts.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _machine import machine, write

from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.parsers.output import OutputFormat

SCHEMA: Final = "ayaka.tool-body/1"


def _fragments(text: str, cut: int) -> list[RequestOutput]:
    pieces = [text[i : i + cut] for i in range(0, len(text), cut)]
    outputs: list[RequestOutput] = []
    for index, piece in enumerate(pieces):
        last = index == len(pieces) - 1
        outputs.append(
            RequestOutput(
                request_id=RequestId("bench"),
                outputs=(
                    CompletionOutput(
                        index=0,
                        token_ids=(1,),
                        text=piece,
                        finish_reason=FinishReason.STOP if last else None,
                    ),
                ),
                finished=last,
                usage=Usage(prompt_tokens=1, completion_tokens=len(pieces) if last else 0),
            )
        )
    return outputs


def time_parse(fmt: OutputFormat, chars: int, cut: int) -> float:
    """Seconds to parse one call with ``chars`` characters of arguments, cut every ``cut``."""
    body = '{"name": "write", "arguments": {"text": "' + "x" * chars + '"}}'
    outputs = _fragments(f"<tool_call>{body}</tool_call>", cut)
    parser = fmt.parser(call_id_prefix="call", tools=True)
    started = time.perf_counter()
    for output in outputs:
        parser.feed(output)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chars", type=int, default=50_000, help="N: characters of arguments")
    parser.add_argument("--cut", type=int, nargs="+", default=[1, 7, 32], help="piece sizes")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()

    fmt = OutputFormat.resolve(tool_calls="hermes")
    rows: list[dict[str, object]] = []
    for cut in args.cut:
        runs: dict[int, list[float]] = {}
        for chars in (args.chars, 4 * args.chars):
            time_parse(fmt, chars, cut)  # warm-up
            runs[chars] = [time_parse(fmt, chars, cut) for _ in range(args.repeat)]
        small, large = (statistics.median(runs[c]) for c in (args.chars, 4 * args.chars))
        rows.append(
            {
                "cut": cut,
                "chars": args.chars,
                "seconds_n": runs[args.chars],
                "seconds_4n": runs[4 * args.chars],
                "median_us_per_char_n": small / args.chars * 1e6,
                "median_us_per_char_4n": large / (4 * args.chars) * 1e6,
                "ratio_4n_over_n": large / small,
            }
        )
        print(
            f"cut {cut:>3}: N={args.chars} {small * 1e3:8.2f} ms, 4N {large * 1e3:8.2f} ms, "
            f"ratio {large / small:.2f} (linear ~4, quadratic ~16)"
        )
    result = {
        "schema": SCHEMA,
        "benchmark": "parsers/tool_body",
        "machine": machine(),
        "rows": rows,
    }
    write(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
