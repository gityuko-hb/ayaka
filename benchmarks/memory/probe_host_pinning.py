"""Measure pinned vs pageable host RAM allocation limits and H2D/D2H bandwidth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ayaka.memory.host_probe import probe_host_pinning


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for probing host pinned RAM allocation and transfer speed."""
    parser = argparse.ArgumentParser(description="Probe host pinned RAM limits and transfer speed")
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=[16, 64, 256],
        help="Buffer sizes in MiB to probe",
    )
    parser.add_argument(
        "--no-pageable-fallback",
        action="store_true",
        help="Fail instead of falling back to pageable memory when pinning fails",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args(argv)

    results = probe_host_pinning(
        sizes_mb=args.sizes_mb,
        allow_pageable=not args.no_pageable_fallback,
    )
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return 0
    for r in results:
        h2d = f"{r.h2d_gbps:.2f} GB/s" if r.h2d_gbps is not None else "cpu-skipped"
        d2h = f"{r.d2h_gbps:.2f} GB/s" if r.d2h_gbps is not None else "cpu-skipped"
        mode = "pinned" if r.pinned_ok else ("pageable" if r.pageable_fallback else "failed")
        print(f"{r.size_mb:5d} MiB  {mode:8s}  alloc={r.alloc_ms:6.2f} ms  H2D={h2d}  D2H={d2h}")
        if r.error:
            print(f"           error={r.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
