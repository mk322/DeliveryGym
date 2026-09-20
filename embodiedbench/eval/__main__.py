"""CLI: `python -m embodiedbench.eval {run,compare} ...`"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="embodiedbench-eval",
        description="Evaluate an OpenAI-compatible model on the courier "
                    "benchmark, or compare two result files (paired, with CIs).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the benchmark against a model endpoint")
    r.add_argument("--model", required=True,
                   help="model name the endpoint expects")
    r.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible base URL (…/v1)")
    r.add_argument("--api-key", default=None)
    r.add_argument("--split", choices=("val", "trainprobe"), default="val",
                   help="val = the 64 immutable benchmark seeds (the number); "
                        "trainprobe = 32 training seeds, for gap analysis")
    r.add_argument("--seed-list", default=None,
                   help="comma-separated explicit seeds (overrides --split; "
                        "results are then NOT the benchmark number)")
    r.add_argument("--limit", type=int, default=0,
                   help="first N seeds only, for smoke tests")
    r.add_argument("--out", default="results", help="output directory")
    r.add_argument("--tag", default=None,
                   help="name for the result files (default: model name)")
    r.add_argument("--workers", type=int, default=4,
                   help="concurrent episodes (each is its own conversation)")
    r.add_argument("--max-tokens", type=int, default=400)
    r.add_argument("--max-requeries", type=int, default=3)
    r.add_argument("--history-turns", type=int, default=8)
    r.add_argument("--max-turns", type=int, default=None,
                   help="turn cap per shift (default: the validation yaml's, "
                        "100). The training runs validated at 100; a shorter "
                        "cap is a different, labelled number")
    r.add_argument("--transcripts", action="store_true",
                   help="keep full per-turn transcripts in memory/results")

    c = sub.add_parser("compare",
                       help="paired comparison of two summary .json files")
    c.add_argument("a"); c.add_argument("b")

    args = parser.parse_args()
    from embodiedbench.eval.run import cmd_compare, cmd_run
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
