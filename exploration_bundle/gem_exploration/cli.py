from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SUPPORTED_OPERATORS, load_problem
from .data import infer_design_variables, load_landscape
from .engine import plan_problem, run_problem


def parser():
    p = argparse.ArgumentParser(
        prog="gem-explore",
        description="GEM generalized calculator-style landscape exploration engine",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Execute one YAML exploration problem")
    r.add_argument("--problem", required=True)
    r.add_argument("--landscape")
    r.add_argument("--output")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--seed", type=int, default=20260822)

    v = sub.add_parser("validate", help="Validate YAML and optionally resolve it against a landscape")
    v.add_argument("--problem", required=True)
    v.add_argument("--landscape")
    v.add_argument("--seed", type=int, default=20260822)

    pl = sub.add_parser("plan", help="Resolve the calculator question without running the analysis")
    pl.add_argument("--problem", required=True)
    pl.add_argument("--landscape", required=True)
    pl.add_argument("--seed", type=int, default=20260822)

    i = sub.add_parser("inspect", help="Inspect an existing landscape")
    i.add_argument("--landscape", required=True)
    i.add_argument("--no-fluxes", action="store_true")

    sub.add_parser("list-operators")
    sub.add_parser("self-test")
    return p


def main(argv=None):
    args = parser().parse_args(argv)

    if args.cmd == "list-operators":
        print("\n".join(sorted(SUPPORTED_OPERATORS)))
        return 0

    if args.cmd == "inspect":
        df, meta = load_landscape(args.landscape, include_fluxes=not args.no_fluxes)
        root = Path(args.landscape) if Path(args.landscape).is_dir() else None
        variables = infer_design_variables(df, root)
        print(
            json.dumps(
                {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "design_variables": variables,
                    "first_columns": list(df.columns[:80]),
                    "source": meta,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "validate":
        spec = load_problem(args.problem)
        print("PROBLEM SPEC PASS:", spec["name"])
        if args.landscape:
            plan = plan_problem(spec, args.landscape, args.seed)
            print("LANDSCAPE CONTRACT PASS")
            print(json.dumps(plan, indent=2, default=str))
        return 0

    if args.cmd == "plan":
        plan = plan_problem(load_problem(args.problem), args.landscape, args.seed)
        print(json.dumps(plan, indent=2, default=str))
        return 0

    if args.cmd == "run":
        out = run_problem(
            load_problem(args.problem),
            args.landscape,
            args.output,
            args.overwrite,
            args.seed,
        )
        print("EXPLORATION COMPLETE:", out)
        return 0

    if args.cmd == "self-test":
        from .selftest import run_self_test

        run_self_test()
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
