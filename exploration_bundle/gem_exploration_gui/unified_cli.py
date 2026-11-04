from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from gem_exploration.config import validate_problem
from .unified import compile_unified_question, dump_unified_yaml, load_unified_yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile a public GEM exploration.yaml into the tested internal problem.yaml execution contract")
    ap.add_argument("exploration", help="Path to exploration.yaml")
    ap.add_argument("--output", "-o", default=None, help="Compiled problem.yaml path")
    ap.add_argument("--normalized-public", default=None, help="Optional path to rewrite normalized public exploration.yaml")
    args = ap.parse_args(argv)

    q = load_unified_yaml(args.exploration)
    public, internal = compile_unified_question(q)
    validate_problem(internal)

    src = Path(args.exploration).expanduser().resolve()
    out = Path(args.output).expanduser().resolve() if args.output else src.with_name(src.stem + ".compiled.problem.yaml")
    out.write_text(yaml.safe_dump(internal, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if args.normalized_public:
        Path(args.normalized_public).expanduser().resolve().write_text(dump_unified_yaml(public), encoding="utf-8")
    print(f"COMPILE PASS: {src}")
    print(f"PUBLIC PRIMITIVE: {q.primitive}/{q.detail}")
    print(f"INTERNAL EXECUTION: {internal['operator']['name']}")
    print(f"OUTPUT: {out}")
    print("NEW FBA CALLS: ZERO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
