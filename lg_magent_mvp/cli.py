from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_env_from_dotenv, load_settings, apply_tracing_env, ensure_env
from .app import compile_graph_with_settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="lg-audit", description="Medical PDF audit (LangGraph)")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run an audit")
    run.add_argument("--doc", required=False, default="data/MC15 Deines Chiropractic.pdf", help="Path to PDF")
    run.add_argument("--question", required=False, default="Audit this medical document for completeness, coding, and quality.")
    run.add_argument("--thread", required=False, default="cli", help="Thread id for persistence")
    run.add_argument("--out", required=False, help="Write structured report JSON to this path")
    run.add_argument("--approve", action="store_true", help="Set approved=true (used with pause-before-finalize)")

    approve = sub.add_parser("approve", help="Approve a paused run and resume to finalize")
    approve.add_argument("--thread", required=True, help="Thread id to resume")
    approve.add_argument("--doc", required=False, default="data/MC15 Deines Chiropractic.pdf")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    load_env_from_dotenv()
    settings = load_settings()
    apply_tracing_env(settings)
    ensure_env(settings)

    graph = compile_graph_with_settings(settings)

    if args.cmd == "run":
        state = {
            "question": args.question,
            "doc_path": args.doc,
        }
        if args.approve:
            state["approved"] = True
        out = graph.invoke(state, config={"configurable": {"thread_id": args.thread}})
        # Print narrative and notes
        print(out.get("narrative") or out.get("answer", ""))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            report = out.get("report")
            if report:
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"Saved report to {args.out}", file=sys.stderr)
            else:
                print(f"Warning: No report data found. Available keys: {list(out.keys())}", file=sys.stderr)
                # Save empty report as fallback
                with open(args.out, "w") as f:
                    json.dump({}, f, indent=2)
        return 0

    if args.cmd == "approve":
        # Resume paused thread and approve
        out = graph.invoke({"approved": True, "doc_path": args.doc}, config={"configurable": {"thread_id": args.thread}})
        print(out.get("narrative") or out.get("answer", ""))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

