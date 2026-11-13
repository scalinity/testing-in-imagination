"""Registry of stage implementations, and the two built-in no-op stages.

A stage implementation is a callable taking one Context (see stage.py). It returns when the work
is complete. When ctx.stop turns true it saves what it needs and raises StageInterrupted; it may
also raise StageInterrupted by itself to continue in a fresh process (exit 10, never counted as a retry).
A stage's command in graph.json is [registered name, arguments...]; modules that register further
stages are listed in graph.json under "stage_modules" and imported before the lookup.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

REGISTRY = {}


class StageInterrupted(Exception):
    """Raised by an implementation after it has checkpointed in response to ctx.stop."""


def register(name):
    def add(function):
        REGISTRY[name] = function
        return function
    return add


def resolve(name, modules=()):
    for module in modules:
        importlib.import_module(module)
    return REGISTRY[name]


@register("noop")
def noop(ctx):
    """Writes one small output and one approved aggregate. --count sets the number it reports."""
    parser = argparse.ArgumentParser(prog="noop")
    parser.add_argument("--count", type=int, default=1)
    count = parser.parse_args(ctx.args).count
    (ctx.out_dir / "result.json").write_text(json.dumps({"count": count}) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, "count": count})
    ctx.progress(1, 1, "item")


@register("noop_slow")
def noop_slow(ctx):
    """Works through --steps items, one completed key at a time, so it can be stopped and resumed."""
    parser = argparse.ArgumentParser(prog="noop_slow")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    options = parser.parse_args(ctx.args)
    resumed_at = (ctx.load_checkpoint() or {}).get("next", 0)
    for step in range(options.steps):
        key = "step-%04d" % step
        if ctx.is_done(key):
            continue
        if ctx.stop:
            ctx.save_checkpoint({"next": step, "resumed_at": resumed_at})
            raise StageInterrupted()
        time.sleep(options.sleep)
        ctx.mark_done(key, {"step": step})
        ctx.save_checkpoint({"next": step + 1, "resumed_at": resumed_at})
        ctx.progress(step + 1, options.steps, "step")
    done = ctx.results()
    (ctx.out_dir / "result.json").write_text(json.dumps({"keys": sorted(done), "count": len(done)}) + "\n")
    ctx.approve(ctx.stage_id + ".json", {"stage": ctx.stage_id, "count": len(done)})
