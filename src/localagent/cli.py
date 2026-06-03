"""LocalAgent CLI. `localagent <command> ...`

Commands:
  model-info <model.yaml>      Construct a config and report param count vs the 100M budget.
  train <stage> <config.yaml>  Run a pipeline stage (pretrain|sft|distill|rl|eval).
  synth <config.yaml>          Generate synthetic agent data.
  eval <checkpoint>            Run the eval harness.
  export <target> <ckpt> <out> Export to gguf|onnx|executorch.
  chat <checkpoint>            Launch the terminal agent demo.
"""

from __future__ import annotations

import argparse

from localagent import __version__


def _model_info(args) -> None:
    from localagent.model.config import ModelConfig

    cfg = ModelConfig.from_yaml(args.config)
    n = cfg.estimate_params()
    cfg.assert_within_budget()
    print(f"{cfg.name}: ~{n/1e6:.2f}M params (budget 100M) — OK")
    print(f"  d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads}/{cfg.n_kv_heads} "
          f"ffn={cfg.ffn_hidden} vocab={cfg.vocab_size}")
    extras = []
    if cfg.factorized:
        extras.append(f"factorized embed_dim={cfg.embed_dim}")
    if cfg.n_loops > 1:
        extras.append(f"depth-recurrence x{cfg.n_loops} (effective depth {cfg.effective_depth})")
    if extras:
        print("  " + " · ".join(extras))


def _train(args) -> None:
    from localagent.pipeline.flow import run_stage

    run_stage(args.stage, args.config)


def _synth(args) -> None:
    from localagent.data.agent_synth import synthesize

    synthesize(args.config)


def _eval(args) -> None:
    from localagent.eval.harness import run

    run(args.checkpoint)


def _export(args) -> None:
    import importlib

    mod = importlib.import_module(f"localagent.inference.export.to_{args.target}")
    mod.export(args.checkpoint, args.out)


def _chat(args) -> None:
    raise NotImplementedError("TODO(phase-7): load ckpt + Agent + REPL (see demos/chat_cli.py)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="localagent", description=__doc__)
    p.add_argument("--version", action="version", version=f"localagent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    mi = sub.add_parser("model-info", help="report param count vs budget")
    mi.add_argument("config")
    mi.set_defaults(func=_model_info)

    tr = sub.add_parser("train", help="run a pipeline stage")
    tr.add_argument("stage", choices=["pretrain", "sft", "distill", "rl", "eval"])
    tr.add_argument("config")
    tr.set_defaults(func=_train)

    sy = sub.add_parser("synth", help="generate synthetic agent data")
    sy.add_argument("config")
    sy.set_defaults(func=_synth)

    ev = sub.add_parser("eval", help="run eval harness")
    ev.add_argument("checkpoint")
    ev.set_defaults(func=_eval)

    ex = sub.add_parser("export", help="export to an on-device runtime")
    ex.add_argument("target", choices=["gguf", "onnx", "executorch"])
    ex.add_argument("checkpoint")
    ex.add_argument("out")
    ex.set_defaults(func=_export)

    ch = sub.add_parser("chat", help="terminal agent demo")
    ch.add_argument("checkpoint")
    ch.set_defaults(func=_chat)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
