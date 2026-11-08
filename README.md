# StylePlan

A JAX implementation of **StylePlan**, a partner-conditioned controller for zero-shot coordination with unfamiliar partners.

StylePlan combines slow behavioral memory from an exponential moving average of behavioral fingerprints with fast intent estimates from recent interaction history. A style-conditioned Gaussian prior and a disagreement-aware gate fuse these representations to condition a recurrent coordination policy.

## Setup

Requires **Python 3.10+**, **JAX**, and **NumPy**. Real Overcooked V2 runs additionally require a compatible **JAXMarl** installation providing `jaxmarl.environments.overcooked_v2`.

With the dependencies and `setuptools>=68` already installed:

```bash
git clone https://github.com/MIKE-TECH-CO/StylePlan.git
cd StylePlan
python -m pip install --no-deps --no-build-isolation -e .
```

For offline servers, transfer the source and dependency wheels in advance; install dependencies from local wheels before the editable installation. `requirements-cpu-verified.txt` records a macOS CPU verification environment and is not a server GPU dependency lockfile.

## Quick start

Run a small CPU check using the bundled synthetic coordination environment:

```bash
JAX_PLATFORMS=cpu python -m styleplan_paper train \
  --env synthetic --seed 0 --updates 1 --pretrain-updates 0 \
  --num-envs 2 --steps 8 --epochs 1 --horizon 8 \
  --history-window 8 --intent-window 4 --eval-episodes 2 \
  --output outputs/smoke
```

Evaluate the saved policy against held-out synthetic partners:

```bash
JAX_PLATFORMS=cpu python -m styleplan_paper evaluate \
  --env synthetic --checkpoint outputs/smoke/checkpoint \
  --seed 100 --horizon 8 --eval-episodes 4 \
  --output outputs/synthetic_eval.json
```

Add `--switch-step 4 --score-window 4` to the evaluation command to test a within-episode partner switch. Use a different output path for each run.

Training saves `run.json`, `metrics.jsonl`, `summary.json`, and `checkpoint/` under the output directory. Evaluation saves a JSON report. The synthetic example is an implementation check, not an Overcooked benchmark result.

## Overcooked V2

Use `--env overcooked --layout cramped_room --horizon 400 --public-actions` with an installed JAXMarl backend. The adapter uses full-grid observations and explicitly observable completed partner actions.

Training requires frozen `--partner-checkpoints`; supply disjoint `--heldout-checkpoints` for held-out evaluation. The built-in model partner adapter accepts `no_partner_model` checkpoints, which can be trained with `--self-play --ablation no_partner_model`. Custom partner pools can be supplied through `--partner-factory module:function`.

Pretrained partner checkpoints and the paper's benchmark results are not bundled. Available ablations include `full`, `no_partner_model`, `style_only`, `intent_only`, `concat`, `no_arbitration`, `no_style`, and `no_intent`.

## Code

- `styleplan_paper/model.py`: behavioral fingerprints, slow memory, intent prediction, prior, and policy conditioning.
- `styleplan_paper/training.py` and `losses.py`: recurrent PPO and auxiliary objectives.
- `styleplan_paper/envs.py` and `partners.py`: environments and partner pools.
- `styleplan_paper/evaluation.py` and `checkpoint.py`: partner-switch evaluation and checkpoint handling.
- `styleplan_paper/cli.py`: training and evaluation entry points.

Run `python -m styleplan_paper train --help` or `python -m styleplan_paper evaluate --help` for all options.
