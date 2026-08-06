# dqn-combat-sim

Hand-written DQN trained on a custom 2D combat simulation (Gymnasium env). See
`dqn_combat_sim_action_plan.md` for the full design spec and build plan.

## Setup

```bash
conda activate 312ml
pip install -r requirements.txt
```

## Layout

See `dqn_combat_sim_action_plan.md` Section 3 for the full repository structure
and Section 4 for build phases.

## Tests

```bash
pytest tests/
```

## Training

```bash
python training/train.py
```

## Watching a trained policy

```bash
python training/watch_policy.py --checkpoint runs/<run>/checkpoints/checkpoint_best.pt
```
