# CartPole RL Trainer

Train a DQN agent to solve CartPole-v1 using Stable-Baselines3 and Farama Gymnasium.
Outputs reward/epsilon plots and episode playback GIFs.

## Quickstart (Docker — recommended)

```bash
# Build the image
docker compose build

# Train with default config
docker compose run train

# Train with custom hyperparams
docker compose run train python train.py --steps 50000 --lr 0.001

# Record playback GIF from a training run
docker compose run playback python playback.py --run runs/20240101_120000
```

Results land in `./runs/<timestamp>/` on your host machine.

---

## Quickstart (local Python)

```bash
pip install -r requirements.txt

# Train
python train.py

# Train with overrides
python train.py --steps 50000 --lr 0.001 --seed 0

# Playback
python playback.py --run runs/20240101_120000
```

---

## Hyperparameters

Edit `config.yaml` between runs to experiment. Every run saves a copy of the
exact config used, so results are reproducible.

Key parameters to play with:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `total_timesteps` | 100000 | How long to train |
| `learning_rate` | 0.0001 | How fast the network updates |
| `exploration_fraction` | 0.1 | How quickly epsilon decays to final value |
| `gamma` | 0.99 | How much future rewards are valued |
| `batch_size` | 64 | Samples per gradient update |
| `buffer_size` | 100000 | Replay memory size |

All values can be overridden via CLI flags — run `python train.py --help` for the full list.

---

## Outputs

Each run creates a directory at `runs/<timestamp>/`:

```
runs/
  20240101_120000/
    model.zip        ← trained model (load with DQN.load())
    config.yaml      ← exact config used (for reproducibility)
    rewards.png      ← episode reward curve with rolling average
    epsilon.png      ← epsilon decay curve
    playback.gif     ← recorded episodes (after running playback.py)
```

### Reward Plot

The reward plot shows raw episode rewards (faded) with a 20-episode rolling average
overlaid. The green dashed line at y=500 marks the "solved" threshold (CartPole-v1
is considered solved when the agent consistently balances the pole for 500 steps).

### Interrupt / Resume

Press `Ctrl+C` at any time to gracefully stop training. The model is saved as
`model_interrupted.zip` so you can run playback on it:

```bash
python playback.py --run runs/<timestamp> --model model_interrupted
```

---

## What is DQN?

Deep Q-Network (DQN) is a reinforcement learning algorithm that learns a
Q-function: "given this state, what's the expected future reward of taking each action?"
It uses:
- A neural network to approximate Q-values
- A replay buffer to break correlations between consecutive samples
- A separate target network for stable training

CartPole-v1 has a 4-dimensional state (position, velocity, angle, angular velocity)
and 2 actions (push left, push right). A well-trained agent balances the pole for
the full 500 steps.

---

## Experimenting

Suggested experiments to learn from:

1. *Lower `learning_rate`* (e.g. 0.00001) — slower but sometimes more stable
2. *Raise `exploration_fraction`* (e.g. 0.3) — more exploration before exploiting
3. *Lower `gamma`* (e.g. 0.9) — agent becomes more short-sighted
4. *Shrink `buffer_size`* (e.g. 10000) — less diverse experience replay
5. *Change `seed`* — different random seeds, same config; see variance in results
