# CartPole RL Trainer

Train a DQN agent on CartPole-v1 (Farama Gymnasium).
Containerized, CLI-driven, with saved plots and episode GIFs.

## Goals
- Learn RL fundamentals through hands-on experimentation
- Simple to run, easy to tweak, results you can see

## Stack
- Python 3.11
- Farama Gymnasium (CartPole-v1)
- Stable-Baselines3 (DQN)
- Matplotlib (plots)
- PyYAML (config)
- Docker + docker-compose

## Features

### Training
- `python train.py` — runs training with config.yaml
- `python train.py --lr 0.001 --steps 50000` — CLI overrides
- Saves model checkpoint to `runs/<timestamp>/model.zip`
- Saves reward curve plot to `runs/<timestamp>/rewards.png`
- Graceful Ctrl+C: saves checkpoint before exiting

### Config (config.yaml)
- learning_rate, batch_size, buffer_size,
  exploration_fraction, total_timesteps, seed

### Episode Playback
- `python playback.py --run runs/<timestamp>`
- Loads saved model, records 3 episodes as GIF
- Saves to `runs/<timestamp>/playback.gif`

### Visualization
- Reward curve (episode reward vs timestep)
- Epsilon decay curve (exploration over time)
- Rolling average overlay on reward plot

## Structure
cartpole-rl/
  train.py
  playback.py
  config.yaml
  Dockerfile
  docker-compose.yml
  README.md
  runs/  (gitignored)
