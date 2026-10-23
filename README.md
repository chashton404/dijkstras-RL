# Dijkstra RL Agent

Trains a PPO agent to play "Are you smarter than Dijkstra?" and watch it play
in a raylib window.

## Setup

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy pyray
```

Place all four files in the same folder:
```
rl/
├── game_core.py   # shared graph logic (no raylib)
├── train.py       # gymnasium env + GNN policy + training
├── viewer.py      # raylib viewer — watch the agent play
└── README.md
```

## Train

```bash
python train.py
```

Training takes ~5–15 minutes depending on your machine.
A progress line prints every 5,000 steps showing the optimal-match rate.
The best model is saved as `best_model.zip` and the final model as `agent.zip`.

## Watch

```bash
python viewer.py
```

Controls in the viewer window:

| Key     | Action                          |
|---------|---------------------------------|
| `A`     | Toggle auto-play on/off         |
| `SPACE` | Step the agent one move         |
| `R`     | New random graph                |
| `Q`     | Quit                            |

The agent's path is shown in **purple**. Once it finishes, the **gold** optimal
path appears so you can compare.

## How it works

### Environment (`DijkstraGameEnv`)

| | Detail |
|---|---|
| **State** | Flat adjacency matrix (N×N weights) + per-node features (is_current, is_start, is_end, is_visited) |
| **Action** | `Discrete(N)` with action masking over legal neighbor moves |
| **Reward** | `-total_cost + bonus` on reaching end; penalty for invalid/revisit moves |

### Policy (`GraphFeatureExtractor`)

A two-branch MLP:
- **Adjacency branch** — processes the N×N weight matrix → 128-dim embedding
- **Node feature branch** — processes the N×4 feature block → 64-dim embedding
- **Fusion layer** — concatenates both → 128-dim → PPO actor/critic heads

### Training

Maskable PPO with 8 parallel environments, 600k timesteps (~30 min on CPU).
The mask removes illegal moves so the agent learns path choice instead of interface noise.
A well-trained agent matches the optimal path ~60–80% of the time.

## What the agent learns

Early training: random walks over legal edges.
Mid training: learns to reach the end without revisiting dead ends.
Late training: learns to prefer shorter edges and approximate Dijkstra's greedy behaviour.

The interesting question is whether the agent discovers something *like*
Dijkstra's algorithm implicitly — or a different heuristic that works
well on this specific graph distribution.
