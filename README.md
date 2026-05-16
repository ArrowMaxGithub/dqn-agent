# Q-learning and Deep-Q-Networks for 2-Player-Durak

Training and testing parameters are set in main.py

To start the container:

1. sudo docker compose build
2. sudo docker compose run --rm dqn-agent

Docker image is based on [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch) and tested with an AMD RX 6800 XT (gfx1030)


## Q-Learning:

Manual CPU implementation based on dynamically growing Q-Value-Tables

## Deep-Q-Networks:

Implementation by [Ray RLLib](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn)

Action masking implemented as TorchRLModule

GPU-supported learning via PyTorch - fallback CPU training

Optional Dueling- and Double-DQN modes

## Environment:

Parallel PettingZoo environment with active player masking

Perfect public knowledge tracking for played cards:

Status(IntEnum):
    Unknown = 0
    MyCard = 1
    OpponentCard = 2
    Attack = 3
    Defense = 4
    InDeck = 5
    Discarded = 6