# Simple Grid World Navigation with DQN and Q-Learning

This is an basic grid world environment and two reinforcement learning agents (DQN and Q-Learning) for navigation.

## Environment (`env.py`)

The environment is a discrete 2D grid world flatten to a single scaler.

### Configuration

The environment can be configured using the following parameters:

* `n`: Grid size (n x n).
* `p`: Probability of a cell being free during random map generation.
* `dirt_count`: The number of dirt cells (goal) in the environment.

### Usage

```python
import env

# Example usage
environment = env.Env(n=5, p=0.8, dirt_count=1)
observation = environment.reset(seed=42) # reset with a seed for reproducibility
# ... perform actions ...
```
# Agent (`agent.py`)
DQNAgent, QLAgent are provided and can be trained using episode loop.

# Example usage
```python
agent = DQNAgent(
    env=environment,
    learning_rate=0.1,
    initial_epsilon=1.0,
    epsilon_decay=0.99,
    final_epsilon=0.01,
    discount_factor=0.9,
)
```
