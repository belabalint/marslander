import gymnasium as gym
import torch
from torch.optim import SGD
from torch.nn import Linear, Softmax, Sequential

# Create a neural network model using Sequential
model = Sequential(
    Linear(8, 16),
    Linear(16, 4),
    Softmax(dim=0)  # Ensure softmax is applied along the correct dimension
)


def policy(observation, model):
    input = torch.tensor(observation, dtype=torch.float32)  # Explicitly set dtype
    action_probabilities = model(input)
    p = action_probabilities.cumsum(0)
    idx = torch.searchsorted(p, torch.rand(1))
    return idx.item()  # Simplify to return the scalar value

def loss_function(observation, desired_move, model):
    f_x = model(observation)
    desired_probability_output = torch.zeros_like(f_x)
    desired_probability_output[desired_move] = 1
    return torch.mean((desired_probability_output - f_x) ** 2)  # Use torch.mean for average loss

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
optimizer = SGD(model.parameters(), lr=0.03)

for i in range(100000):
    recent_rewards_array = []

    for j in range(10):
        action = policy(observation, model)  # Pass the model to the policy function
        observation, reward, terminated, truncated, info = env.step(action)
        recent_rewards_array.append([observation, action, reward])

        if terminated or truncated:
            observation, info = env.reset()
            print(i)

    reward_values = [entry[2] for entry in recent_rewards_array]  # Extract reward values

    # Find the indices of the top 3 rewards using argsort
    top_3_indexes = torch.argsort(torch.tensor(reward_values), descending=True)[:3]

    best_observations = [recent_rewards_array[j][0] for j in top_3_indexes]
    best_actions = [recent_rewards_array[j][1] for j in top_3_indexes]


    for j in range(3):
        loss = loss_function(torch.tensor(best_observations[j], dtype=torch.float32),
                             best_actions[j], model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
env.close()