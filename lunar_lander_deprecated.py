import gymnasium as gym
import math
import torch
from torch.optim import SGD
from torch.nn import Linear, Softmax, Sequential

model = Sequential(Linear(8,16),Linear(16,4), Softmax())

print(list(model.parameters()))
def policy(observation):
   input = torch.tensor(observation)
   action_probabilities = model(input)
   p = action_probabilities.cumsum(0)
   idx = torch.searchsorted(p, torch.rand(1))
   return idx[0].item()


def loss_function(observation, desired_move, model):

   f_x = model(observation)

   desired_probability_output = torch.tensor([0 for _ in range(len(f_x))])
   desired_probability_output[desired_move]=1
   return ((desired_probability_output-f_x)**2).mean()


env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
optimizer = SGD(params=model.parameters(), lr=0.03)
for i in range(100000):
   recent_rewards_array = []
   for j in range(10):
      action = policy(observation)  # this is where we inserted our policy
      observation, reward, terminated, truncated, info = env.step(action)
      recent_rewards_array.append([observation, action, reward])
      if terminated or truncated:
         observation, info = env.reset()
         print(i)
         break
   #itt jon az hogy tanul
   biggest_3_indexes = []
   reward_values = [recent_rewards_array[k][2] for k in range(len(recent_rewards_array))]
   reward_values_sorted = [recent_rewards_array[k][2] for k in range(len(recent_rewards_array))]
   reward_values_sorted.sort(reverse=True)
   for j in range(3):
      print(reward_values_sorted)
      biggest_3_indexes.append(reward_values.index(reward_values_sorted[j]))
   best_observations = [recent_rewards_array[j][0] for j in biggest_3_indexes]
   best_actions = [recent_rewards_array[j][1] for j in biggest_3_indexes]
   for j in range(3):
      loss = loss_function(torch.tensor(best_observations[j]), best_actions[j], model)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
   


#print(recent_rewards_array)

env.close()