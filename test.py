import gymnasium as gym
import torch
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw, ImageFont
from train import DQN
import sys


# Training parameters
BATCH_SIZE = 128
LR = 1e-3
EPISODES = 5000
TARGET_SCORE = 250.     # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 10000     # max memory buffer size
LEARN_STEP = 5          # how often to learn
TAU = 1e-3              # for soft update of target parameters
SAVE_CHKPT = True       # save trained network .pth file

# Environment parameters
GRAVITY = -10.0
WIND_POWER = 20
TURBULENCE_POWER = 0.0
ENABLE_WIND = False


#%% Test Lunar Lander
def testLander(env, agent, loop=3):
    for i in range(loop):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for idx_step in range(500):
            action = agent.getAction(state, epsilon=0)
            env.render()
            state, reward, done, _, _ = env.step(action)
            if done:
                break
    env.close()

def TextOnImg(img, score):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"Score={score: .2f}", fill=(255, 255, 255))

    return np.array(img)

def save_frames_as_gif(frames, filename, path="gifs/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    fps = 30
    duration = 1000 * 1/fps
    print("Saving gif...", end="")
    imageio.mimsave(path + filename + ".gif", frames, duration=duration)

    print("Done!")

def gym2gif(env, agent, filename="gym_animation", loop=3):
    for i in range(loop):
        frames = []
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        score = 0
        for idx_step in range(500):
            frame = env.render()
            frames.append(TextOnImg(frame, score))
            action = agent.getAction(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
            score += reward
            if done:
                break
        save_frames_as_gif(frames, filename=filename+f"_{i:02d}")
    env.close()



env = gym.make(
    'LunarLander-v2', 
    render_mode="rgb_array",
    gravity=GRAVITY,
    enable_wind=ENABLE_WIND,
    wind_power=WIND_POWER,
    turbulence_power=TURBULENCE_POWER,
)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
agent = DQN(
    n_states = num_states,
    n_actions = num_actions,
    batch_size = BATCH_SIZE,
    lr = LR,
    gamma = GAMMA,
    mem_size = MEMORY_SIZE,
    learn_step = LEARN_STEP,
    tau = TAU,
)

checkpoint = sys.argv[1]
agent_state_dict = torch.load(checkpoint)
agent.net_eval.load_state_dict(agent_state_dict)

gym2gif(env, agent, loop=5)
# testLander(env, agent, loop=3)