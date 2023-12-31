import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import gymnasium as gym
import gymnasium.spaces as sp
from tqdm import trange
from time import sleep
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Training parameters
'''BATCH_SIZE = 128
LR = 1e-4
EPISODES = 1000
TARGET_SCORE = 250.     # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 10000     # max memory buffer size
LEARN_STEP = 5          # how often to learn
TAU = 0.005             # for soft update of target parameters
SAVE_CHKPT = True       # save trained network .pth file'''

BATCH_SIZE = 128
LR = 1e-3
EPISODES = 5000
TARGET_SCORE = 250.     # early training stop at avg score of last 100 episodes
GAMMA = 0.99            # discount factor
MEMORY_SIZE = 10000     # max memory buffer size
LEARN_STEP = 5          # how often to learn
TAU = 1e-3              # for soft update of target parameters
SAVE_CHKPT = True      # save trained network .pth file



#%% Policy network
class QNet(nn.Module):
    # Policy Network
    def __init__(self, n_states, n_actions, n_hidden=64):
        super(QNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
            )

    def forward(self, x):
        return self.fc(x)

#%% dqn    
class DQN():
    def __init__(self, n_states, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        # model
        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # memory
        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0    # update cycle counter

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action

    def save2memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)          # target, if terminal then y_j = rewards
        q_eval = self.net_eval(states).gather(1, actions)

        # loss backprop
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen = memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

def train(env, agent, n_episodes=2000, max_steps=300, eps_start=1.0, eps_end=0.1, eps_decay=0.995, target=200, chkpt=False, checkpoint_fname="def.ckpt", fig_fname="def.png"):
    best_score = -np.inf
    best_state_dict = None

    score_hist = []
    epsilon = eps_start

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    # bar_format = '{l_bar}{bar:10}{r_bar}'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    for idx_epi in pbar:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            # next_state = next_state[0]
            agent.save2memory(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        score_hist.append(score)
        score_avg = np.mean(score_hist[-100:])
        epsilon = max(eps_end, epsilon*eps_decay)

        pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}")
        pbar.update(0)

        # if (idx_epi+1) % 100 == 0:
        #     print(" ")
        #     sleep(0.1)

        # Early stop
        if len(score_hist) >= 100:
            if score_avg >= target:
                break

        if score_avg > best_score:
            best_score = score_avg
            best_state_dict = agent.net_eval.state_dict()
            if chkpt:
                torch.save(agent.net_eval.state_dict(), checkpoint_fname)

        if idx_epi % 20 == 0:
            plotScore(score_hist, fig_fname)

    if (idx_epi+1) < n_episodes:
        print("\nTarget Reached!")
    else:
        print("\nDone!")
    

    if chkpt:
        torch.save(agent.net_eval.state_dict(), checkpoint_fname)

    return score_hist

def plotScore(scores, fig_fname):
    plt.figure()
    plt.plot(scores, linestyle='None', marker='.')
    plt.title(f"Score History (100 avg) - {fig_fname}")
    plt.xlabel("Episodes")
    plt.ylim(-500, 300)
    # plt.savefig("score_"+str(abs(args.gravity))+".png")
    plt.savefig(fig_fname)
    plt.close()


# If env is G10 and init checkpoint is from scratch
# the output will look like
# - "10.ckpt"
# - "10.png"

# If env is G5 and init checkpoint is 10.ckpt
# - "10-5.ckpt"
# - "10-5.png"

# If env is G5 and init checkpoint is from scratch
# - "5.ckpt"
# - "5.pnt"

# If env is G3 and init checkpoint is 10-5.ckpt
# - "10-5-3.ckpt"
# - "10-5-3.png"

# If env is G1 and init checkpoint is 10-5-3.ckpt
# - "10-5-3-1.ckpt"
# - "10-5-3-1.png"



def get_fnames(gravity, checkpoint_fname):
    if checkpoint_fname is None:
        checkpoint_fname = "S.ckpt"
    checkpoint_fname = checkpoint_fname.split(".ckpt")[0]
    new_fname = checkpoint_fname + "-" + str(abs(gravity))
    new_checkpoint_fname = new_fname + ".ckpt"
    new_score_fig_fname = new_fname + ".png"
    return new_checkpoint_fname, new_score_fig_fname


# Train the network
def training(args):
    env = gym.make(
        'LunarLander-v2',
        gravity=args.gravity,
        enable_wind=False
    )

    #print(filename(env.gravity,'-', args.pretrained_weights))

    checkpoint_fname, fig_fname = get_fnames(args.gravity, args.pretrained_weights)
    print("CHECKPOINT FNAME = ", checkpoint_fname)
    print("FIG FNAME = ", fig_fname)
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

    if args.pretrained_weights is not None:
        agent.net_eval.load_state_dict(torch.load(args.pretrained_weights))
        print("Weight loaded!")
    score_hist = train(
        env, 
        agent,
        n_episodes=EPISODES, 
        target=TARGET_SCORE, 
        chkpt=SAVE_CHKPT, 
        checkpoint_fname=checkpoint_fname,
        fig_fname=fig_fname,
    )
    plotScore(score_hist, fig_fname=fig_fname)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--gravity", type=float, default=-10, help="Gravity")

    args = parser.parse_args()

    training(args)