# compare_prodo_and_arz.py

# Arz
# 2024 AUG 31 (SUN)

"""
verify replay buffer implementations:
    - Yellow Card Prodo, Arz
"""

# reference:


import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
import gymnasium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)


# common
class Q_Estimator(nn.Module):
    """
    action value estimator.
    """

    def __init__(self, num_observations: int, num_actions: int):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=num_observations, out_features=64),  # (minibatch_size, num_observations) -> (minibatch_size, 64)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),  # (minibatch_size, 64) -> (minibatch_size, 64)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_actions)  # (minibatch_size, 64) -> (minibatch_size, num_actions)
        )

    def forward(self, x):
        x = self.main(x)
        return x

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
num_actions = env.action_space.n
observation, observation_info = env.reset()
num_observations = len(observation)

learning_rate = 1E-3
gamma = 0.99

q_estimator_prodo = Q_Estimator(num_observations, num_actions).to(device)
target_q_estimator_prodo = Q_Estimator(num_observations, num_actions).to(device)
target_q_estimator_prodo.load_state_dict(q_estimator_prodo.state_dict())
loss_function_prodo = nn.MSELoss()
optimizer_prodo = torch.optim.Adam(q_estimator_prodo.parameters(), lr=learning_rate)

q_estimator_arz = Q_Estimator(num_observations, num_actions).to(device)
target_q_estimator_arz = Q_Estimator(num_observations, num_actions).to(device)
loss_function_arz = nn.MSELoss()
optimizer_arz = torch.optim.Adam(q_estimator_arz.parameters(), lr=learning_rate)
q_estimator_arz.load_state_dict(q_estimator_prodo.state_dict())
target_q_estimator_arz.load_state_dict(q_estimator_prodo.state_dict())


# Yellow Card Prodo
def unpack_memory_prodo(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    return states, actions, rewards, next_states, dones

# Arz
Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))

class Replay_Memory():
    def __init__(self, memory_capacity):
        self.storage = deque([], maxlen=memory_capacity)

    def remember(self, experience: Experience):
        self.storage.append(experience)

    def retrieve_random_experiences(self, batch_size):
        return random.sample(self.storage, batch_size)

    def __len__(self):
        return len(self.storage)

def unpack_memory_arz(experiences):
    experience_minibatch = Experience(*zip(*experiences))
    state_j_minibatch = torch.stack(experience_minibatch.state_t).to(device)
    action_j_minibatch = torch.LongTensor(np.array(experience_minibatch.action_t)).unsqueeze(1).to(device)
    reward_jp1_minibatch = torch.FloatTensor(np.array(experience_minibatch.reward_tp1)).to(device)
    state_jp1_minibatch = torch.stack(experience_minibatch.state_tp1).to(device)
    return state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch

def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
             q_estimator, target_q_estimator, optimizer, gamma, loss_function):

    with torch.no_grad():
        target_q_values = reward_jp1_minibatch + \
                          gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values  # (minibatch_size,)

    print("action_j_minibatch:", action_j_minibatch)
    q_values = q_estimator(state_j_minibatch).gather(1, action_j_minibatch).squeeze(1)  # (minibatch_size,)

    loss = loss_function(target_q_values, q_values)

    # optimize Q-estimator
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_values, target_q_values, loss


# example
# initialize memory
memory_capacity = 10
memory_prodo = deque(maxlen=memory_capacity)
memory_arz = Replay_Memory(memory_capacity)

# data
state_ts = [[0.0, 0.0, 0.0, 0.0], [0.3, 0.1, 0.2, 0.5]]
action_ts = [1, 0]
reward_tp1s = [-1, 1]
state_tp1s = [[0.3, 0.1, 0.2, 0.5], [1.7, 0.7, 1.2, 3.4]]
dones = False, True

# remember experience
for state_t, action_t, reward_tp1, state_tp1, done in zip(state_ts, action_ts, reward_tp1s, state_tp1s, dones):
    memory_prodo.append((state_t, action_t, reward_tp1, state_tp1, done))

    state_t = torch.FloatTensor(state_t).to(device)
    state_tp1 = torch.FloatTensor(state_tp1).to(device)
    experience = Experience(state_t, action_t, reward_tp1, state_tp1)
    memory_arz.remember(experience)

# sample
batch = list(memory_prodo)[:]
experiences = list(memory_arz.storage)[:]

# unpack
output_prodo = unpack_memory_prodo(batch)
output_arz = unpack_memory_arz(experiences)

print("output_prodo:\n", output_prodo)
print("output_arz:\n", output_arz)

# update q
states = output_prodo[0]
actions = output_prodo[1]
rewards = output_prodo[2]
next_states = output_prodo[3]
dones = output_prodo[4]

print("actions.unsqueeze(1):", actions.unsqueeze(1))
q_values_prodo = q_estimator_prodo(states).gather(1, actions.unsqueeze(1)).squeeze()
next_q_values = target_q_estimator_prodo(next_states).max(1)[0]
# target_q_values_prodo = rewards + gamma * next_q_values * (1 - dones)
target_q_values_prodo = rewards + gamma * next_q_values

loss_prodo = nn.MSELoss()(q_values_prodo, target_q_values_prodo.detach())
optimizer_prodo.zero_grad()
loss_prodo.backward()
optimizer_prodo.step()

q_values_arz, target_q_values_arz, loss_arz = \
    update_q(output_arz[0], output_arz[1], output_arz[2], output_arz[3],
             q_estimator_arz, target_q_estimator_arz, optimizer_arz, gamma,
             loss_function_arz)

print("q_values_prodo:\n", q_values_prodo, q_values_prodo.shape)
print("q_values_arz:\n", q_values_arz, q_values_arz.shape)

print("target_q_values_prodo:\n", target_q_values_prodo, target_q_values_prodo.shape)
print("target_q_values_arz:\n", target_q_values_arz, target_q_values_arz.shape)

print("loss_prodo:\n", loss_prodo, loss_prodo.shape)
print("loss_arz:\n", loss_arz, loss_arz.shape)
