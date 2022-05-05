from model_wrapper_init import wrappers
from model_wrapper_init import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99 #For Bellman approximation
BATCH_SIZE = 32 #Size of batch from reply buffer
REPLAY_SIZE = 10000 #Max capacity of buffer
LEARNING_RATE = 1e-4 #Learning rate used in adam optimizer
SYNC_TARGET_FRAMES = 1000 #How frquently we sync model weights from training to target model
REPLAY_START_SIZE = 10000 #Count of frames to wait before starting training to populate reply buffer

#The decay of randomness over time
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
    #Create a list of random indicies and repack entries into NumPy arrays
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states) #\ for line continuation

#Interacts with environment and saves interaction into experience replay buffer
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample() #for random action
        else:
            #Choose best action from all possible actions
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        #store in experience buffer
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        
        #Handle end of episode situation
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    
    #Convert to tensors
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    #pass observations to first model and extract specific Q-values for the action
    #Image available in issues->images->figure 2
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    #Apply target network on next state observations and get max Q-value along same action dimension of 1
    next_state_values = tgt_net(next_states_v).max(1)[0]

    #If transition in the batch is from last step in episode, then action value
    #should not have a discounted reward of the next state since there is no next state
    next_state_values[done_mask] = 0.0

    #Detach the value from its computation graph so gradients don't flow into the neural network
    next_state_values = next_state_values.detach()

    #Bellman approximation with mean squared error loss
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    #Create our environemnt with the required wrappers
    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    #Create experience reply buffer of the required replay size and pass to agent
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) #optimizer
    total_rewards = [] #full episode rewards
    frame_idx = 0 #coutner for frames
    #track speed
    ts_frame = 0 
    ts = time.time()
    #best mean reward
    best_mean_reward = None

    while True:
        frame_idx += 1
        #decrese epsilon
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        #make a single step in the environment
        reward = agent.play_step(net, epsilon, device=device)
        #if it is not the final step in the episode, report progress
        #progress looks at speed as a count of frames per second
        #number of episodes played
        #mean reward for last 100 episodes
        #current value for epsilon
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            #if mean reward for last 100 episodes reaches max, report and save parameters
            #if mean reward exceeds boundary step training
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break
        #wait for enough data to start
        if len(buffer) < REPLAY_START_SIZE:
            continue    
        
        #syncs parameters from main network
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        #training loop algorith
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
