import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    #Constructor, has a table
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    #Iteraction with the enviroment to obtain next transition from environment
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, extra_info = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state) #tuple for training later

    #Receives the state of the environemnt and finds the best action to take from this state
    #Finds the best action by taking largest value we have in the table
    #Used in the test method and in method that performs the value update to get value of next state
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    #Update vlaues table using one step from the environment
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v #Bellman approximation
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + ALPHA * (new_val)

    #plays one full episode using the provided test environemnt
    #uses value table to find the best action to take
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, extra_info = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment= "-q-learning")

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        #Training
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        #Output of results
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)

        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
    
        if reward > best_reward:
            print("Best reward updated %.3f to %.3f " % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iteration!" %iter_no)
            break
    writer.close()

