import numpy as np
import wandb
from collections import deque
from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space 
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        G = 0
        ret = 0
        for i in reversed(range(len(state_trace))):
            state, action, reward = state_trace[i], action_trace[i], reward_trace[i]
            G = self.discount_factor * G + reward
            ret += abs(G - self.q_values[state][action])
            self.q_values[state][action] += self.lr * (G - self.q_values[state][action])
        return ret
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        for state in range(self.state_space):
            self.policy[state] = np.argmax(self.q_values[state])


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        # wandb.init(
                # project="RL_HW2_log",
                # name=f"MC_epsilon_{self.epsilon}",
        # )
        for i in self.q_values:
            i = 0;
        iter_episode = 0
        current_state = self.grid_world.reset()
        reward_buffer = []
        loss_buffer = []
        while iter_episode < max_episode:

            state_trace   = []
            action_trace  = []
            reward_trace  = []

            done = False
            episode_reward = 0
            cnt = 0
            while not done:
                cnt += 1
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(4)
                else:
                    action = np.argmax(self.q_values[current_state])

                next_state, reward, done = self.grid_world.step(action)
                episode_reward += reward

                state_trace.append(current_state)
                action_trace.append(action)
                reward_trace.append(reward)
                current_state = next_state

            episode_loss = self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()

            reward_buffer.append(episode_reward / cnt)
            loss_buffer.append(episode_loss / cnt)



            iter_episode += 1
            print(f"Episode {iter_episode} completed.")

class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        td_error = r + self.discount_factor * self.q_values[s2][a2] * (1 - is_done) - self.q_values[s][a]
        self.q_values[s][a] += self.lr * td_error
        return abs(td_error)

        

    def greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.q_values[state])


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        for i in self.q_values:
            i = 0;
        iter_episode = 0
        current_state = self.grid_world.reset()
        reward_buffer = []
        loss_buffer = []
        while iter_episode < max_episode:
            episode_reward = 0
            episode_loss = 0
            current_action = self.greedy_action(current_state)
            done = False
            cnt = 0
            while not done:
                cnt += 1
                next_state, reward, done = self.grid_world.step(current_action)
                next_action = self.greedy_action(next_state)
                episode_loss += self.policy_eval_improve(current_state, current_action, reward, next_state, next_action, done)
                current_state = next_state
                current_action = next_action
                episode_reward += reward

            reward_buffer.append(episode_reward / cnt)
            loss_buffer.append(episode_loss / cnt)


            iter_episode += 1
            print(f"Episode {iter_episode} completed.")

class Q_Learning(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> list:
        indices = np.random.choice(len(self.buffer), self.sample_batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def policy_eval_improve(self, s, a, r, s2, is_done):
        td_error = r + self.discount_factor * np.max(self.q_values[s2]) * (1 - is_done) - self.q_values[s][a]
        self.q_values[s][a] += self.lr * td_error
        return abs(td_error)

    def run(self, max_episode=1000) -> None:
        for i in self.q_values:
            i = 0;
        reward_buffer = []
        loss_buffer = []
        for iter_episode in range(max_episode):
            current_state = self.grid_world.reset()
            done = False

            episode_reward = 0
            episode_loss = 0
            cnt = 0
            while not done:
                cnt += 1
                action = np.random.choice(4) if np.random.rand() < self.epsilon else np.argmax(self.q_values[current_state])

                next_state, reward, done = self.grid_world.step(action)

                self.add_buffer(current_state, action, reward, next_state, done)

                if len(self.buffer) >= self.sample_batch_size and iter_episode % self.update_frequency == 0:
                    batch = self.sample_batch()
                    for s, a, r, s2, d in batch:
                        episode_loss += self.policy_eval_improve(s, a, r, s2, d)
                episode_reward += reward
                current_state = next_state
                # print("one iter")

            reward_buffer.append(episode_reward / cnt)
            loss_buffer.append(episode_loss / cnt)



            print(f"Episode {iter_episode + 1} completed.")


