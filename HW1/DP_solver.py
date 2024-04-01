import numpy as np
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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        raise NotImplementedError


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        return self.values[state]
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        delta = float('inf')
        space = self.grid_world.get_state_space()
        while(delta >= self.threshold):
            delta = 0
            for s in range(space):
                v = 0
                for action in range(4):
                    new_state, reward, done = self.grid_world.step(s, action)
                    v += self.policy[s, action] * (reward + self.discount_factor * self.values[new_state] * (1 - done))
                delta = max(delta, abs(v - self.values[s]))
                self.values[s] = v


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        stable = True
        space = self.grid_world.get_state_space()
        for s in range(space):
            cmp = float('-inf')
            old_action = self.policy[s]
            for action in range(4):
                new_state, reward, done = self.grid_world.step(s, action)
                if(cmp < reward + self.discount_factor * self.values[new_state] * (1 - done)):
                    cmp = reward + self.discount_factor * self.values[new_state] * (1 - done)
                    self.policy[s] = action
            stable = False if (old_action != self.policy[s]) else stable
        return stable



        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        stable = False
        space = self.grid_world.get_state_space()
        while not stable:
            delta = float('inf')
            while(delta >= self.threshold):
                delta, vv = 0, [0] * space
                for s in range(space):
                    new_state, reward, done = self.grid_world.step(s, self.policy[s])
                    vv[s] = reward + self.discount_factor * self.values[new_state] * (1 - done)
                    delta = max(delta, abs(vv[s] - self.values[s]))
                for s in range(space):
                    self.values[s] = vv[s]
            stable = self.policy_improvement()


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        delta = float('inf')
        space = self.grid_world.get_state_space()
        while(delta >= self.threshold):
            delta, vv = 0, [0] * space
            for s in range(space):
                cmp = float('-inf')
                for action in range(4):
                    new_state, reward, done = self.grid_world.step(s, action)
                    if(cmp < reward + self.discount_factor * self.values[new_state] * (1 - done)):
                        cmp = reward + self.discount_factor * self.values[new_state] * (1 - done)
                        self.policy[s] = action
                delta = max(delta, abs(self.values[s] - cmp))
                vv[s] = cmp
            for s in range(space):
                self.values[s] = vv[s]

import random

class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        delta = float('inf')
        space = self.grid_world.get_state_space()
        step_dict = {}
        while(delta >= self.threshold):
            delta, v = 0, 0
            Iter = [i for i in range(space - 1, -1, -1)]
            random.shuffle(Iter)
            for s in Iter:
                cmp = float('-inf')
                for action in range(4):
                    if (s, action) not in step_dict:
                        new_state, reward, done = self.grid_world.step(s, action)
                        step_dict[(s, action)] = (new_state, reward, done)
                        if done:
                            step_dict[(s, 0)] = step_dict[(s, 1)] = step_dict[(s, 2)] = step_dict[(s, 3)] = (new_state, reward, done)
                    else:
                        new_state, reward, done = step_dict[(s, action)]
                    if(cmp < reward + self.discount_factor * self.values[new_state] * (1 - done)):
                        cmp = reward + self.discount_factor * self.values[new_state] * (1 - done)
                        self.policy[s] = action
                delta = max(delta, abs(self.values[s] - cmp))
                self.values[s] = cmp
