"""
File holds self contained TRPO agent with simple interface.
"""
import torch
import gym
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters


class TRPOAgent:

    def __init__(self, policy, optim, discount, delta=0.01):
        self.policy = policy
        self.optim = optim
        self.discount = discount
        self.delta = delta

        policy_modules = [module for module in policy.modules() if not
                          isinstance(module, torch.nn.Sequential)]
        self.action_dims = policy_modules[-1].out_features
        self.distribution = torch.distributions.normal.Normal
        self.logstd = torch.ones(self.action_dims, requires_grad=True)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()

        self.buffers = {'log_prob': [], 'action': [], 'episode_reward': [],
                        'completed_reward': [], 'states': []}

    def __call__(self, state):
        """
        Peforms forward pass on the NN and parameterized distribution.

        Parameters
        ----------
        state : torch.Tensor
            Tensor passed into NN and distribution.

        Returns
        -------
            Action choice for each action dimension.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        normal_dist = self.distribution(self.policy(state), self.logstd.exp())
        action = normal_dist.sample()
        self.buffers['action'].append(action)
        print(action)
        exit()
        self.buffers['log_prob'].append(normal_dist.log_prob(action))
        self.buffers['states'].append(state)
        return action.numpy()

    def update(self, reward, done):
        self.buffers['episode_reward'].append(reward)
        if done:
            episode_reward = self.buffers['episode_reward']
            for ind in range(len(episode_reward) - 2, -1, -1):
                episode_reward[ind] += self.discount * episode_reward[ind + 1]
            self.buffers['completed_reward'].extend(episode_reward)
            self.buffers['episode_reward'] = []

    def kl(self, new_policy, new_std, states):
        with torch.no_grad():
            old_dist = self.distribution(self.policy(states), self.logstd.exp())
            new_dist = self.distribution(new_policy(states), new_std.exp())
            kl_matrix = torch.distributions.kl.kl_divergence(old_dist, new_dist)

        return kl_matrix.sum(1).mean()

    def surrogate_objective(self, new_policy, new_std, states, advantages):
        with torch.no_grad():
            new_dist = self.distribution(new_policy(states), new_std.exp())
            new_prob = new_dist.log_prob(torch.stack(self.buffers['action']))
            ratio = new_prob / torch.stack(self.buffers['log_prob'])
            return (ratio * advantages.view(-1, 1)).mean()

    def line_search(self):
        gradients = parameters_to_vector(
            [param.grad for param in self.policy.parameters()])
        states = torch.stack(self.buffers['states'])
        current_parameters = parameters_to_vector(self.policy.parameters())
        # TODO: Fix the initial step size to be according to paper.
        step_size = 1

        new_policy = deepcopy(self.policy)
        vector_to_parameters(current_parameters + step_size * gradients,
                             new_policy.parameters())
        new_std = self.logstd.detach() + step_size * self.logstd.grad

        decay = 2
        line_search_attempts = 10
        for attempt in range(line_search_attempts):

            if self.kl(new_policy, new_std, states) > self.delta:
                print('Shrinking Gradients')
                step_size /= decay
                vector_to_parameters(current_parameters + step_size * gradients,
                                     new_policy.parameters())
                new_std = self.logstd.detach() + step_size * self.logstd.grad
            else:
                break
        else:
            return self.policy, self.logstd

        return new_policy, new_std.requires_grad_()

    def optimize(self):
        if len(self.buffers['completed_reward']) == 0:
            return

        # Normalize rewards over episodes
        self.optim.zero_grad()
        rewards = torch.tensor(self.buffers['completed_reward'])
        print(rewards.sum() / rewards.numel())
        rewards = (rewards - rewards.mean()) / rewards.std()
        num_steps_in_batch = rewards.numel()

        # Find REINFORCE gradient
        expected_reward = 0
        for rew, prob in zip(rewards, self.buffers['log_prob']):
            expected_reward += prob * rew
        expected_reward /= num_steps_in_batch
        expected_reward.mean().backward()

        # Line search from the initial gradient
        self.policy, self.logstd = self.line_search()

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            if key != 'episode_reward':
                del storage[:num_steps_in_batch]


def main():
    env = gym.make('LunarLanderContinuous-v2')

    nn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                             torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(nn.parameters(), lr=1e-2)
    agent = TRPOAgent(nn, opt, 0.99)

    iterations = 10
    steps = 10000

    ob = env.reset()
    for _ in range(iterations):
        for _ in range(steps):
            ob, rew, done, _ = env.step(agent(ob))
            agent.update(rew, done)
            if done:
                ob = env.reset()
        agent.optimize()


main()
