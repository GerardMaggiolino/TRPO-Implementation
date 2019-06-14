"""
TODO: File holds self contained TRPO agent with simple interface.
"""
import torch
import gym


class TRPOAgent:
    def __init__(self, policy, optim, discount):
        self.policy = policy
        self.optim = optim
        self.discount = discount

        policy_modules = [module for module in policy.modules() if not
                          isinstance(module, torch.nn.Sequential)]
        self.action_dims = policy_modules[-1].out_features
        self.distribution = torch.distributions.normal.Normal
        self.logstd = torch.ones(self.action_dims)

        self.buffers = {'log_prob': [], 'action': [], 'episode_reward': [],
                        'completed_reward': []}

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
        normal_dist = self.distribution(self.policy(state), self.logstd)
        action = normal_dist.sample()
        self.buffers['action'].append(action)
        self.buffers['log_prob'].append(normal_dist.log_prob(action))
        return action

    def update(self, reward, done):
        self.buffers['episode_reward'].append(reward)
        if done:
            episode_reward = self.buffers['episode_reward']
            for ind in range(len(episode_reward) - 2, -1, -1):
                episode_reward[ind] += self.discount * episode_reward[ind + 1]
            self.buffers['completed_reward'].extend(episode_reward)
            self.buffers['episode_reward'] = []

    def optimize(self):
        self.optim.zero_grad()
        rewards = torch.tensor(self.buffers['completed_reward'])
        returnboi = rewards.sum().item()
        rewards = (rewards - rewards.mean()) / rewards.std()
        num_steps_in_batch = rewards.numel()

        expected_reward = 0
        for rew, prob in zip(rewards, self.buffers['log_prob']):
            expected_reward -= prob * rew
        expected_reward /= num_steps_in_batch
        expected_reward.mean().backward()
        self.optim.step()

        del self.buffers['log_prob'][:num_steps_in_batch]
        del self.buffers['action'][:num_steps_in_batch]
        self.buffers['completed_reward'] = []

        return returnboi


def main():
    env = gym.make('LundarLanderContinuous-v2')

    nn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                             torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(nn.parameters(), lr=1e-2)
    agent = TRPOAgent(nn, opt, 0.99)

    iterations = 10
    steps = 5000

    ob = env.reset()
    for _ in range(iterations):
        for _ in range(steps):
            ob, rew, done, _ = env.step(agent(ob))
            agent.update(rew, done)
            if done:
                ob = env.reset()
        print(agent.optimize())


main()
