"""
File holds self contained TRPO agent with simple interface.
"""
import torch
import gym
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters


class TRPOAgent:

    def __init__(self, policy, discount, delta=0.01):
        self.policy = policy
        self.discount = discount
        self.delta = delta

        policy_modules = [module for module in policy.modules() if not
                          isinstance(module, torch.nn.Sequential)]
        self.action_dims = policy_modules[-1].out_features
        self.distribution = torch.distributions.normal.Normal
        self.logstd = torch.ones(self.action_dims, requires_grad=True)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()

        self.buffers = {'log_probs': [], 'actions': [], 'episode_reward': [],
                        'completed_rewards': [], 'states': []}

        self.recording = []

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
        self.buffers['actions'].append(action)
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        self.buffers['states'].append(state)
        return action.numpy()

    def update(self, reward, done):
        self.buffers['episode_reward'].append(reward)
        if done:
            self.recording.append(sum(self.buffers['episode_reward']))
            episode_reward = self.buffers['episode_reward']
            for ind in range(len(episode_reward) - 2, -1, -1):
                episode_reward[ind] += self.discount * episode_reward[ind + 1]
            self.buffers['completed_rewards'].extend(episode_reward)
            self.buffers['episode_reward'] = []

    def kl(self, new_policy, new_std, states):
        with torch.no_grad():
            mu1 = self.policy(states)
            sigma1 = self.logstd.exp()
            mu2 = new_policy(states)
            sigma2 = new_std.exp()
            kl_matrix = ((sigma2/sigma1).log() + 0.5 * (sigma1.pow(2) +
                         (mu1 - mu2).pow(2)) / sigma2.pow(2) - 0.5)
        return kl_matrix.sum(1).mean()

    def surrogate_objective(self, new_policy, new_std, states, actions,
                            log_probs, advantages):
        with torch.no_grad():
            new_dist = self.distribution(new_policy(states), new_std.exp())
            new_prob = new_dist.log_prob(actions)
            ratio = new_prob.exp() / log_probs.exp()
            return (ratio * advantages.view(-1, 1)).mean()

    def line_search(self, gradients, states, actions, log_probs, rewards):
        # TODO: Fix the initial step size to be according to paper.
        step_size = 1
        step_size_decay = 2
        line_search_attempts = 10

        current_parameters = parameters_to_vector(self.policy.parameters())
        new_policy = deepcopy(self.policy)
        vector_to_parameters(current_parameters + step_size * gradients,
                             new_policy.parameters())
        new_std = self.logstd.detach() + step_size * self.logstd.grad

        #  Shrink gradient until KL constraint met and improvement
        for attempt in range(line_search_attempts):
            # Shrink gradient if KL constraint not met or reward lower
            if (self.kl(new_policy, new_std, states) > self.delta or
                self.surrogate_objective(new_policy, new_std, states, actions,
                                         log_probs, rewards) < 0):
                print('Shrinking Gradients')
                step_size /= step_size_decay
                vector_to_parameters(current_parameters + step_size * gradients,
                                     new_policy.parameters())
                new_std = self.logstd.detach() + step_size * self.logstd.grad
            # Break and return new policy and std if KL and reward met
            else:
                break
        else:
            # Return old policy and std if constraints never met
            return self.policy, self.logstd

        return new_policy, new_std.requires_grad_()

    def optimize(self):
        # Return if no completed episodes
        if len(self.buffers['completed_rewards']) == 0:
            return

        # Convert all buffers to tensors
        num_batch_steps = len(self.buffers['completed_rewards'])
        rewards = torch.tensor(self.buffers['completed_rewards'])
        actions = torch.stack(self.buffers['actions'][:num_batch_steps])
        states = torch.stack(self.buffers['states'][:num_batch_steps])
        log_probs = torch.stack(self.buffers['log_probs'][:num_batch_steps])

        # Normalize rewards over episodes
        rewards = (rewards - rewards.mean()) / rewards.std()
        num_steps_in_batch = rewards.numel()

        # Find REINFORCE gradient
        self.policy.zero_grad()
        expected_reward = 0
        for rew, prob in zip(rewards, log_probs):
            expected_reward += prob * rew
        expected_reward /= num_steps_in_batch
        expected_reward.mean().backward()

        # Line search with new gradients
        gradients = parameters_to_vector(
            [param.grad for param in self.policy.parameters()])
        self.policy, self.logstd = self.line_search(gradients, states, actions,
                                                    log_probs, rewards)

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            if key != 'episode_reward':
                del storage[:num_steps_in_batch]
        print(sum(self.recording) / len(self.recording))
        self.recording = []


def main():
    env = gym.make('LunarLanderContinuous-v2')

    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(nn, 0.99, 0.001)

    iterations = 10
    steps = 8000

    ob = env.reset()
    t = 0
    for _ in range(iterations):
        for _ in range(steps):
            t += 1
            ob, rew, done, _ = env.step(agent(ob))
            if t >= 800:
                t = 0
                agent.update(rew, True)
                ob = env.reset()
            else:
                agent.update(rew, done)
                if done:
                    t = 0
                    ob = env.reset()
        agent.optimize()


main()
