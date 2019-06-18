"""
File holds self contained TRPO agent with simple interface.
"""
import torch
import gym
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters
from time import time

class TRPOAgent:

    def __init__(self, policy, discount, delta=0.01, cg_iteration=10,
                 cg_dampening=0.001):
        self.policy = policy
        self.discount = discount
        self.delta = delta
        self.cg_iteration = cg_iteration
        self.cg_dampening = cg_dampening

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
        # Parameterize distribution with policy, sample action
        normal_dist = self.distribution(self.policy(state), self.logstd.exp())
        action = normal_dist.sample()
        # Save information
        self.buffers['actions'].append(action)
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        self.buffers['states'].append(state)
        return action.numpy()

    def update(self, reward, done):
        """Updates TRPOAgents reward buffer and done status.

        Parameters
        ----------
        reward : float
            Reward for previous timestep.
        done : bool
            Done status for previous timestep.
        """
        self.buffers['episode_reward'].append(reward)
        # If episode is done
        if done:
            # Recording of episode reward
            self.recording.append(sum(self.buffers['episode_reward']))
            # Compute discounted reward
            episode_reward = self.buffers['episode_reward']
            for ind in range(len(episode_reward) - 2, -1, -1):
                episode_reward[ind] += self.discount * episode_reward[ind + 1]
            self.buffers['completed_rewards'].extend(episode_reward)
            self.buffers['episode_reward'] = []

    def kl(self, new_policy, new_std, states):
        """Compute KL divergence between current policy and new one.

        Parameters
        ----------
        new_policy : TRPOAgent
        new_std : torch.Tensor
        states : torch.Tensor
            States to compute KL divergence over.
        """
        mu1 = self.policy(states)
        sigma1 = self.logstd.exp()
        mu2 = new_policy(states)
        sigma2 = new_std.exp()
        kl_matrix = ((sigma2/sigma1).log() + 0.5 * (sigma1.pow(2) +
                     (mu1 - mu2).pow(2)) / sigma2.pow(2) - 0.5)
        return kl_matrix.sum(1).mean()

    def surrogate_objective(self, new_policy, new_std, states, actions,
                            log_probs, advantages):
        new_dist = self.distribution(new_policy(states), new_std.exp())
        new_prob = new_dist.log_prob(actions)
        ratio = new_prob.exp() / log_probs.exp()
        return (ratio * advantages.view(-1, 1)).mean()

    def line_search(self, gradients, states, actions, log_probs, rewards):
        step_size = (2 * self.delta / gradients.dot(
                     self.fisher_vector_direct(states, gradients))).sqrt()
        step_size_decay = 2
        line_search_attempts = 10
        print(step_size)

        current_parameters = parameters_to_vector(self.policy.parameters())
        new_policy = deepcopy(self.policy)
        vector_to_parameters(current_parameters + step_size * gradients,
                             new_policy.parameters())
        new_std = self.logstd.detach() + step_size * self.logstd.grad

        #  Shrink gradient until KL constraint met and improvement
        for attempt in range(line_search_attempts):
            # Obtain kl divergence and objective
            with torch.no_grad():
                kl_value = self.kl(new_policy, new_std, states)
                objective = self.surrogate_objective(new_policy, new_std,
                                                     states, actions, log_probs,
                                                     rewards)
            # Shrink gradient if KL constraint not met or reward lower
            if kl_value > self.delta or objective < 0:
                step_size /= step_size_decay
                vector_to_parameters(current_parameters + step_size *
                                     gradients, new_policy.parameters())
                new_std = self.logstd.detach() + step_size * self.logstd.grad
            # Break and return new policy and std if KL and reward met
            else:
                break
        else:
            # Return old policy and std if constraints never met
            return self.policy, self.logstd

        return new_policy, new_std.requires_grad_()

    def fisher_vector_direct(self, states, vector):
        vector = vector.clone().requires_grad_()
        # Compute gradient of KL w.r.t. network param
        self.policy.zero_grad()
        self.kl(self.policy, self.logstd, states).backward(retain_graph=True)
        kl_gradient = parameters_to_vector([p.grad for p in
                                            self.policy.parameters()])
        # Compute gradient of gradient dot product w.r.t. network param
        kl_gradient.dot(vector).backward()
        output_vector = parameters_to_vector([p.grad for p in
                                              self.policy.parameters()])
        return output_vector + self.cg_dampening * vector.detach()

    def conjugate_gradient(self, states, b):
        p = b.clone()
        r = b.clone()
        x = torch.zeros(*p.shape)
        rdotr = r.double().dot(r.double())
        for _ in range(self.cg_iteration):
            z = self.fisher_vector_direct(states, p)
            v = rdotr / p.double().dot(z.double())
            x += v * p.cpu()
            r -= v * z
            newrdotr = r.double().dot(r.double())
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            print('rdotr value: ', rdotr)
            if rdotr < 1e-10:
                break
        return x

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

        s = time()
        # Find REINFORCE gradient
        self.policy.zero_grad()
        expected_reward = 0
        for rew, prob in zip(rewards, log_probs):
            expected_reward += prob * rew
        expected_reward /= num_steps_in_batch
        expected_reward.mean().backward()

        gradients = parameters_to_vector(
            [param.grad for param in self.policy.parameters()])
        print('REINFORCE time: ', time() - s)

        s = time()
        # Line search with new gradients
        gradients = self.conjugate_gradient(states, gradients)
        self.policy, self.logstd = self.line_search(gradients, states, actions,
                                                    log_probs, rewards)
        print('CG time: ', time() - s)

        # Print average reward over steps
        print(sum(self.recording) / len(self.recording))

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            if key != 'episode_reward':
                del storage[:num_steps_in_batch]
        self.recording = []


def main():
    env = gym.make('LunarLanderContinuous-v2')

    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                             torch.nn.Linear(64, 2))
    agent = TRPOAgent(nn, 0.99, 0.01)

    iterations = 1500
    steps = 4000

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
