"""
File holds self contained REINFORCE agent with simple interface.
"""
import torch


class REINFORCEAgent:
    """REINFORCE with ADAM optimization."""

    def __init__(self, policy, discount, optim_lr=0.01):
        self.policy = policy
        self.discount = discount
        self.optim = torch.optim.Adam(policy.parameters(), lr=optim_lr)

        policy_modules = [module for module in policy.modules() if not
                          isinstance(module, torch.nn.Sequential)]
        self.action_dims = policy_modules[-1].out_features
        self.distribution = torch.distributions.normal.Normal
        self.logstd = torch.ones(self.action_dims, requires_grad=True)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()

        self.buffers = {'log_probs': [],  'episode_reward': [],
                        'completed_rewards': []}

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
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        return action.numpy()

    def update(self, reward, done):
        """Updates REINFORCE reward buffer and done status.

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
            # Compute discounted reward
            episode_reward = self.buffers['episode_reward']
            for ind in range(len(episode_reward) - 2, -1, -1):
                episode_reward[ind] += self.discount * episode_reward[ind + 1]
            self.buffers['completed_rewards'].extend(episode_reward)
            self.buffers['episode_reward'] = []

    def optimize(self):
        # Return if no completed episodes
        if len(self.buffers['completed_rewards']) == 0:
            return

        # Convert all buffers to tensors
        num_batch_steps = len(self.buffers['completed_rewards'])
        rewards = torch.tensor(self.buffers['completed_rewards'])
        log_probs = torch.stack(self.buffers['log_probs'][:num_batch_steps])

        # Normalize rewards over episodes
        advantages = (rewards - rewards.mean()) / rewards.std()

        # Optimize
        self.policy.zero_grad()
        (-log_probs * advantages.view(-1, 1)).mean().backward()
        self.optim.step()

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            if key != 'episode_reward':
                del storage[:num_batch_steps]
