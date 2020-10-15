"""
File holds self contained REINFORCE agent.
"""
import torch
import gym


class REINFORCEAgent:
    """REINFORCE with ADAM optimization."""
    def __init__(self, policy, discount=0.98, optim_lr=0.01):
        self.policy = policy
        self.discount = discount
        self.optim = torch.optim.Adam(policy.parameters(), lr=optim_lr)
        self.distribution = torch.distributions.normal.Normal
        
        # Cuda check
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                       else torch.device('cpu'))
        policy.to(self.device)
        
        # Set logstd
        policy_modules = [module for module in policy.modules() if not
                          isinstance(module, torch.nn.Sequential)]
        action_dims = policy_modules[-1].out_features
        self.logstd = torch.ones(action_dims, requires_grad=True,
                                 device=self.device)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()
        self.optim.add_param_group({'params': self.logstd})

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
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        # Parameterize distribution with policy, sample action
        normal_dist = self.distribution(self.policy(state), self.logstd.exp())
        action = normal_dist.sample()
        # Save information
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        return action.cpu().numpy()

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
        rewards, log_probs = rewards.to(self.device), log_probs.to(self.device)
        
        # Normalize rewards over episodes
        advantages = (rewards - rewards.mean()) / rewards.std()

        # Optimize
        self.optim.zero_grad()
        (-log_probs * advantages.view(-1, 1)).mean().backward()
        self.optim.step()

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            if key != 'episode_reward':
                del storage[:num_batch_steps]

    def train(self, env_name, seed=None, batch_size=12000, iterations=100,
              max_episode_length=None, verbose=False):

        # Initialize env
        env = gym.make(env_name)
        if seed is not None:
            torch.manual_seed(seed)
            env.seed(seed)
        if max_episode_length is None:
            max_episode_length = float('inf')
        # Recording
        recording = {'episode_reward': [[]],
                     'episode_length': [0],
                     'num_episodes_in_iteration': []}

        # Begin training
        observation = env.reset()
        for iteration in range(iterations):
            # Set initial value to 0
            recording['num_episodes_in_iteration'].append(0)

            for step in range(batch_size):
                # Take step with agent
                observation, reward, done, _ = env.step(self(observation))

                # Recording, increment episode values
                recording['episode_length'][-1] += 1
                recording['episode_reward'][-1].append(reward)

                # End of episode
                if (done or
                        recording['episode_length'][-1] >= max_episode_length):
                    # Calculate discounted reward
                    discounted_reward = recording['episode_reward'][-1].copy()
                    for index in range(len(discounted_reward) - 2, -1, -1):
                        discounted_reward[index] += self.discount * \
                                                    discounted_reward[index + 1]
                    self.buffers['completed_rewards'].extend(discounted_reward)

                    # Set final recording of episode reward to total
                    recording['episode_reward'][-1] = \
                        sum(recording['episode_reward'][-1])
                    # Recording
                    recording['episode_length'].append(0)
                    recording['episode_reward'].append([])
                    recording['num_episodes_in_iteration'][-1] += 1

                    # Reset environment
                    observation = env.reset()

            # Print information if verbose
            if verbose:
                num_episode = recording['num_episodes_in_iteration'][-1]
                avg = (round(sum(recording['episode_reward'][-num_episode:-1])
                             / (num_episode - 1), 3))
                print(f'Average Reward over Iteration {iteration}: {avg}')
            # Optimize after batch
            self.optimize()

        # Return recording information
        return recording

    def save_model(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'logstd': self.logstd
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.logstd = checkpoint['logstd'].to(self.device)
