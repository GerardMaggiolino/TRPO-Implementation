"""
File holds self contained REINFORCE agent using line search.
"""
import torch
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters
import gym


class LineSearchAgent:
    """Continuous  agent."""
    def __init__(self, policy, discount=0.98, optim_lr=0.01, kl_delta=0.01):
        self.policy = policy
        self.discount = discount
        self.kl_delta = kl_delta
        self.optim = torch.optim.Adam(policy.parameters(), lr=optim_lr)
        self.distribution = torch.distributions.normal.Normal
        
        # Cuda check
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                       else torch.device('cpu'))
        policy.to(self.device)

        # Infer action and state dimensions
        policy_modules = [module for module in policy.modules() if not
                  isinstance(module, torch.nn.Sequential)]
        action_dims = policy_modules[-1].out_features

        # Set logstd
        self.logstd = torch.ones(action_dims, requires_grad=True,
                                 device=self.device)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()
        self.optim.add_param_group({'params': self.logstd})

        self.buffers = {'log_probs': [], 'actions': [], 'episode_reward': [],
                        'completed_rewards': [], 'states': []}

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
        self.buffers['actions'].append(action)
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        self.buffers['states'].append(state)
        return action.cpu().numpy()

    def kl(self, new_policy, new_std, states, grad_new=True):
        """Compute KL divergence between current policy and new one.

        Parameters
        ----------
        new_policy : TRPOAgent
        new_std : torch.Tensor
        states : torch.Tensor
            States to compute KL divergence over.
        grad_new : bool, optional
            Enable gradient of new policy.
        """
        mu1 = self.policy(states)
        log_sigma1 = self.logstd
        mu2 = new_policy(states)
        log_sigma2 = new_std

        # Detach other as gradient should only be w.r.t. to one
        if grad_new:
            mu1, log_sigma1 = mu1.detach(), log_sigma1.detach()
        else:
            mu2, log_sigma2 = mu2.detach(), log_sigma2.detach()

        # Compute KL over all states
        kl_matrix = ((log_sigma2 - log_sigma1) + 0.5 * (log_sigma1.exp().pow(2)
                     + (mu1 - mu2).pow(2)) / log_sigma2.exp().pow(2) - 0.5)

        # Sum over action dim, average over all states
        return kl_matrix.sum(1).mean()

    def surrogate_objective(self, new_policy, new_std, states, actions,
                            log_probs, advantages):
        new_dist = self.distribution(new_policy(states), new_std.exp())
        new_prob = new_dist.log_prob(actions)
        ratio = new_prob.exp() / log_probs.detach().exp()
        return (ratio * advantages.view(-1, 1)).mean()

    def line_search(self, policy_gradients, std_gradients,
                    states, actions, log_probs, rewards):
        step_size = 1
        step_size_decay = 1.5
        line_search_attempts = 10

        # New policy
        current_parameters = parameters_to_vector(self.policy.parameters())
        new_policy = deepcopy(self.policy)
        new_policy_parameters = (current_parameters +
                                 step_size * policy_gradients)
        vector_to_parameters(new_policy_parameters, new_policy.parameters())
        new_std = self.logstd.detach() + step_size * std_gradients

        #  Shrink gradient until KL constraint met and improvement
        for attempt in range(line_search_attempts):
            # Obtain kl divergence and objective
            with torch.no_grad():
                kl_value = self.kl(new_policy, new_std, states)
                objective = self.surrogate_objective(new_policy, new_std,
                                                     states, actions, log_probs,
                                                     rewards)

            # Shrink gradient if KL constraint not met or reward lower
            if kl_value > self.kl_delta or objective < 0:
                step_size /= step_size_decay
                new_policy_parameters = (current_parameters +
                                         step_size * policy_gradients)
                vector_to_parameters(new_policy_parameters,
                                     new_policy.parameters())
                new_std = self.logstd.detach() + step_size * std_gradients
            # Transfer new policy and std if KL and reward met
            else:
                vector_to_parameters(new_policy_parameters,
                                     self.policy.parameters())
                with torch.no_grad():
                    self.logstd.data[:] = new_std
                return

        print('No step computed.')

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
        rewards, actions, states, log_probs = (rewards.to(self.device), 
                                               actions.to(self.device), 
                                               states.to(self.device), 
                                               log_probs.to(self.device))
        
        # Normalize rewards over episodes
        rewards = (rewards - rewards.mean()) / rewards.std()

        # Save current parameters
        self.optim.zero_grad()
        old_policy_param = parameters_to_vector(
            [param for param in self.policy.parameters()]).detach().clone()
        old_std_param = self.logstd.detach().clone()

        # Compute regular gradient and step
        (-log_probs * rewards.view(-1, 1)).mean().backward()
        self.optim.step()

        # Find search direction by Adam
        new_policy_param = parameters_to_vector(
            [param for param in self.policy.parameters()]).detach()
        policy_gradients = new_policy_param - old_policy_param
        std_gradients = self.logstd.detach() - old_std_param

        # Restore old policy
        vector_to_parameters(old_policy_param, self.policy.parameters())
        with torch.no_grad():
            self.logstd[:] = old_std_param

        # Find new policy and std with line search using Adam gradient
        self.line_search(policy_gradients, std_gradients, states, actions,
                         log_probs, rewards)

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
