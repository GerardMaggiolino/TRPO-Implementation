"""
Policy gradient with REINFORCE over steps.

Z scores rewards over potentially multiple episodes, averages gradient
over batches of steps.
"""

import gym
import util
import torch



def load_checkpoint(checkpoint_path, config):
    '''
    Loads a checkpoint if it exists. Otherwise, initializes.
    '''
    # Policy
    policy = util.Network(config['ob_dim'], config['policy_hidden_units'] +
                          [config['action_dim']])
    optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])
    # Std 
    std = torch.ones(config['action_dim'], requires_grad=True)
    # Recording 
    ep_reward = []

    # Try to load from checkpoint
    try:
        checkpoint = torch.load(checkpoint_path)

        # Reset policy
        policy_param = checkpoint['policy_param']
        policy = util.Network(*policy_param)
        optim = torch.optim.Adam(policy.parameters(), config['policy_lr'])

        # Load state dicts
        policy.load_state_dict(checkpoint['policy'])
        std = checkpoint['std']
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        optim.load_state_dict(checkpoint['optim'])
        ep_reward = checkpoint['ep_reward']
        print(f'Resuming training from episode {len(ep_reward)}')
    except FileNotFoundError:
        optim.add_param_group({'params': std, 'lr': config['std_lr']})
        print('NOTE: Training from scratch.')

    return (policy, optim, std, ep_reward)


def save_checkpoint(checkpoint_path, policy, optim, std, ep_reward, config):
    '''
    Saves checkpoint.
    '''
    torch.save({
        'policy': policy.state_dict(),
        'policy_param': (config['ob_dim'], config['policy_hidden_units'] +
                         [config['action_dim']]),
        'optim': optim.state_dict(),
        'std': std,
        'ep_reward': ep_reward
    }, checkpoint_path)


def main():
    '''
    Calls training procedure.
    '''

    # Hyperparameters
    config = {
        'action_dim': None,
        'ob_dim': None,
        'policy_hidden_units': [],
        'max_steps': 500,
        'batch_size': 5000,
        'max_episodes': 10,
        'discount': 0.99,
        'policy_lr': 0.1,
        'std_lr': 1e-2,
        'epochs': 50
    }

    env = gym.make('CartPole-v1')
    config['action_dim'] = 2 # env.action_space.low.size
    config['ob_dim'] = env.observation_space.low.size

    # Load checkpoint
    checkp_path = 'checkpoint_reinforce_steps.tar'
    policy, optim, std, episode_reward = load_checkpoint(checkp_path, config)
    device = (torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    policy.to(device)
    std.to(device)

    # Train over epochs (batches of normalized steps)
    for ep in range(1, config['epochs'] + 1):
        # Recording for batch
        rewards = []
        log_probs = []
        # Recording for episodes within a batch
        episode_start_step = 0
        episodes = 1

        # Run fixed number of steps with multiple episodes
        step = 0
        ob = torch.from_numpy(env.reset()).float().to(device)
        while step < config['batch_size']:
            # Perform action
            action, prob = util.classify_discrete(policy(ob))
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float().to(device)
            rewards.append(reward)
            log_probs.append(prob)
            step += 1

            # If done with episode
            if done or (step - episode_start_step >= config['max_steps']):
                episodes += 1
                episode_reward.append(sum(rewards[episode_start_step:]))
                # Discounted rewards over episode
                running = 0
                for ind in range(len(rewards) - 1, episode_start_step - 1, -1):
                    running = rewards[ind] + config['discount'] * running
                    rewards[ind] = running
                # Exit if can't run another episode in batch size
                if step + config['max_steps'] > config['batch_size']:
                    break
                # Exit if over number of episodes desired
                if episodes > config['max_episodes']:
                    break
                # Continue otherwise
                episode_start_step = step
                ob = torch.from_numpy(env.reset()).float().to(device)

        # Average rewards over all steps in the batch
        rewards = util.z_score_rewards(rewards)
        # Compute loss
        loss = 0
        for prob, rew in zip(log_probs, rewards):
            loss -= prob * rew
        loss /= step
        # Average over output dimensions and backwards
        optim.zero_grad()
        loss.mean().backward()
        optim.step()

        # Save checkpoint 
        if ep % 1 == 0:
            print('Sigma: ', [round(i, 3) for i in std.detach().numpy()])
            print(f'Episode {len(episode_reward)}\t'
                  'Reward:\t', round(sum(episode_reward[-(episodes - 1):])
                                     / (episodes - 1), 2))
            save_checkpoint(checkp_path, policy, optim, std, episode_reward,
                            config)

    # Graph info 
    util.graph_reward(episode_reward)

    # Render 
    ob = torch.from_numpy(env.reset()).float().to(device)
    for step in range(config['max_steps']):
        with torch.no_grad():
            env.render()
            action, prob = util.classify_discrete(policy(ob))
            ob, reward, done, _ = env.step(action)
            ob = torch.from_numpy(ob).float().to(device)
            if done:
                break
    env.close()


if __name__ == '__main__':
    main()
