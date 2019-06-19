import gym
import gym_driving
import torch
import numpy.random
import matplotlib.pyplot as plt
from trpoagent import TRPOAgent
from reinforceagent import REINFORCEAgent
from linesearchagent import LineSearchAgent


def main():
    env = gym.make('Driving-v2')
    torch.manual_seed(1)
    env.seed(1)
    numpy.random.seed(1)

    state_dim = len(env.observation_space.low)
    action_dim = len(env.action_space.low)

    # Agent
    nn = torch.nn.Sequential(torch.nn.Linear(state_dim, 64), torch.nn.Tanh(),
                             torch.nn.Linear(64, 64), torch.nn.Tanh(), 
                             torch.nn.Linear(64, action_dim))
    init_weights = (lambda param: torch.nn.init.xavier_normal_(param.weight) if
                    isinstance(param, torch.nn.Linear) else None)
    nn.apply(init_weights)
    # Switch to train other agent
    agent = TRPOAgent(policy=nn, discount=0.99, kl_delta=0.01)
    # agent = REINFORCEAgent(policy=nn, discount=0.99, optim_lr=0.01)
    # agent = LineSearchAgent(policy=nn, discount=0.99, optim_lr=0.01,
    #                        kl_delta=0.01)

    # Training
    iterations = 100
    steps_per_iter  = 16000
    episode_steps = 0
    max_length_episode = 800

    # Recording
    episode_reward = []
    reward_per_iteration = []

    ob = env.reset()
    for iteration in range(iterations):
        reward_per_iteration.append([])
        for step in range(steps_per_iter):
            # Take step with agent
            ob, rew, done, _ = env.step(agent(ob))
            agent.update_reward(rew)

            # Recording
            episode_reward.append(rew)
            episode_steps += 1

            # End of episode
            if done or episode_steps >= max_length_episode:
                agent.update_done()
                episode_steps = 0
                reward_per_iteration[-1].append(sum(episode_reward))
                episode_reward = []
                ob = env.reset()

        reward_per_iteration[-1] = (sum(reward_per_iteration[-1]) /
                                    len(reward_per_iteration[-1]))
        print(f'Iteration {iteration}: ', round(reward_per_iteration[-1], 3))
        agent.optimize()

    ob = env.reset()
    while True:
        with torch.no_grad():
            ob, rew, done, _ = env.step(agent(ob))
            agent.update(rew, done)
            env.render()
            if done:
                ob = env.reset()

    plt.plot(reward_per_iteration)
    plt.title(f'TRPO reward on Driving-v2')
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.save('trpo_reward.png')


if __name__ == '__main__':
    main()
