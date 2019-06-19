import gym
import torch
import numpy.random
import matplotlib.pyplot as plt
from trpoagent import TRPOAgent
from reinforceagent import REINFORCEAgent
from linesearchagent import LineSearchAgent


def main():
    env = gym.make('LunarLanderContinuous-v2')
    torch.manual_seed(1)
    env.seed(1)
    numpy.random.seed(1)

    # Agent
    nn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                             torch.nn.Linear(32, 2))
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
    steps_per_iter = 10000
    episode_steps = 0
    max_length_episode = 500

    # Recording
    episode_reward = []
    reward_per_iteration = []

    ob = env.reset()
    for iteration in range(iterations):
        reward_per_iteration.append([])
        for step in range(steps_per_iter):
            # Take step with agent
            ob, rew, done, _ = env.step(agent(ob))
            agent.update(rew, done)

            # Recording
            episode_reward.append(rew)
            episode_steps += 1

            # End of episode
            if done or episode_steps >= max_length_episode:
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
    plt.title(f'TRPO reward on LunarLanderContinuous-v2')
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    #plt.show()


if __name__ == '__main__':
    main()
