import gym
import torch
import os
from trpoagent import TRPOAgent
from linesearchagent import LineSearchAgent
from reinforceagent import REINFORCEAgent
import pickle
import matplotlib.pyplot as plt


def main():
    # Policy
    torch.manual_seed(1)
    nn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                             torch.nn.Linear(32, 4))
    init_weights = (lambda param: torch.nn.init.xavier_normal_(param.weight) if
                    isinstance(param, torch.nn.Linear) else None)
    nn.apply(init_weights)

    # Initialize the agent
    # agent = TRPOAgent(policy=nn)
    # agent = REINFORCEAgent(policy=nn)
    agent = LineSearchAgent(policy=nn)
    # Uncomment to load existing policy
    # if os.path.exists(f"{policy_name}.pth"):
    #     agent.load_model(f"{policy_name}.pth")
    policy_name = type(agent).__name__

    # Train
    records = agent.train('LunarLanderContinuous-v2', seed=1, batch_size=10000,
                          iterations=50, max_episode_length=500, verbose=True)
    agent.save_model(f'{policy_name}.pth')
    pickle.dump(records, open(f'{policy_name}_records.pickle', 'wb'))
    plot_info(records, policy_name)

    # Display
    env = gym.make('LunarLanderContinuous-v2')
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()


def plot_info(records, policy_name):
    rewards_per_iter = []
    s = e = 0
    for i in range(5, len(records['num_episodes_in_iteration'])):
        num = records['num_episodes_in_iteration'][i]
        e += records['num_episodes_in_iteration'][i]
        avg = sum(records['episode_reward'][s:e]) / num
        rewards_per_iter.append(avg)
        s = e
    print(max(rewards_per_iter))
    max_iter = max(enumerate(rewards_per_iter), key=lambda t: t[1])[0]
    plt.plot(list(range(5, len(rewards_per_iter) + 5)), rewards_per_iter)
    plt.text(min(max_iter, len(rewards_per_iter) - 10),
             rewards_per_iter[max_iter],
             "Max Reward: " + str(round(rewards_per_iter[max_iter], 2)))

    plt.title(f"{policy_name} reward per iter on LunarLanderContinuous-v2")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.show()


if __name__ == '__main__':
    main()
