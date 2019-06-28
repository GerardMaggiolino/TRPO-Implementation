import torch
import gym
from trpoagent import TRPOAgent


def main():
    torch.manual_seed(1)
    # Agent
    nn = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(),
                             torch.nn.Linear(32, 1))
    init_weights = (lambda param: torch.nn.init.xavier_normal_(param.weight) if
                    isinstance(param, torch.nn.Linear) else None)
    nn.apply(init_weights)

    # Initialize TRPOAgent
    agent = TRPOAgent(policy=nn)
    env = gym.make('MountainCarContinuous-v0')
    ob = env.reset()

    # Train
    agent.train('MountainCarContinuous-v0', seed=1, batch_size=5000,
                iterations=15, max_episode_length=1000, verbose=True)

    while True:
        ob, _, done, _ = env.step(agent(ob))
        env.render()
        if done:
            ob = env.reset()


if __name__ == '__main__':
    main()
