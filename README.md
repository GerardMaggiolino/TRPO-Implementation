### TRPO, REINFORCE, and REINFORCE LineSearch

This repo contains PyTorch implementations of TRPO, REINFORCE, and REINFORCE 
with a KL divergence line search constrained by the TRPO surrogate objective. 

#### Usage

All implementations have a common interface, demonstrated below. TRPOAgent 
can be simply replaced with LineSearchAgent or REINFORCEAgent. 

```Python3
import gym
import torch
from trpoagent import TRPOAgent

nn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                         torch.nn.Linear(32, 4))
agent = TRPOAgent(policy=nn)
agent.train('LunarLanderContinuous-v2', seed=1, batch_size=10000,
             iterations=50, max_episode_length=500, verbose=True)
agent.save_model("agent.pth")
agent.load_model("agent.pth")

env = gym.make('LunarLanderContinuous-v2')
ob = env.reset()
while True:
    action = agent(ob)
    ob, _, done, _ = env.step(action)
    env.render()
    if done:
        ob = env.reset()
```

#### Evaluation 

Unlike the original TRPO implementation, the fisher-vector product used in 
conjugate gradient is computed through the direct method. Additionally, all 
states for the iteration are used in the double gradient required for the
fisher-vector product, while the original uses only 10% of the states. The 
former change has no effect in accuracy, while the latter should sacrifice 
efficiency for increased accuracy. The result on efficiency is show beneath. 
A learnable standard deviation separate from the neural network which parameterizes
the mean of the gaussian policy is used. 

The LineSearchAgent is identical to REINFORCE, but linearly reduces parameter 
updates until the KL constraint is satisfied. 

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/Time.png width=480 height=360>

Computing a search direction through conjugate gradient and the line search take 
~6x and ~1.5x the time of computing only the REINFORCE gradient, respectively. 
This aligns well with the paper's findings, as they claim that performing conjugate 
gradient over only 10% of the sampled data requires the same time as a normal gradient. 

Experiments are run over [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/).
All agents are run for 50 iterations of 10,000 steps per iteration with the same 
NN policy show in train.py. The average reward for all completed episodes in an 
iteration is plotted beneath.

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/TRPO%20reward.png width=480 height=360>

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/REINFORCE%20reward.png width=480 height=360>

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/LineSearchAgent%20reward.png width=480 height=360>

TRPO obtains the highest score after 500,000 simulation steps. 

Constraining the REINFORCE update based on KL divergence of the realized 
policy from the network results in a lower score; it should be noted that the 
implementation ONLY reduces the gradient based on KL divergence, and does not
increase the gradient underneath the threshold. Thus, slower convergence in the 
same number of iterations is expected. 

