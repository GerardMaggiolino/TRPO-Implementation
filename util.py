'''
Contains generic code for use in RL algorithms.
'''
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal as norm_dist

class Network(nn.Module):
    '''
    Basic NN network.

    layer_unit is list of int, output of each layer.
    len(layer_unit) must be >= 1. 
    For linear layer, layer_units just contains the output units.

    '''
    def __init__(self, in_shape, layer_units, critic=False):
        super().__init__()
        self.hidden_act = nn.ReLU()

        layers = []
        for out_shape in layer_units[:-1]: 
            layers.append(nn.Linear(in_shape, out_shape))
            layers.append(self.hidden_act)
            in_shape = out_shape
        layers.append(nn.Linear(in_shape, layer_units[-1]))

        self.layers = nn.ModuleList(layers)

        for layer in self.layers: 
            if isinstance(layer, nn.Linear):
                if critic: 
                    nn.init.normal_(layer.weight, std=1e-2)
                    nn.init.normal_(layer.bias, std=1e-3)
                else: 
                    nn.init.xavier_normal_(layer.weight,
                        gain=nn.init.calculate_gain('tanh'))
        

    def forward(self, state): 
        for layer in self.layers: 
            state = layer(state)
        return state


def classify_continuous(means, stds): 
    '''
    Takes in means and std of gaussian distribution.

    Returns action, grad log prob 
    '''
    dist = norm_dist(means, torch.exp(stds))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.cpu().numpy(), log_prob


def classify_discrete(logits, std=None): 
    '''
    Takes in logits for softmax discrete policy.

    Returns action, grad log prob
    '''
    probs = nn.functional.softmax(logits, dim=0)
    action = torch.multinomial(probs, 1).cpu().item()
    log_prob = probs[action].log()
    return action, log_prob


def graph_reward(ep_reward):
    '''
    Graph info of ep_reward
    '''
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(ep_reward)
    ran = (max(ep_reward) - min(ep_reward)) * 0.1
    ax.set_ylim((min(ep_reward) - ran, max(ep_reward) + ran))
    ax.set_title('Reward per episode.')
    plt.show()


def z_score_rewards(rewards):
    '''
    Returns np array of rewards, z-scored.
    '''
    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()
    return rewards


def shuffle_together(seq1, seq2):
    state = np.random.get_state()
    np.random.shuffle(seq1)
    np.random.set_state(state)
    np.random.shuffle(seq2)


def batch(iterable, batch_size): 
    l = len(iterable)
    for b_ind in range(0, l, batch_size):
        yield iterable[b_ind : b_ind + batch_size]


class SGDOptim: 
    '''
    Wrapper for SGD optimizer with optional clipping and lr scheduler.
    '''
    def __init__(self, model, optim, scheduler=None, clip=None): 
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.clip = clip

    def step(self, val_loss=None): 
        if self.scheduler is not None: 
            self.scheduler.step(val_loss)
        if self.clip is not None: 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                self.clip)
        self.optim.step()
    
    def state_dict(self): 
        return self.optim.state_dict(), self.scheduler.state_dict()

    def load_state_dict(self, state): 
        self.optim.load_state_dict(state[0])
        self.scheduler.load_state_dict(state[1])

    def zero_grad(self):
        self.optim.zero_grad()
