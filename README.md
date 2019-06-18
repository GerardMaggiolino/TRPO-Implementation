### TRPO and REINFORCE 
This repo contains PyTorch implementations of TRPO and REINFORCE. 

Unlike the original TRPO implementation, the fisher-vector product used in 
conjugate gradient is computed through the direct method. Additionally, all 
states for the iteration are used in the double gradient required for the
fisher-vector product, while the original uses only 10% of the states. The 
former change has no effect in accuracy, while the latter should sacrifice 
efficiency for increased accuracy. The result on efficiency is show beneath.  

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/Time.png width=480 height=360>

Computing a search direction through conjugate gradient and the line search take ~6x and ~1.5x the time of computing the REINFORCE gradient, respectively. This aligns well with the paper's findings, as they claim that performing conjugate gradient over 10% of the sampled data requires the same time as a normal gradient. 

Experiments over OpenAI Gym's LunarLanderContinuous-v2 demonstrate that this method does not produce significantly better results than using the first-order Adam optimizer with REINFORCE. 

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/TRPO%20reward.png width=480 height=360>
<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/REINFORCE.png width=480 height=360>

Both experiments used identical networks and training parameters for 50 iterations over 500,000 time steps. 
