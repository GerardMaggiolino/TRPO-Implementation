### TRPO and REINFORCE 
This repo contains PyTorch implementations of TRPO and REINFORCE. 

Unlike the original TRPO implementation, the fisher-vector product used in 
conjugate gradient is computed through the direct method. Additionally, all 
states for the iteration are used in the double gradient required for the
fisher-vector product, while the original uses only 10% of the states. The 
former change has no effect in accuracy, while the latter should sacrifice 
efficiency for increased accuracy. The result on efficiency is show beneath. 
A learnable standard deviation separate from the neural network which parameterizes
the mean of the gaussian policy is used. 

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/Time.png width=480 height=360>

Computing a search direction through conjugate gradient and the line search take ~6x and ~1.5x the time of computing the REINFORCE gradient, respectively. This aligns well with the paper's findings, as they claim that performing conjugate gradient over 10% of the sampled data requires the same time as a normal gradient. 

Experiments over OpenAI Gym's LunarLanderContinuous-v2 demonstrate that this method does not produce significantly better results than using the first-order Adam optimizer with REINFORCE. Both experiments used identical networks and training parameters for 50 iterations over 500,000 time steps. In this problem, moving towards a goal yields reward; however, crashing with the goal incurs a large penalty. As agents learn to navigate towards the goal, they also collide more frequently, as seen in the dip in reward midway through training. 


<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/TRPO%20reward.png width=480 height=360>

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/REINFORCE%20reward.png width=480 height=360>

A third agent using REINFORCE with Adam optimization also utilizes a line search. The line search enforces constraints of the original TRPO objective and KL divergence, gradually reducing the gradients applied by Adam until constraints are met. This achieves marginally better performance than the standard REINFORCE algorithm, with the line search only requiring ~1/25 the time to execute compared to gradient calculations. The initial scaling to the step size, which requires a fisher-vector product to compute, is replaced with a constant. 

<img src=https://github.com/GerardMaggiolino/TRPO-Implementation/blob/master/images/LineSearchAgent%20reward.png width=480 height=360>

As such, a line search with KL divergence and TRPO surrogate objective constraints and is an efficient method to prevent damaging updates. 

