###TRPO and REINFORCE 
This repo contains PyTorch implementations of TRPO and REINFORCE. 

Unlike the original TRPO implementation, the fisher-vector product used in 
conjugate gradient is computed through the direct method. Additionally, all 
states for the iteration are used in the double gradient required for the
fisher-vector product, while the original uses only 10% of the states. The 
former change has no effect in accuracy, while the latter should sacrifice 
efficiency for increased accuracy. The result on efficiency is show beneath.  
