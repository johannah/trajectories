# Catching a moving goal in a dangerous world 
### Planning with anticipation using a conditional generative model of the environment

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-fast.gif)

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-rollout.gif)

This gif shows our green MCTS agent trying to catch the yellow goal while avoiding the cyan/blue obstacles. We learned a dynamics model of the agent-independent environment and use this imagined future to select actions at every state. 

The below image shows the same episode as above with the imagined future model depicted. The first column is the observed state, the second column is the oracle rollout (for human reference only) and the third column is the model rollout that the agent used for planning. 
The fourth column describes model error where red pixels are false negatives (predicted free space where there is an obstacle) and blue pixels indicate false positives (predicted obstacle where there was free space). In the error plot, the predicted goal is plotted in orange over the true yellow goal.

More agent examples can be found at [https://imgur.com/a/6DJbrB1](https://imgur.com/a/6DJbrB1)

Please refer to our [paper](https://github.com/johannah/trajectories/blob/master/icml18-vqvae-model-camera-ready.pdf) presented at the PGMRL Workshop at ICML 2018 for implementation details.

[README](https://github.com/johannah/trajectories/blob/master/README.md)

