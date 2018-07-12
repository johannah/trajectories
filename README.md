# Planning with a conditional generative model 

In this work, we learn a dynamics model of an agent-independent environment and use this model for forward planning with MCTS. 

Please refer to our [paper](https://github.com/johannah/trajectories/blob/master/icml18-vqvae-model-camera-ready.pdf) presented at the PGMRL Workshop at ICML 2018 for implementation details.

To train the environment model, complete the following steps: 

1) Generate a training set in pixel-space of the environment: 
[road.py](https://github.com/johannah/trajectories/blob/master/trajectories/road.py).

2) Train VQ-VAE to learn a discrete latent representation of individual frames:
[train_vqvae.py](https://github.com/johannah/trajectories/blob/master/trajectories/train_vqvae.py).

3) Train a PixelCNN on the latent space:
[train_pixel_cnn.py](https://github.com/johannah/trajectories/blob/master/trajectories/train_pixel_cnn.py).

4) Run MCTS using the learned model: 
[roadway_model.py](https://github.com/johannah/trajectories/blob/master/examples/roadway_model.py).

--- 

### Example MCTS agent using our forward model for planning 

This demonstrates a MCTS agent using our VQ-VAE+PCNN model with 5 samples for planning 10 time steps ahead. In this example, the agent is twice the speed of the moving goal.

This image depicts the true observed state at each time step with the corresponding action.  Obstacles are denoted in cyan, agent in green, and the goal in yellow.

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-fast.gif)


The below image shows the same episode with the future model depicted. The first column is the observed state, the second column is the oracle rollout for human reference and the third column is the model rollout that the agent used. 
The fourth column describes model error where red pixels are false negatives (predicted free space where there is an obstacle) and blue pixels indicate false positives (predicted obstacle where there was free space). In the error plot, the predicted goal is plotted in orange over the true yellow goal.

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-rollout.gif)

More examples can be found at [https://imgur.com/a/6DJbrB1](https://imgur.com/a/6DJbrB1)

---
# Below we demonstrate reconstruction error 

Below is a demonstration of the reconstruction from our VQ-VAE model:

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vqvae.gif)

Below is a demonstration of the reconstruction from our VAE model:

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vae.gif)

---
# Acknowledgements

We based our VQ-VAE implementation on the excellent code from [@Ritesh Kumar](https://github.com/ritheshkumar95/vq-vae-exps). 
The implementation of discretized logistic mixture loss we use is from [@Lucas Caccia](https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py).

Thanks to [@kastnerkyle](https://github.com/kastnerkyle) for discussions and advice on all things.

