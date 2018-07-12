# Planning with a conditional generative model 

Please refer to our [paper](https://github.com/johannah/trajectories/blob/master/icml18-vqvae-model-camera-ready.pdf) presented at the PGMRL Workshop at ICML 2018 for implementation details.

We learn a dynaics model of an agent-independent environment and use this model for forward planning with MCTS. 

To train the environment model, complete teh following steps: 

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

VQ-VAE+PCNN model with 5 samples and 10-step rollout with agent 2X speed of goal

This image depicts the true observed state at each time step with the corresponding action.  Obstacles are denoted in cyan, agent in green, and the goal in yellow.

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-fast.gif)

The above agent with MCTS rollouts in VQ-VAE+PCNN model.

The first column is observed state, second column is oracle rollout for human reference, third column is model rollout. Obstacles are denoted in cyan, agent in green, and the goal in yellow. The fourth column describes model error. Red pixels are false negatives (predicted free space where there is an obstacle) and blue pixels indicate false positives (predicted obstacle where there was free space). In the error plot, the predicted goal is plotted in orange over the true yellow goal.

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/10-step-playout.gif)

---
# Below we show some deomstrations from zero-step models with a static goal. 
Here we show a zero-step ahead rollouts using VQ-VAE. 
![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vqvae.gif)

We also learn a model of the environment using a VAE and perform zero-step ahead rollouts.  

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vae.gif)

---
# Acknowledgements

We based our VQ-VAE implementation on the excellent code from [@Ritesh Kumar](https://github.com/ritheshkumar95/vq-vae-exps). 
The implementation of discretized logistic mixture loss we use is from [@Lucas Caccia](https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py).

Thanks to [@kastnerkyle](https://github.com/kastnerkyle) for discussions and advice on all things.

