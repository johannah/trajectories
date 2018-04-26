# Probabilistic MCTS for navigating to a goal in a dangerous environment

We learn a model of the environment using a vqvae and perform one-step ahead rollouts using this model. 
![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vqvae.gif)

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vqvae.gif)


We also learn a model of the environment using a vae and perform one-step ahead rollouts. This result is noiser, but the latent space has meaningful relationships. 

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vae.gif)


![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vae.gif)
