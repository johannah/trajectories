# Probabilistic MCTS for navigating to a goal in a dangerous environment

We learn a model of the environment using a vqvae and perform one-step ahead rollouts using this model. 
![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vqvae.gif)

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vqvae.gif)


We also learn a model of the environment using a vae and perform one-step ahead rollouts. This result is noiser, but the latent space has meaningful relationships. 

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vae.gif)


![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vae.gif)

We based our VQ-VAE implementation on the excellent code from [@Ritesh Kumar](https://github.com/ritheshkumar95/vq-vae-exps). 

The implementation of discretized logistic mixture loss we use is from  [@Lucas Caccia](https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py).

Thanks to [@kastnerkyle](https://github.com/kastnerkyle) for discussions and advice on all things neural-networky. 

