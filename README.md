# Planning with a conditional generative model 

Please refer to our [paper](https://github.com/johannah/trajectories/blob/master/icml18-vqvae-model-camera-ready.pdf) presented at the PGMRL Workshop at ICML 2018.
---
# Below we show some deomstrations from zero-step models with a static goal. 
Here we show a zero-step ahead rollouts using VQ-VAE. 
![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vqvae.gif)

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vqvae.gif)

We also learn a model of the environment using a VAE and perform zero-step ahead rollouts.  

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/true_step_seed_930_vae.gif)

![alt_text](https://github.com/johannah/trajectories/blob/master/imgs/playout_step_seed_930_vae.gif)

---
# Acknowledgements

We based our VQ-VAE implementation on the excellent code from [@Ritesh Kumar](https://github.com/ritheshkumar95/vq-vae-exps). 
The implementation of discretized logistic mixture loss we use is from [@Lucas Caccia](https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py).

Thanks to [@kastnerkyle](https://github.com/kastnerkyle) for discussions and advice on all things.

