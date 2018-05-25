# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

import matplotlib.pyplot as plt
from imageio import imwrite
from gym_trajectories.envs.road import RoadEnv, max_pixel, min_pixel
import time
import numpy as np
from IPython import embed
from copy import deepcopy
import logging
import os
import sys
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from gym_trajectories.envs.vqvae import AutoEncoder, to_scalar
from gym_trajectories.envs.pixel_cnn import GatedPixelCNN

from gym_trajectories.envs.utils import discretized_mix_logistic_loss, get_cuts
from gym_trajectories.envs.utils import sample_from_discretized_mix_logistic


def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_vqvae_pcnn_model(state_index, cond_states, rollout_length):
    print("starting vqvae pcnn for %s predictions" %rollout_length)
    # normalize data before putting into vqvae
    st = time.time()
    broad_states = ((cond_states-min_pixel)/float(max_pixel-min_pixel) ).astype(np.float32)[:,None] 
    # transofrms HxWxC in range 0,255 to CxHxW and range 0.0 to 1.0
    nroad_states = Variable(torch.FloatTensor(broad_states))
    x_d, z_e_x, z_q_x, cond_latents = vmodel(nroad_states)

    latent_shape = (6,6)
    _, ys, xs = cond_states.shape
    proad_states = np.zeros((rollout_length,ys,xs)) 
    rollout_length = proad_states.shape[0]
    est = time.time()
    print("condition prep time", round(est-st,2))
    for ind in range(rollout_length):
        pst = time.time()
        print("predicting latent: %s/%s for state index %s" %(ind, rollout_length, state_index))
        # predict next
        pred_latents = pcnn_model.generate(spatial_cond=cond_latents[None], shape=latent_shape, batch_size=1)

        # add this predicted one to the tail
        cond_latents = torch.cat((cond_latents[1:],pred_latents))

        if not ind:
            all_pred_latents = pred_latents
        else:
            all_pred_latents = torch.cat((all_pred_latents, pred_latents))

        ped = time.time()
        print("latent pred time", round(ped-pst, 2))

    print("starting image")
    ist = time.time()
    # generate road
    z_q_x = vmodel.embedding(all_pred_latents.view(all_pred_latents.size(0),-1))
    z_q_x = z_q_x.view(all_pred_latents.shape[0],6,6,-1).permute(0,3,1,2)
    x_d = vmodel.decoder(z_q_x)

    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    proad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
    iet = time.time()
    print("image pred time", round(iet-ist, 2))
    #proad_states = np.vstack((proad_states,pred[None]))
    ## input x is between 0 and 1
    #f, ax = plt.subplots(1,3, figsize=(10,3))
    #real = road_states[frame_num]
    #ax[0].imshow(real, vmin=0, vmax=max_pixel)
    #ax[0].set_title("original frame %s"%frame_num)
    #ax[1].imshow(pred, vmin=0, vmax=max_pixel)
    #ax[1].set_title("pred")
    #ax[2].imshow((pred-real)**2, cmap='gray')
    #ax[2].set_title("error")
    #f.tight_layout()
    #plt.savefig('imgs/frame%04d'%frame_num)
    #plt.close()
    return proad_states.astype(np.int)[:,0]

class TestRollouts():
    def __init__(self, env, rollout_length, estimator=get_vqvae_pcnn_model, history_size=4):
        self.env = env
        self.history_size = history_size
        self.rollout_length = rollout_length
        self.estimator = estimator
        self.road_map_ests = np.zeros_like(self.env.road_maps)
        self.road_map_ests[:self.history_size] = self.env.road_maps[:self.history_size]

    def get_false_neg_counts(self, state_index):
        true_road_map = self.env.road_maps[state_index]
        pred_road_map = self.road_map_ests[state_index]
        # predict free where there was car  # bad
        false_neg = (true_road_map*np.abs(true_road_map-pred_road_map))
        false_neg[false_neg>0] = 1
        false_neg_count = false_neg.sum()
        return false_neg_count, false_neg
 
    def estimate_the_future(self, current_road_map, state_index):
        #######################
        # determine how different the predicted was from true for the last state
        # pred_road should be all zero if no cars have been predicted
        assert state_index>= self.history_size
        last_false_neg_counts,last_false_neg = self.get_false_neg_counts(state_index)
    
        print("running all rollouts")
        # ending index is noninclusive
        # starting index is inclusive
        est_from = state_index+1
        pred_length = self.rollout_length
    
        # put in the true road map for this step
        self.road_map_ests[state_index] = current_road_map
        s = range(self.road_map_ests.shape[0])
        # limit prediction lengths
        est_from = min(self.road_map_ests.shape[0], est_from)
        est_to = est_from + pred_length
        est_to = min(self.road_map_ests.shape[0], est_to)
        cond_to = est_from
        # end of conditioning
        cond_from = cond_to-self.history_size
        print("state index", state_index)
        print("condition", cond_from, cond_to)
        print(s[cond_from:cond_to])
        print("estimate", est_from, est_to)
        print(s[est_from:est_to])
        rinds = range(est_from, est_to)
        if not len(rinds):
            ests = []
        else:
            # can use past frames because we add them as we go
            cond_frames = self.road_map_ests[cond_from:cond_to]
            ests = self.estimator(state_index, cond_frames, pred_length)
            self.road_map_ests[est_from:est_to] = ests
        false_negs = []
        inds = []
        for xx, i in enumerate(rinds):
            fnc,fn = self.get_false_neg_counts(i)
            false_negs.append(fnc)
            inds.append(xx)
            name = "rollout_length%02d_state%03d_frame_num%03d_step%02d.png"%(self.rollout_length,state_index,i, xx)
            f, ax = plt.subplots(1,3)
            ax[0].imshow(self.env.road_maps[i], origin='lower')
            ax[0].set_title('true frame %s' %i)
            ax[1].imshow(self.road_map_ests[i], origin='lower')
            ax[1].set_title('pred step %s' %xx)
            ax[2].imshow(fn, origin='lower')
            ax[2].set_title('false negs %s'%fnc)
            plt.savefig(name)

        print("####################################")
        print(false_negs)
        print(inds)
        print("####################################")
        #embed()

def run_test(seed=3432, ysize=48, xsize=48, level=6, 
        n_playouts=300, 
        max_rollout_length=50, estimator=get_vqvae_pcnn_model,
        history_size=4, 
        do_render=False):
    # restart at same position every time
    rdn = np.random.RandomState(seed)
    true_env = RoadEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level)

    start_state = true_env.reset(experiment_name=seed, goal_distance=30)
    # fast forward history steps so agent observes through conditioning steps
    state = [start_state[0], start_state[1], true_env.road_maps[history_size]]
    true_env.set_state(state, history_size)


    test = TestRollouts(env=deepcopy(true_env), rollout_length=max_rollout_length, history_size=history_size, estimator=estimator)

    for t in range(history_size, true_env.road_maps.shape[0], 5):
        print(t)
        test.estimate_the_future(true_env.road_maps[t], t)


if __name__ == "__main__":
    import argparse
    # this seems to work well
    #python roadway_pmcts.py --seed 45 -r 100  -p 100 -l 6

    default_base_savedir = '../../trajectories_frames/saved/vqvae'
    # train loss .0488, test_loss sum .04 epoch 25
    # false negs over 10 steps  for seed 35 [17, 21, 26, 25, 32, 40, 38, 38, 39, 41]
    #vq_name = 'vqvae4layer_base_k512_z32_dse00025.pkl'
    # train loss .034, test_loss sum .0355 epoch 51
    vq_name = 'vqvae4layer_base_k512_z32_dse00051.pkl'
    default_vqvae_model_loadpath = os.path.join(default_base_savedir, 
            vq_name)
    # train loss of .39, epoch 31
    #pcnn_name = 'rpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00031.pkl'
    # false negs over 10 steps for seed 35 
    # [11, 13, 14, 12, 19, 27, 30, 35, 38, 37] 
    # [10, 12, 13, 12, 19, 27, 30, 33, 36, 38]
    # [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # train loss of 1.09, test loss 1.25 epoch 10
    pcnn_name = 'nrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00010.pkl'
    # false negs over 10 steps for seed 35 
    # [13, 13, 16, 13, 23, 29, 31, 36, 35, 37]
    # [12, 12, 15, 16, 23, 29, 31, 33, 35, 38]
    # [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # [27, 23, 26, 27, 30, 29, 32, 32, 26, 26]
    #pcnn_name = 'nrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00026.pkl'
    default_pcnn_model_loadpath = os.path.join(default_base_savedir,
            pcnn_name)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=100, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=48, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=48, help='pixel size of game in x direction')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-r', '--rollout_steps', type=int, default=10, help='limit number of steps taken be random rollout')
    parser.add_argument('-vq', '--vqvae_model_loadpath', type=str, default=default_vqvae_model_loadpath)
    parser.add_argument('-pcnn', '--pcnn_model_loadpath', type=str, default=default_pcnn_model_loadpath)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')
    parser.add_argument('-t', '--model_type', type=str, default='vqvae_pcnn_model')

    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--do_plot_error', action='store_true', default=True)
    parser.add_argument('--plot_playouts', action='store_true', default=False)
    parser.add_argument('--plot_playout_gap', type=int, default=3, help='gap between plot playouts for each step')
    parser.add_argument('-f', '--prior_fn', type=str, default='goal', help='options are goal or equal')
    #equal_node_probs_fn(
    args = parser.parse_args()
    use_cuda = args.cuda
    seed = args.seed
    dsize = 40
    nr_logistic_mix = 10
    probs_size = (2*nr_logistic_mix)+nr_logistic_mix

    num_z = 32
    nr_logistic_mix = 10
    num_clusters = 512

    if use_cuda:
        DEVICE = 'cuda'
        print("using gpu")
    else:
        DEVICE = 'cpu'

    N_LAYERS = 15 # layers in pixelcnn
    DIM = 256
    history_size = 4
    cond_size = history_size*DIM

    if args.model_type == 'vqvae_pcnn_model':
        estimator = get_vqvae_pcnn_model
        if os.path.exists(args.vqvae_model_loadpath):
            vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=num_z).to(DEVICE)
            vqvae_model_dict = torch.load(args.vqvae_model_loadpath, map_location=lambda storage, loc: storage)
            vmodel.load_state_dict(vqvae_model_dict['state_dict'])
            epoch = vqvae_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, 
                                                   args.vqvae_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.vqvae_model_loadpath))
            sys.exit()



        if os.path.exists(args.pcnn_model_loadpath):
            pcnn_model = GatedPixelCNN(num_clusters, DIM, N_LAYERS, 
                    history_size, spatial_cond_size=cond_size).to(DEVICE)
            pcnn_model_dict = torch.load(args.pcnn_model_loadpath, map_location=lambda storage, loc: storage)
            pcnn_model.load_state_dict(pcnn_model_dict['state_dict'])
            epoch = pcnn_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, 
                                                   args.pcnn_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.pcnn_model_loadpath))
            sys.exit()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fname = 'test_model_%s_length_%s.pkl' %(
                                    args.model_type, 
                                    args.rollout_steps)


    ffile = open(fname, 'a+')
    if os.path.exists(fname):
        print('loading previous results from %s' %ffile)
        try:
            all_results = pickle.load(ffile)
            print('found %s runs in file' %len(all_results.keys())-1)
        except EOFError, e:
            print('unable to load ffile:%s' %ffile)
            all_results = {'args':args}
    else:
        all_results = {'args':args}

    for i in range(args.num_episodes):
        print("STARTING EPISODE %s seed %s" %(i,seed))
        if seed in all_results.keys():
            print("seed %s already in results - skipping" %seed)
            seed +=1
        else:
            st = time.time()
            r = run_test(seed=seed, ysize=args.ysize, xsize=args.xsize, 
                          level=args.level, estimator=estimator,
                          max_rollout_length=args.rollout_steps, 
                          history_size=history_size, do_render=args.render)

            #et = time.time()
            #r['full_end_time'] = et
            #r['full_start_time'] = st
            #all_results[seed] = r
            #pickle.dump(all_results,ffile)
            seed +=1
    embed()
    print("FINISHED")


