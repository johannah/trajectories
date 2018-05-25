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

def perfect_policy_fn(state, valid_actions):
    action_probs = np.zeros_like(valid_actions).astype(np.float)
    action_probs[state] = 1.0
    comb = tuple(zip(valid_actions, action_probs))
    return comb, 0

class PTreeNode(object):
    def __init__(self, parent, prior_prob, name='unk'):
        self.name = name
        self.parent = parent
        self.Q_ = 0.0
        self.P_ = float(prior_prob)
        # action -> tree node
        self.children_ = {}
        self.n_visits_ = 0
        self.past_actions = []
        self.n_wins = 0

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                child_name = (self.name[0],action)
                self.children_[action] = PTreeNode(self, prior_prob=prob, name=child_name)

    def is_leaf(self):
        return self.children_ == {}

    def is_root(self):
        return self.parent is None

    def _update(self, value):
        self.n_visits_ += 1
        self.Q_ += (value-self.Q_)/float(self.n_visits_)

    def update(self, value):
        if self.parent != None:
            self.parent.update(value)
        self._update(value)

    def get_value(self, c_puct):
        self.U_ = c_puct * self.P_ * np.sqrt(float(self.parent.n_visits_)) / float(1+self.n_visits_)
        return self.Q_ + self.U_

    def get_best(self, c_puct):
        best = max(self.children_.iteritems(), key=lambda x: x[1].get_value(c_puct))
        return best

def goal_node_probs_fn(state, state_index, env):
    # TODO make env take in state to give goal/robot
    bearing = env.get_goal_bearing(state)
    action_distances = np.abs(env.angles-bearing)
    actions_and_distances = list(zip(env.action_space, action_distances))
    best_angles = sorted(actions_and_distances, key=lambda tup: tup[1])
    best_actions = [b[0] for b in best_angles]
    best_angles = np.ones(len(env.action_space))
    best_angles[len(env.action_space)/2:len(env.action_space)/3] = 2
    best_angles[:len(env.action_space)/2] = 2.4
    best_angles[0] = 2.5
    best_angles = best_angles/float(best_angles.sum())

    unsorted_actions_and_probs = list(zip(best_actions, best_angles))
    actions_and_probs = sorted(unsorted_actions_and_probs, key=lambda tup: tup[0])

    return actions_and_probs

def equal_node_probs_fn(state, state_index, env):
    probs = np.ones(len(env.action_space))/float(len(env.action_space))
    actions_and_probs = list(zip(env.action_space, probs))
    return actions_and_probs

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
        frame_num = ind+state_index
        pst = time.time()
        print("predicting latent: %s" %frame_num)
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

class PMCTS(object):
    def __init__(self, env, random_state, node_probs_fn, c_puct=1.4,
            n_playouts=1000, rollout_length=20, estimator=get_vqvae_pcnn_model,
            history_size=4):
        # use estimator for planning, if false, use env
        self.env = env
        self.rdn = random_state
        self.node_probs_fn = node_probs_fn
        self.root = PTreeNode(None, prior_prob=1.0, name=(0,-1))
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.tree_subs_ = []
        self.warn_at_tree_size = 1000
        self.tree_subs_ = []
        self.step = 0
        self.rollout_length = rollout_length
        self.nodes_seen = {}
        self.estimator = eval('get_'+estimator) # get_vqvae_pcnn_model
        self.road_map_ests = np.zeros_like(self.env.road_maps)
        self.history_size = history_size
        # infil the first road maps
        if self.estimator == 'none':
            self.road_map_ests = self.env.road_maps
        else:
            self.road_map_ests[:self.history_size] = self.env.road_maps[:self.history_size]
        # debuging none

    def get_children(self, node):
        print('node name', node.name)
        for i in node.children_.keys():
            print(node.children_[i].__dict__)
        return [node.children_[i].__dict__ for i in node.children_.keys()]

    def playout(self, playout_num, state, state_index):
        # set new root of MCTS (we've taken a step in the real game)
        # only sets robot and goal states
        # get future playouts from past states 

        logging.debug('+++++++++++++START PLAYOUT NUM: {} FOR STATE: {}++++++++++++++'.format(playout_num,state_index))
        init_state = state
        init_state_index = state_index
        node = self.root
        won = False
        frames = []
        # stack true state then vstate
        vstate = [state[0], state[1], self.road_map_ests[state_index]]
        frames.append((self.env.get_state_plot(state), self.env.get_state_plot(vstate)))
        # always use true state for first
        finished,value = self.env.set_state(state, state_index)

       
        while True:
            ry,rx = self.env.get_robot_state(state)
            self.playout_states[state_index,ry,rx] = self.env.robot.color
            if node.is_leaf():
                if not finished:
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action
                    actions_and_probs = self.node_probs_fn(state, state_index, self.env)
                    node.expand(actions_and_probs)
                    # if you have a neural network - use it here to bootstrap the value
                    # otherwise, playout till the end
                    # rollout one randomly
                    value, rframes = self.rollout_from_state(state, state_index)
                    frames.extend(rframes)
                    finished = True
                # finished the rollout
                node.update(value)
                if value > 0:
                    node.n_wins+=1
                    won = True
                return won, frames, value
            else:
                # greedy select
                # trys actions based on c_puct
                action, new_node = node.get_best(self.c_puct)
                next_state, value, finished, _ = self.env.step(state, state_index, action)
                # time step
                state_index +=1
                # gets true step back
                next_vstate = [next_state[0], next_state[1], self.road_map_ests[state_index]]
                # stack true state then vstate
                frames.append((self.env.get_state_plot(next_state), self.env.get_state_plot(next_vstate)))
                node = new_node
                state = next_vstate

    def get_rollout_action(self, state):
        valid_actions = self.env.action_space
        action_probs = self.rdn.rand(len(valid_actions))
        action_probs = action_probs / np.sum(action_probs)
        act_probs = tuple(zip(valid_actions, action_probs))
        acts, probs = zip(*act_probs)
        act = self.rdn.choice(acts, p=probs)
        return act, act_probs

    def rollout_from_state(self, state, state_index):
        logging.debug('-------------------------------------------')
        logging.debug('starting random rollout from state: {}'.format(state_index))

        # comes in already transformed
        rframes = []
        #try:
        if 1:
            finished,value = self.env.set_state(state, state_index)
            if finished:
                return value, rframes
            c = 0
            while not finished:
                if c < self.rollout_length:
                    a, action_probs = self.get_rollout_action(state)
                    self.env.set_state(state, state_index)
                    ry,rx = self.env.get_robot_state(state)
                    self.playout_states[state_index,ry,rx] = self.env.robot.color
                    next_state, reward, finished,_ = self.env.step(state, state_index, a)

                    state_index+=1
                    #next_vstate = [next_state[0], next_state[1], get_vq_from_road(next_state[2])]
                    next_vstate = [next_state[0], next_state[1], self.road_map_ests[state_index]]
                    # stack true state then vstate
                    rframes.append((self.env.get_state_plot(next_state), self.env.get_state_plot(next_vstate)))
                    state = next_vstate

                    # true and vq state
                    c+=1
                    if finished:
                        logging.debug('finished rollout after {} steps with value {}'.format(c,value))
                        value = reward
                else:
                    # stop early
                    value = self.env.get_timeout_reward(c)
                    logging.debug('stopping rollout after {} steps with value {}'.format(c,value))
                    finished = True

            logging.debug('-------------------------------------------')
        #except Exception, e:
        #    print(e)
        #    embed()
        return value, rframes


    def get_action_probs(self, state, state_index, temp=1e-2):
        print("-----------------------------------")
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0
        logging.debug("starting playouts for state_index %s" %state_index)
        # only run last rollout
        finished,value = self.env.set_state(state, state_index)
        all_frames = {}
        self.playout_states = np.zeros((self.env.max_steps, self.env.ysize, self.env.xsize))
        if not finished:
            for n in range(self.n_playouts):
                from_state = deepcopy(state)
                from_state_index = deepcopy(state_index)
                w, fs, v = self.playout(n, from_state, from_state_index)
                if len(all_frames.keys())>2:
                    vs = [k[1] for k in all_frames.keys()]
                    if v > min(vs):
                        varg = np.argmin(vs)
                        bad = all_frames.keys()[varg]
                        del all_frames[bad]
                        all_frames[(n,v)] = fs
                        #print('adding', (n, v))
                        #print('deleting', bad)
                        #print(all_frames.keys())
                else:
                    all_frames[(n,v)] = fs
                won+=w
        else:
            logging.info("GIVEN STATE WHICH WILL DIE - state index {} max env {}".format(state_index, self.env.max_steps))
        self.env.set_state(state, state_index)
        act_visits = [(act,float(node.n_visits_)) for act, node in self.root.children_.items()]
        try:
            actions, visits = zip(*act_visits)
        except Exception, e:
            print("ACTIONS VISITS")
            print(e)
            embed()

        action_probs = softmax(1.0/temp*np.log(visits))
        return actions, action_probs, all_frames


    def sample_action(self, state, state_index, temp=1E-3, add_noise=True,
                      dirichlet_coeff1=0.25, dirichlet_coeff2=0.3):
        vsz = len(self.state_manager.get_action_space())
        act_probs = np.zeros((vsz,))
        acts, probs = self.get_action_probs(state, temp)
        act_probs[list(acts)] = probs
        if add_noise:
            act = self.random_state.choice(acts, p=(1. - dirichlet_coeff1) * probs + dirichlet_coeff1 * self.random_state.dirichlet(dirichlet_coeff2 * np.ones(len(probs))))
        else:
            act = self.random_state.choice(acts, p=probs)
        return act, act_probs

    def estimate_the_future(self, state, state_index):
        #######################
        # determine how different the predicted was from true for the last state
        # pred_road should be all zero if no cars have been predicted
        true_road_map = self.env.road_maps[state_index]
        pred_road_map = self.road_map_ests[state_index]
        # predict free where there was car  # bad
        false_neg = (true_road_map*np.abs(true_road_map-pred_road_map))
        false_neg[false_neg>0] = 1
        false_neg_count = false_neg.sum()
        # get local box
        ry,rx = self.env.get_robot_state(state)
        ys = self.env.ysize-1
        xs = self.env.xsize-1
        bs = 5
        lby = min(ry+bs, ys)
        lbx = min(rx+bs, xs)
        iby = max(ry-bs, 0)
        ibx = max(rx-bs, 0)
        print("box inds")
        print(iby, ry, lby)
        print(ibx, rx, lbx)
        
        local_false_neg_count = np.sum(false_neg[iby:lby, ibx:lbx])
 
        print('false neg is', false_neg_count)
        # false_neg_count is ~ 25 when the pcnn predicts all zeros
        print('local false neg is', local_false_neg_count)

        #f,a = plt.subplots(1,4)
        #a[0].imshow(true_road_map); a[0].set_title('true'); 
        #a[1].imshow(pred_road_map);  a[1].set_title('prev pred')
        #a[2].imshow(false_neg); a[3].imshow(np.abs(true_road_map-pred_road_map));  
        #plt.show()
        #embed()

        if (local_false_neg_count > 0) or (false_neg_count > 15) :
            print("running all rollouts")
            # ending index is noninclusive
            # starting index is inclusive
            est_from = state_index+1
            pred_length = self.rollout_length
        else:
            print("running one rollout")
            # only run last rollout that was not finished
            est_from = state_index+self.rollout_length+1
            pred_length = 1


        # put in the true road map for this step
        self.road_map_ests[state_index] = true_road_map
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
        return ests, rinds


    def get_best_action(self, state, state_index):
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        self.env.set_state(state, state_index)
        if self.estimator == 'none':
            state_ests,state_indexes = [], []
        else:
            state_ests, state_indexes = self.estimate_the_future(state, state_index)

        acts, probs, playout_frames = self.get_action_probs(state, state_index, temp=1e-3)
        act = self.rdn.choice(acts, p=probs)
        logging.info("mcts chose action {} in state: {}".format(act,state_index))
        return act, probs, playout_frames, self.playout_states, state_ests, state_indexes

    def update_tree_move(self, action):
        # keep previous info
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_tree_size:
                logging.warn("WARNING: over {} tree_subs_ detected".format(len(self.tree_subs_)))
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            logging.error("Move argument {} to update_to_move not in actions {}, resetting".format(action, self.root.children_.keys()))

    def reset_tree(self):
        logging.warn("Resetting tree")
        self.root = PTreeNode(None, prior_prob=1.0, name=(0,-1))
        self.tree_subs_ = []

def plot_playout_scatters(true_env, base_path, model_type, seed, reward, sframes,
                         model_road_maps, rollout_length,
                         t,plot_error=False,gap=3,min_agents_alive=4):
    true_road_maps = true_env.road_maps
    true_goal_map = true_env.goal_map
    if plot_error:
        fpath = os.path.join(base_path,model_type,'Eseed_{}'.format(seed))
    else:
        fpath = os.path.join(base_path,model_type,'Pseed_{}'.format(seed))

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fast_path = os.path.join(base_path,model_type,'Tseed_{}'.format(seed))
    if not os.path.exists(fast_path):
        os.makedirs(fast_path)
    for ts in range(len(sframes)):
        print("plotting true frame {}/{}".format(ts,t))
        # true frame
        state_index = sframes[ts][0]
        ry,rx  = sframes[ts][1]
        ry = ry/float(true_env.ysize)
        rx = rx/float(true_env.xsize)
        gy = true_env.goal.y/float(true_env.ysize)
        gx = true_env.goal.x/float(true_env.xsize)
        true_state = true_env.road_maps[state_index]
        state = ((gy, gx), (ry,rx), true_state)
        actual_frame = true_env.get_state_plot(state)

        model_state = model_road_maps[state_index]
        vstate = ((gy, gx), (ry,rx), model_state)
        model_frame = true_env.get_state_plot(vstate)

        err = np.zeros_like(true_road_maps[0])
        true_car = true_road_maps[state_index]>0
        pred_car = model_road_maps[state_index]>0
        true_free = true_road_maps[state_index]<1
        pred_free = model_road_maps[state_index]<1
        # predict car where there was free space
        false_pos = true_free.astype(np.int)+pred_car.astype(np.int)
        err[false_pos>1] = 15 # false pos

        # predict free where there was car  # bad
        false_neg = true_car.astype(np.int)+pred_free.astype(np.int)
        err[false_neg>1] = 250 # false neg


        fast_fname = 'fast_seed_%06d_step_%04d.png'%(seed, ts)
        ft,axt=plt.subplots(1,3, figsize=(9,3))
        axt[0].imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
        axt[0].set_title("true step:{}/{}".format(ts,t))
        axt[1].imshow(model_frame, origin='lower', vmin=0, vmax=255 )
        axt[1].set_title("{} model step:{}/{}".format(model_type, ts,t))
        axt[2].imshow(err, origin='lower', cmap=plt.cm.gray )
        axt[2].set_title("model error".format(model_type))
        ft.tight_layout()
        plt.savefig(os.path.join(fast_path,fast_fname))
        plt.close()

        # list of tuples with (real playout state, est playout state)

        playouts = sframes[ts][2]
        c = 0
        for pn, pframe in enumerate(playouts):
                if not c%gap:
                    if c > 10:
                        if pframe.sum()<(min_agents_alive*true_env.robot.color-1):
                            continue

                    print("plotting step {}/{} playout step {}".format(ts,t,pn))
                    true_playout_frame = true_road_maps[pn]+pframe+true_goal_map
                    est_playout_frame = model_road_maps[pn]+pframe+true_goal_map

                    fname = 'seed_%06d_tstep_%04d_pstep_%04d.png'%(seed, ts, pn)
                    if plot_error:
                        f,ax=plt.subplots(1,4, figsize=(16,3.5))
                    else:
                        f,ax=plt.subplots(1,3, figsize=(12,3.5))

                    ax[0].imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
                    ax[0].set_title("true state step:{}/{}".format(ts,t))
                    ax[1].imshow(true_playout_frame, origin='lower', vmin=0, vmax=255 )
                    ax[1].set_title("true rollout step:{}/{}".format(pn,rollout_length))
                    ax[2].imshow(est_playout_frame, origin='lower', vmin=0, vmax=255 )
                    ax[2].set_title("{} model rollout step:{}/{}".format(model_type,pn,rollout_length))
                    if plot_error:
                        err = np.zeros_like(true_road_maps[0])
                        true_car = true_road_maps[pn]>0
                        pred_car = model_road_maps[pn]>0
                        true_free = true_road_maps[pn]<1
                        pred_free = model_road_maps[pn]<1
                        # predict car where there was free space
                        false_pos = true_free.astype(np.int)+pred_car.astype(np.int)
                        err[false_pos>1] = 15 # false pos

                        # predict free where there was car  # bad
                        false_neg = true_car.astype(np.int)+pred_free.astype(np.int)
                        err[false_neg>1] = 250 # false neg

                        ax[3].imshow(err, origin='lower', cmap=plt.cm.gray)
                        ax[3].set_title("error in model:{}/{}".format(pn,rollout_length))
                    f.tight_layout()
                    plt.savefig(os.path.join(fpath,fname))
                    plt.close()
                c+=1
    print("making gif")
    gif_path = os.path.join(fpath, 'seed_{}_reward_{}_gap_{}.gif'.format(seed, int(reward),gap))
    search = os.path.join(fpath, 'seed_*.png')
    cmd = 'convert -delay 1/100000 %s %s'%(search, gif_path)
    #os.system(cmd)

    fast_gif_path = os.path.join(fast_path, 'fast_seed_{}_reward_{}.gif'.format(seed, int(reward),gap))
    fsearch = os.path.join(fast_path, '*.png')
    cmd = 'convert -delay 1/30 %s %s'%(fsearch, fast_gif_path)
    #os.system(cmd)


def run_trace(seed=3432, ysize=48, xsize=48, level=6, 
        max_goal_distance=100, n_playouts=300, 
        max_rollout_length=50, estimator='none', 
        prob_fn=goal_node_probs_fn, history_size=4, 
        do_render=False):

    # log params
    results = {'decision_ts':[],'decision_sts':[], 'dis_to_goal':[], 'actions':[],
               'ysize':ysize, 'xsize':xsize, 'level':level,
               'n_playouts':n_playouts, 'seed':seed, 'ests':[], 'est_inds':[],
               'max_rollout_length':max_rollout_length}

    states = []
    # restart at same position every time
    rdn = np.random.RandomState(seed)
    true_env = RoadEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level)
    start_state = true_env.reset(experiment_name=seed, goal_distance=max_goal_distance)

    # fast forward history steps so agent observes 4
    t = history_size
    state = [start_state[0], start_state[1], true_env.road_maps[t]]
    true_env.set_state(state)

    mcts_rdn = np.random.RandomState(seed+1)
    pmcts = PMCTS(env=deepcopy(true_env),random_state=mcts_rdn,node_probs_fn=prob_fn,
                n_playouts=n_playouts, rollout_length=max_rollout_length, 
                estimator=estimator,history_size=history_size)

    finished = False
    # draw initial state
    if do_render:
        true_env.render(state)
    print("SEED", seed)
    frames = []
    sframes = []
    while not finished:
        states.append(state)
        ry,rx = true_env.get_robot_state(state)
        current_goal_distance = true_env.get_distance_to_goal()

        # search for best action
        st = time.time()
        action, action_probs, playout_frames, playout_states, state_ests, state_est_indexes = pmcts.get_best_action(deepcopy(state), t)
        frames.append((true_env.get_state_plot(state), playout_frames))
        sframes.append((t, (ry,rx),  playout_states))
        et = time.time()

        next_state, reward, finished, _ = true_env.step(state, t, action)

        results['ests'].append(state_ests)
        results['est_inds'].append(state_est_indexes)
        print("decision took %s seconds"%round(et-st, 2))
        results['decision_sts'].append(st)
        results['decision_ts'].append(et-st)
        results['dis_to_goal'].append(current_goal_distance)
        results['actions'].append(action)
        if not finished:
            pmcts.update_tree_move(action)
            state = next_state
            t+=1
        else:
            results['reward'] = reward
            states.append(next_state)
        if do_render:
            true_env.render(next_state)
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    if reward>0:
        print("robot won reward={} after {} steps".format(reward,t))
    else:
        print("robot died reward={} after {} steps".format(reward,t))
        print("robot died reward={} after {} steps".format(reward,t))
        print("robot died reward={} after {} steps".format(reward,t))
        print("robot died reward={} after {} steps".format(reward,t))
        #embed()
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    true_env.close_plot()

    plt.clf()
    plt.close()
    #plot_true_scatters('trials',  seed=seed, reward=reward, sframes=sframes, t=t)
    if args.plot_playouts:
        plot_playout_scatters(true_env, 'trials', str(estimator), seed, reward, sframes,
                          pmcts.road_map_ests, pmcts.rollout_length,
                          t,plot_error=args.do_plot_error,gap=args.plot_playout_gap,min_agents_alive=4)
    return results

if __name__ == "__main__":
    import argparse
    # this seems to work well
    #python roadway_pmcts.py --seed 45 -r 100  -p 100 -l 6

    default_base_savedir = '../../trajectories_frames/saved/vqvae'
    # train loss .0488, test_loss sum .04 epoch 25
    default_vqvae_model_loadpath = os.path.join(default_base_savedir, 
            'vqvae4layer_base_k512_z32_dse00025.pkl')
    # train loss of .39, epoch 31
    default_pcnn_model_loadpath = os.path.join(default_base_savedir,
            'rpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00031.pkl')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=100, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=48, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=48, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=50, help='number of playouts for each step')
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
    prior = equal_node_probs_fn
    if args.prior_fn == 'goal':
        prior = goal_node_probs_fn
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

    goal_dis = args.max_goal_distance
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fname = 'all_results_model_%s_rollouts_%s_length_%s_prior_%s.pkl' %(
                                    args.model_type, 
                                    args.num_playouts, 
                                    args.rollout_steps,args.prior_fn)

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
            r = run_trace(seed=seed, ysize=args.ysize, xsize=args.xsize, level=args.level,
                          max_goal_distance=goal_dis, n_playouts=args.num_playouts,
                          max_rollout_length=args.rollout_steps, 
                          prob_fn=prior, estimator=args.model_type,
                          history_size=history_size, do_render=args.render)

            et = time.time()
            r['full_end_time'] = et
            r['full_start_time'] = st
            all_results[seed] = r
            pickle.dump(all_results,ffile)
            seed +=1
    embed()
    print("FINISHED")


