# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
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

from gym_trajectories.envs.vqvae import AutoEncoder
from gym_trajectories.envs.pixel_cnn import GatedPixelCNN

from gym_trajectories.envs.utils import discretized_mix_logistic_loss, get_cuts, to_scalar
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
        # JRH TIRED AND LIKELY HAVE A BUG
        #self.n_visits_ = 0
        self.n_visits_ = 1e-6
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

def get_relative_bearing(gy, gx, ry, rx):
    dy = gy-ry
    dx = gx-rx
    return np.rad2deg(math.atan2(dy,dx))

def goal_node_probs_fn(state, state_index, env, goal_loc):
    # TODO make env take in state to give goal/robot
    if not len(goal_loc[0]):
        print('couldnt find goal in given state estimate')
        embed()
        return equal_node_probs_fn(state, state_index, env, goal_loc)

    gy, gx = goal_loc[0][0], goal_loc[1][0]
    ry, rx = env.get_robot_state(state)
    bearing = get_relative_bearing(gy,gx,ry,rx)
    action_distances = np.abs(env.angles-bearing)
    actions_and_distances = list(zip(env.action_space, action_distances))
    best_angles = sorted(actions_and_distances, key=lambda tup: tup[1])
    best_actions = [b[0] for b in best_angles]

    best_angles = np.ones(len(env.action_space), dtype=np.float)
    top = len(env.action_space)/2
    best_angles[:top] = 2.0
    best_angles[:2] = 2.5
    #best_angles[1] = 2.5
    #best_angles[0] = 3.0
    best_angles = np.round(best_angles/float(best_angles.sum()), 2)

    unsorted_actions_and_probs = list(zip(best_actions, best_angles))
    actions_and_probs = sorted(unsorted_actions_and_probs, key=lambda tup: tup[0])
    #print("Bearing is", bearing)
    #print("GOAL")
    #print(gy, gx)
    #print("best action is", best_angles[0], best_angles[0])
    #print(zip(env.angles, actions_and_probs))
    return actions_and_probs

def equal_node_probs_fn(state, state_index, env, gm):
    probs = np.ones(len(env.action_space))/float(len(env.action_space))
    actions_and_probs = list(zip(env.action_space, probs))
    return actions_and_probs

#def get_min_goal_pixel(frame):

def get_vqvae_pcnn_model(state_index, est_inds, true_states, cond_states):
    rollout_length = len(est_inds)
    print("starting vqvae pcnn for %s predictions" %len(est_inds))
    # normalize data before putting into vqvae
    st = time.time()
    broad_states = ((cond_states-min_pixel)/float(max_pixel-min_pixel) ).astype(np.float32)[:,None]
    # transofrms HxWxC in range 0,255 to CxHxW and range 0.0 to 1.0
    nroad_states = Variable(torch.FloatTensor(broad_states)).to(DEVICE)
    x_d, z_e_x, z_q_x, cond_latents = vmodel(nroad_states)

    latent_shape = (6,6)
    _, ys, xs = cond_states.shape
    est = time.time()
    print("condition prep time", round(est-st,2))
    for ind, frame_num  in enumerate(est_inds):
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

    proad_states = np.zeros((rollout_length,ys,xs))
    print("starting image")
    ist = time.time()
    # generate road
    z_q_x = vmodel.embedding(all_pred_latents.view(all_pred_latents.size(0),-1))
    z_q_x = z_q_x.view(all_pred_latents.shape[0],6,6,-1).permute(0,3,1,2)
    x_d = vmodel.decoder(z_q_x)

    #x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=True)
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=True)
    proad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel
    goal_y_ests = {}
    goal_x_ests = {}
    for frame in range(proad_states.shape[0]):
        goal_y_ests[frame] = []
        goal_x_ests[frame] = []
        cgoal_est = np.where(proad_states[frame,0] == max_pixel)
        print(frame, "estimate goal from mean", cgoal_est)
        cest_len = len(cgoal_est[0])
        # set to zero
        if cest_len:
            proad_states[frame,0, cgoal_est[0], cgoal_est[1]] = 0.0
        if cest_len == 4:
            ye = cgoal_est[0]
            xe = cgoal_est[1]
            if ye.max()-ye.min() == 1:
                goal_y_ests[frame].append(ye.min())
            if xe.max()-xe.min() == 1:
                goal_x_ests[frame].append(xe.min())

    for cc in range(args.num_samples):
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, only_mean=False)
        sroad_states = (((np.array(x_tilde.cpu().data)+1.0)/2.0)*float(max_pixel-min_pixel)) + min_pixel

        print("STARTING", cc)
        # for each predicted state
        for frame in range(proad_states.shape[0]):
            sgoal_est = np.where(sroad_states[frame,0] == max_pixel)
            print('sampling', frame, sgoal_est)
            # if goal previously predicted is the right size - use it
            sest_len = len(sgoal_est[0])
            # zero out est goal
            if sest_len:
                sroad_states[frame,0,sgoal_est[0], sgoal_est[1]] = 0.0
            if sest_len == 4:
                ye = sgoal_est[0]
                xe = sgoal_est[1]
                print(ye,xe)
                if ye.max()-ye.min() == 1:
                    goal_y_ests[frame].append(ye.min())
                if xe.max()-xe.min() == 1:
                    goal_x_ests[frame].append(xe.min())
        proad_states = np.maximum(proad_states, sroad_states)

    maxy = proad_states.shape[2]-1
    maxx = proad_states.shape[3]-1
    for frame in range(proad_states.shape[0]):

        ly = sorted(goal_y_ests[frame])
        lx = sorted(goal_x_ests[frame])
        lly  = len(goal_y_ests[frame])
        llx  = len(goal_x_ests[frame])
        print('frame', frame)
        print(ly)
        print(lx)

        if lly and llx:
            my = np.int(np.median(ly))
            mx = np.int(np.median(lx))
            myp = min(my+1, maxy)
            mxp = min(mx+1, maxx)
            print('my', my, myp)
            print('mx', mx, mxp)
            proad_states[frame,0, my, mx]  = max_pixel
            proad_states[frame,0, myp, mx]  = max_pixel
            proad_states[frame,0, my, mxp]  = max_pixel
            proad_states[frame,0, myp, mxp]  = max_pixel

    iet = time.time()
    print("image pred time", round(iet-ist, 2))
    return proad_states.astype(np.int)[:,0]

def get_zero_model(state_index, est_inds, true_states, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions" %rollout_length)
    # normalize data before putting into vqvae
    return np.zeros_like(true_states[est_inds])

def get_none_model(state_index, est_inds, true_states, cond_states):
    rollout_length = len(est_inds)
    print("starting none %s predictions" %rollout_length)
    # normalize data before putting into vqvae
    return true_states[est_inds]

def get_false_neg_counts(true_road_map, pred_road_map):
   # true_road_map = self.env.road_maps[state_index]
   # pred_road_map = self.road_map_ests[state_index]
    road_true_road_map = deepcopy(true_road_map)
    road_pred_road_map = deepcopy(pred_road_map)
    true_goal = np.where(true_road_map==max_pixel)
    pred_goal = np.where(pred_road_map==max_pixel)
    print(true_goal, pred_goal)
    road_true_road_map[true_goal] = 0
    road_pred_road_map[pred_goal] = 0
    road_true_road_map[road_true_road_map>0] = 1
    road_pred_road_map[road_pred_road_map>0] = 1

    # measure error of 0th pixel of goal
    goal_err = 0.0
    if len(pred_goal[0]):
        if len(true_goal[0]):
            goal_err = np.abs(pred_goal[0][0]-true_goal[0][0])
    # need to measure false positive of goal
    # predict free where there was car  # bad
    false_neg = (road_true_road_map*np.abs(road_true_road_map-road_pred_road_map))
    false_neg[false_neg>0] = 1
    false_neg_count = false_neg.sum()

    print('max',road_true_road_map.max())
    print('max',road_pred_road_map.max())
    false_pos = road_pred_road_map*np.abs(road_true_road_map-road_pred_road_map)
    false_neg = road_true_road_map*np.abs(road_true_road_map-road_pred_road_map)

    error = np.ones_like(road_true_road_map)*254
    error[false_pos>0] = 30
    error[true_goal] = 160
    error[pred_goal] = 130
    error[false_neg>0] = 1
    #print('false neg count', state_index, false_neg_count)

    return false_neg_count, error



class PMCTS(object):
    def __init__(self, env, random_state, node_probs_fn, c_puct=1.4,
            n_playouts=1000, rollout_length=20, estimator=get_vqvae_pcnn_model,
            history_size=4):
        # use estimator for planning, if false, use env
        # make sure it does a full rollout the first time
        self.full_rollouts_every = 1
        self.last_full_rollout = self.full_rollouts_every*10
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
        #  what was estimated when we received a state
        self.decision_time_road_map_ests = np.zeros_like(self.env.road_maps)

    def get_children(self, node):
        print('node name', node.name)
        for i in node.children_.keys():
            print(node.children_[i].__dict__)
        return [node.children_[i].__dict__ for i in node.children_.keys()]

    def playout(self, playout_num, state, state_index):
        # set new root of MCTS (we've taken a step in the real game)
        # only sets robot and goal states
        # get future playouts from past states

        cnt = 0
        logging.debug('+++++++++++++START PLAYOUT NUM: {} FOR STATE: {}++++++++++++++'.format(playout_num,state_index))
        init_state = state
        init_state_index = state_index
        node = self.root
        won = False
        frames = []
        # stack true state then vstate
        vstate = [state[0], self.road_map_ests[state_index]]
        frames.append((self.env.get_state_plot(state), self.env.get_state_plot(vstate)))
        # always use true state for first
        finished,reward = self.env.set_state(state, state_index)


        while True:
            if node.is_leaf():
                if (not finished) and (state_index+1 < self.last_state_index_est):
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action
                    #gl = self.playout_goal_locs[len(self.playout_goal_locs)/2]
                    gl = self.playout_goal_locs[len(self.playout_goal_locs)//2]
                    actions_and_probs = self.node_probs_fn(state, state_index, self.env, gl)
                    node.expand(actions_and_probs)
                    # if you have a neural network - use it here to bootstrap the value
                    # otherwise, playout till the end
                    # rollout one randomly
                    reward = self.rollout_from_state(state, state_index, cnt)
                    #frames.extend(rframes)
                    finished = True
                # finished the rollout
                node.update(reward)
                if reward > 0:
                    node.n_wins+=1
                    won = True
                return won, reward
            else:
                # greedy select
                # trys actions based on c_puct
                action, new_node = node.get_best(self.c_puct)
                # cant actually use next state because we dont know it
                next_state_index = state_index + 1
                vnext_road_map = self.road_map_ests[next_state_index]
                next_vstate, reward, finished, _ = self.env.model_step(state, state_index, action, vnext_road_map)
                logging.debug("GREEDY SELECT state_index:%s action:%s reward:%s finished %s" %(state_index,action,reward,finished))
                cnt+=1
                node = new_node
                state = next_vstate
                state_index = next_state_index
                self.add_robot_playout(state, state_index, reward)


    def get_rollout_action(self, state):
        valid_actions = self.env.action_space
        action_probs = self.rdn.rand(len(valid_actions))
        action_probs = action_probs / np.sum(action_probs)
        act_probs = tuple(zip(valid_actions, action_probs))
        acts, probs = zip(*act_probs)
        act = self.rdn.choice(acts, p=probs)
        return act, act_probs

    def rollout_from_state(self, state, state_index, cnt):
        logging.debug('-------------------------------------------')
        logging.debug('starting random rollout from state: {} limit {} rollout length'.format(state_index,self.rollout_length))
        #print('starting random rollout from state: {}'.format(state_index))

        # comes in already transformed
        #rframes = []
        finished,reward = self.env.set_state(state, state_index)
        if finished:
            return reward#, rframes
        c = 0
        while not finished:
            # one less because we want next_state to be modeled
            if state_index < self.last_state_index_est-1:
                action, action_probs = self.get_rollout_action(state)
                logging.debug("rollout --- state_index %s action %s"%( state_index,action))
                self.env.set_state(state, state_index)
                next_state_index = state_index + 1
                vnext_road_map = self.road_map_ests[next_state_index]
                next_vstate, reward, finished, _ = self.env.model_step(state, state_index, action, vnext_road_map)
                state_index = next_state_index
                state = next_vstate
                # get robot location from previous step
                self.add_robot_playout(state, state_index, reward)

                if next_vstate[1].sum() < 1:
                    print("rollout next_state has no sum!", state_index)
                #    embed()


                # true and vq state
                c+=1
                if finished:
                    logging.debug('finished rollout after {} steps with reward {}'.format(c,reward))
                    #print('finished rollout after {} steps with reward {}'.format(c,reward))
            else:
                # stop early
                #print('stopping rollout after {} steps with reward {}'.format(c,reward))
                logging.debug('stopping rollout after {} steps with reward {}'.format(c,reward))
                finished = True

        return reward


    def reset_playout_states(self, start_state_index):
        self.start_state_index = start_state_index
        self.playout_robots = np.zeros((self.rollout_length+1, self.env.ysize, self.env.xsize))
        self.playout_road_maps = np.zeros((self.rollout_length+1, self.env.ysize, self.env.xsize))

    def get_relative_index(self, state_index):
        return state_index-self.start_state_index

    def add_robot_playout(self, state, state_index, bonus):
        # bonus - can feed in reward and we will convert it to pixel color
        if bonus > 1:
            bonus = 1
        if bonus < 1:
            bonus = -10
        ry,rx = self.env.get_robot_state(state)
        relative_state = self.get_relative_index(state_index)
        #print('robot playout', self.start_state_index, state_index, relative_state)
        try:
            self.playout_robots[relative_state,ry,rx] = self.env.robot.color
        except Exception, e:
            print(e, 'rob')
            embed()



    def get_action_probs(self, state, state_index, temp=1e-2):
        #print("-----------------------------------")
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0
        logging.debug("starting playouts for state_index %s" %state_index)
        # only run last rollout
        finished,rr = self.env.set_state(state, state_index)
        self.add_robot_playout(state, state_index, rr)
        if not finished:
            for n in range(self.n_playouts):
                from_state = deepcopy(state)
                from_state_index = deepcopy(state_index)
                self.playout(n, from_state, from_state_index)
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
        return actions, action_probs


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

    def get_local_false_neg_counts(self, state, state_index):
        true_road_map = self.env.road_maps[state_index]
        pred_road_map = self.road_map_ests[state_index]

        ry,rx = self.env.get_robot_state(state)
        ys = self.env.ysize-1
        xs = self.env.xsize-1
        bs = 5
        lby = min(ry+bs, ys)
        lbx = min(rx+bs, xs)
        iby = max(ry-bs, 0)
        ibx = max(rx-bs, 0)
        # predict free where there was car  # bad
        false_neg = (true_road_map*np.abs(true_road_map-pred_road_map))
        false_neg[false_neg>0] = 1
        false_neg_count = false_neg.sum()
        local_false_neg_count = np.sum(false_neg[iby:lby, ibx:lbx])
        return local_false_neg_count

    def estimate_the_future(self, state, state_index):
        self.reset_playout_states(state_index)
        #######################
        # determine how different the predicted was from true for the last state
        # pred_road should be all zero if no cars have been predicted
        self.decision_time_road_map_ests[state_index] = deepcopy(state[1])
        false_neg_count, false_neg = get_false_neg_counts(self.env.road_maps[state_index], self.road_map_ests[state_index])
        local_false_neg_count = self.get_local_false_neg_counts(state, state_index)

        print('false neg is', false_neg_count)
        # false_neg_count is ~ 25 when the pcnn predicts all zeros
        print('local false neg is', local_false_neg_count, 'last full rollout', self.last_full_rollout)

        #if (local_false_neg_count > 0) or (false_neg_count > 15) or (self.last_full_rollout > self.full_rollouts_every):
        # always replan
        if True:
            self.last_full_rollout = 1
            print("running all rollouts")
            # ending index is noninclusive
            # starting index is inclusive
            est_from = state_index+1
            pred_length = self.rollout_length
        else:
            print("running one rollout")
            # only run last rollout that was not finished
            est_from = state_index+self.rollout_length
            pred_length = 1
            self.last_full_rollout += 1
        try:
            # put in the true road map for this step
            self.road_map_ests[state_index] = self.env.road_maps[state_index]
            s = range(self.road_map_ests.shape[0])
            # limit prediction lengths
            est_from = min(self.road_map_ests.shape[0], est_from)
            est_to = est_from + pred_length
            est_to = min(self.road_map_ests.shape[0], est_to)
            cond_to = est_from
            # end of conditioning
            cond_from = cond_to-self.history_size
            #print("state index", state_index)
            #print("condition", cond_from, cond_to)
            #print(s[cond_from:cond_to])
            #print("estimate", est_from, est_to)
            #print(range(est_from, est_to))
            rinds = range(est_from, est_to)
            self.last_state_index_est = est_to
            if not len(rinds):
                ests = []
            else:
                # can use past frames because we add them as we go
                cond_frames = self.road_map_ests[cond_from:cond_to]
                ests = self.estimator(state_index, rinds, self.env.road_maps, cond_frames)
                self.road_map_ests[rinds] = ests
                est_inds = range(state_index,est_to)
                #print('this_rollout', est_inds) #should be rollout_length +1 for the current_state
                self.playout_road_maps = self.road_map_ests[est_inds]
                self.playout_goal_locs = []
                # start assuming it is zero
                gl = [[self.env.ysize//2], [self.env.xsize//2]]
                for i, rm in enumerate(self.playout_road_maps):
                    found_goal, fgl = self.env.get_goal_from_roadmap(rm)
                    if not found_goal:
                        print("didnt find goal", i, fgl)
                    else:
                        print("found goal", i, gl)
                        gl = fgl
                    self.playout_goal_locs.append(gl)
                    ccc = state_index+i
                    #print(state_index, ccc)

                    false_neg_count, err = get_false_neg_counts(self.env.road_maps[ccc], self.road_map_ests[ccc])
                    # = get_false_neg_counts(ccc)
                    #f,ax = plt.subplots(1,3,figsize=(12,4))
                    #ax[0].imshow(self.env.road_maps[ccc], origin='lower', vmin=0, vmax=max_pixel)
                    #ax[0].set_title("Step %d: Oracle" %i)
                    #ax[1].imshow(rm, origin='lower', vmin=0, vmax=max_pixel)
                    #ax[1].set_title("%s Sample Model"%args.num_samples)
                    #ax[2].imshow(err, origin='lower', cmap='Set1', vmin=0, vmax=max_pixel )  #cmap='Set1',
                    #ax[2].set_title("Error")
                    #plt.savefig("example_sample_%02d_step_%02d.png"%(args.num_samples,i))
                    #plt.close()
            #sys.exit()
            false_negs = []
            for xx, i in enumerate(rinds):
                #fnc,fn = self.get_false_neg_counts(i)
                fnc,fn = get_false_neg_counts(self.env.road_maps[i], self.road_map_ests[i])
                false_negs.append(fnc)
        except Exception, e:
            print(e)
            embed()
        return false_negs, rinds


    def get_best_action(self, state, state_index):
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        self.env.set_state(state, state_index)
        self.estimate_the_future(state, state_index)
        acts, probs = self.get_action_probs(state, state_index, temp=1e-3)
        act = self.rdn.choice(acts, p=probs)
        logging.info("mcts chose action {} in state: {}".format(act,state_index))
        return act, probs

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

#def get_error_frame(true_road, model_road):
#    err = np.zeros_like(true_road)
#    true_car = true_road>0
#    pred_car = model_road>0
#    true_free = true_road<1
#    pred_free = model_road<1
#    # predict car where there was free space
#    false_pos = true_free.astype(np.int)+pred_car.astype(np.int)
#    err[false_pos>1] = 15 # false pos
#    # predict free where there was car  # bad
#    false_neg = true_car.astype(np.int)+pred_free.astype(np.int)
#    err[false_neg>1] = 250 # false neg
#    return err



def plot_playout_scatters(true_env, base_path,  fname,
                         model_type,
                         seed, reward, playout_frames,
                         model_road_maps, decision_time_road_map_ests,  rollout_length,
                         plot_error=True, gap=3, min_agents_alive=4,
                         do_plot_playouts=False, history_size=4):
    plt.ioff()
    true_road_maps = true_env.road_maps
    if plot_error:
        fpath = os.path.join(base_path,model_type,'E_seed_%s_%s'%(seed,fname))
    else:
        fpath = os.path.join(base_path,model_type,'P_seed_%s_%s'%(seed,fname))

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fast_path = os.path.join(base_path,model_type,'T_seed_%s_%s'%(seed,fname))
    if not os.path.exists(fast_path):
        os.makedirs(fast_path)
    start_state_index = playout_frames[0]['state_index']
    last_state_index = playout_frames[-1]['state_index']
    total_steps = start_state_index+len(playout_frames)
    for ts, step_frame in enumerate(playout_frames):
        state_index = step_frame['state_index']
        print("plotting true frame {}/{} state_index {}/{}".format(ts,total_steps,state_index, last_state_index))
        # true frame
        ry,rx = step_frame['robot_yx']
        ry = ry/float(true_env.ysize)
        rx = rx/float(true_env.xsize)

        true_state = true_env.road_maps[state_index]
        state = ((ry,rx), true_state)
        true_frame = true_env.get_state_plot(state)

        model_state = decision_time_road_map_ests[state_index]
        vstate = ((ry,rx), model_state)
        model_frame = true_env.get_state_plot(vstate)

        _, model_error = get_false_neg_counts(deepcopy(true_env.road_maps[state_index]), deepcopy(model_road_maps[state_index]))
        #model_error = get_error_frame(
        try:
            fast_fname = 'fast_seed_%06d_step_%04d.png'%(seed, state_index)
            ft,axt=plt.subplots(1,1, figsize=(3,3))
            axt.imshow(true_frame, origin='lower', vmin=0, vmax=255 )
            axt.set_title("true step:{}/{}".format(ts,total_steps))
            #axt[1].imshow(model_frame, origin='lower', vmin=0, vmax=255 )
            #axt[1].set_title("{} model step:{}/{}".format(model_type, ts, total_steps))
            #axt[2].imshow(model_error, origin='lower', cmap='Set1')
            #axt[2].set_title("model error".format(model_type))
            ft.tight_layout()
            plt.savefig(os.path.join(fast_path,fast_fname))
            plt.close()
        except Exception, e:
            print(e, 'plot')
            embed()

        # playtouts is size episode_length, y, x
        c = 0
        if do_plot_playouts:
            playout_robot_states = step_frame['playout_robot_states']
            playout_model_states = step_frame['playout_model_states']
            num_playout_steps = playout_model_states.shape[0]
            for playout_ind in range(0,num_playout_steps):
                playout_state_index = min(state_index+playout_ind, true_road_maps.shape[0]-1)
                print("plotting playout state_index {} - {} step {}/{}".format(state_index, playout_state_index, playout_ind, num_playout_steps))

                true_playout_frame = true_road_maps[playout_state_index]+playout_robot_states[playout_ind]
                est_playout_frame = playout_model_states[playout_ind]+playout_robot_states[playout_ind]
                _, rollout_model_error  = get_false_neg_counts(deepcopy(true_playout_frame), deepcopy(est_playout_frame))
                fname = 'seed_%06d_tstep_%04d_pstep_%04d.png'%(seed, state_index, playout_state_index)
                f,ax=plt.subplots(1,4, figsize=(16,3.5))
                ax[0].imshow(true_frame, origin='lower', vmin=0, vmax=255 )
                ax[0].set_title("decision t: {}/{}".format(state_index,last_state_index))
                ax[1].imshow(true_playout_frame, origin='lower', vmin=0, vmax=255 )
                ax[1].set_title("oracle rollout {} step:{}/{}".format(playout_state_index, playout_ind, num_playout_steps ))
                ax[2].imshow(est_playout_frame, origin='lower', vmin=0, vmax=255 )
                ax[2].set_title("model rollout {} step:{}/{}".format(playout_state_index, playout_ind, num_playout_steps ))
                ax[3].imshow(rollout_model_error , origin='lower', cmap='Set1')
                ax[3].set_title("error in model")
                f.tight_layout()
                plt.savefig(os.path.join(fpath,fname))
                plt.close()

    print("making gif")
    gif_path = 'seed_{}.gif'.format(seed)
    search = os.path.join(fpath, 'seed_*.png')
    cmd = 'convert -delay 1/100000 *.png %s \n'%( gif_path)
    sh_path = os.path.join(fpath, 'run_seed_{}.sh'.format(seed))
    sof = open(sh_path, 'w')
    sof.write(cmd)
    sof.close()
    print("FINISHED WRITING TO", os.path.split(sh_path)[0])

    fast_gif_path = 'fast_seed_{}.gif'.format(seed)
    cmd = 'convert -delay 1/30 *.png %s\n'%(fast_gif_path)
    fast_sh_path = os.path.join(fast_path, 'run_fast_seed_{}.sh'.format(seed))
    of = open(fast_sh_path, 'w')
    of.write(cmd)
    of.close()
    print("FINISHED WRITING TO", os.path.split(fast_sh_path)[0])



def run_trace(fname, seed=3432, ysize=48, xsize=48, level=6,
        max_goal_distance=100, n_playouts=300,
        max_rollout_length=50, estimator='empty',
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
    true_env = RoadEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level, agent_max_speed=args.agent_max_speed)
    state = true_env.reset(experiment_name=seed, goal_distance=max_goal_distance,
                           condition_length=history_size, goal_speed=args.goal_speed)

    # fast forward history steps so agent observes 4
    t = history_size
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
    playout_frames = []
    value = 0
    while not finished:
        states.append(state)
        #try:
        #    assert(state[1].max() == max_pixel)
        #except:
        #    print("TRACE", t)
        #    embed()
        # search for best action
        st = time.time()
        #action, action_probs, this_playout_frames, this_playout_states, state_ests, state_est_indexes = pmcts.get_best_action(deepcopy(state), t)
        action, action_probs = pmcts.get_best_action(deepcopy(state), t)
        #JRH
        #frames.append((true_env.get_state_plot(state), this_playout_frames))
        et = time.time()

        ry,rx = true_env.get_robot_state(state)
        next_state, reward, finished, _ = true_env.step(state, t, action)

        playout_frames.append({'state_index':t, 'robot_yx':(ry,rx),
                               'playout_robot_states':deepcopy(pmcts.playout_robots),
                               'playout_model_states':deepcopy(pmcts.playout_road_maps),
                               })



        print("CHOSE ACTION", action)

        print("decision took %s seconds"%round(et-st, 2))
        results['decision_sts'].append(st)
        results['decision_ts'].append(et-st)
        #results['dis_to_goal'].append(current_goal_distance)
        results['actions'].append(action)
        if not finished:
            pmcts.update_tree_move(action)
            state = next_state
            t+=1
        else:
            results['reward'] = reward
            states.append(next_state)
        if do_render:
            print("stepping action", action)
            true_env.render(next_state)
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    if reward>0:
        print("robot won reward={} after {} steps".format(reward,t))
    else:
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
    if args.save_plots:

        plot_playout_scatters(true_env, os.path.join(savedir, 'trials'), fname.replace('.pkl',''),
                          str(estimator), seed, reward,
                          playout_frames=playout_frames,
                          model_road_maps=pmcts.road_map_ests,
                          decision_time_road_map_ests = pmcts.decision_time_road_map_ests,
                          rollout_length=pmcts.rollout_length,
                          plot_error=args.do_plot_error,
                          gap=args.plot_playout_gap,
                          min_agents_alive=4,
                          do_plot_playouts=args.plot_playouts,
                          history_size=history_size)
    return results

if __name__ == "__main__":
    import argparse
    # this seems to work well
    #python roadway_pmcts.py --seed 45 -r 100  -p 100 -l 6

    default_base_savedir = '../../models'
    savedir = '../../results'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # false negs over 10 steps  for seed 35 [17, 21, 26, 25, 32, 40, 38, 38, 39, 41]
    #vq_name = 'vqvae4layer_base_k512_z32_dse00025.pkl'
    # train loss .034, test_loss sum .0355 epoch 51
    vq_static_name = 'vqvae4layer_base_k512_z32_dse00051.pkl'
    # train loss of .39, epoch 31
    #pcnn_name = 'rpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00031.pkl'
    # false negs over 10 steps for seed 35
    # [11, 13, 14, 12, 19, 27, 30, 35, 38, 37]
    # [10, 12, 13, 12, 19, 27, 30, 33, 36, 38]
    # [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # train loss of 1.09, test loss 1.25 epoch 10
    pcnn_static_name = 'nrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00010.pkl'
    # false negs over 10 steps for seed 35
    # [13, 13, 16, 13, 23, 29, 31, 36, 35, 37]
    # [12, 12, 15, 16, 23, 29, 31, 33, 35, 38]
    # [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # [27, 23, 26, 27, 30, 29, 32, 32, 26, 26]
    #pcnn_name = 'nrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00026.pkl'
    vq_moving_name = 'vqvae4layer_base_k512_z32_dse00064.pkl'
    #pcnn_moving_name = 'mrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00004.pkl'
    #pcnn_moving_name = 'mrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00008.pkl'
    pcnn_moving_name = 'mrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00008.pkl'
    #pcnn_moving_name = 'nrpcnn_id512_d256_l15_nc4_cs1024_base_k512_z32e00003.pkl'

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=100, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=48, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=48, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=10, help='limit number of steps taken be random rollout')
    #parser.add_argument('-vq', '--vqvae_model_loadpath', type=str, default=default_vqvae_model_loadpath)
    #parser.add_argument('-pcnn', '--pcnn_model_loadpath', type=str, default=default_pcnn_model_loadpath)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')
    parser.add_argument('-t', '--model_type', type=str, default='vqvae_pcnn_model')
    parser.add_argument('-sams', '--num_samples', type=int , default=5)
    parser.add_argument('-gs', '--goal_speed', type=float , default=0.5)
    parser.add_argument('-as', '--agent_max_speed', type=float , default=1.0)
    parser.add_argument('--save_pkl', action='store_false', default=True)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--do_plot_error', action='store_false', default=True)
    parser.add_argument('--plot_playouts', action='store_true', default=False)
    parser.add_argument('--save_plots', action='store_true', default=False)
    parser.add_argument('-gap', '--plot_playout_gap', type=int, default=3, help='gap between plot playouts for each step')
    parser.add_argument('-f', '--prior_fn', type=str, default='goal', help='options are goal or equal')

    args = parser.parse_args()
    if args.goal_speed == 0.5:
        print("MOVING")
        pcnn_name = pcnn_moving_name
        vq_name = vq_moving_name
    else:
        print("STATIC")
        pcnn_name = pcnn_static_name
        vq_name = vq_static_name
    default_pcnn_model_loadpath = os.path.join(default_base_savedir, pcnn_name)
    default_vqvae_model_loadpath = os.path.join(default_base_savedir, vq_name)

    #equal_node_probs_fn(
    if args.prior_fn == 'goal':
        prior = goal_node_probs_fn
    else:
        prior = equal_node_probs_fn
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
        if os.path.exists(default_vqvae_model_loadpath):
            vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix,num_clusters=num_clusters, encoder_output_size=num_z).to(DEVICE)
            vqvae_model_dict = torch.load(default_vqvae_model_loadpath, map_location=lambda storage, loc: storage)
            vmodel.load_state_dict(vqvae_model_dict['state_dict'])
            epoch = vqvae_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_vqvae_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_vqvae_model_loadpath))
            sys.exit()

        if os.path.exists(default_pcnn_model_loadpath):
            pcnn_model = GatedPixelCNN(num_clusters, DIM, N_LAYERS,
                    history_size, spatial_cond_size=cond_size).to(DEVICE)
            pcnn_model_dict = torch.load(default_pcnn_model_loadpath, map_location=lambda storage, loc: storage)
            pcnn_model.load_state_dict(pcnn_model_dict['state_dict'])
            epoch = pcnn_model_dict['epochs'][-1]
            print('loaded checkpoint at epoch: {} from {}'.format(epoch,
                                                   default_pcnn_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(default_pcnn_model_loadpath))
            sys.exit()

    if args.model_type == 'vqvae_pcnn_model':
       pcnn_name  = default_pcnn_model_loadpath.split('_e')[1].replace('.pkl', '')
       vq_name  = default_vqvae_model_loadpath.split('_e')[1].replace('.pkl', '')
    else:
       pcnn_name  = 'na'
       vq_name = 'na'

    goal_dis = args.max_goal_distance
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fname = 'sample_%s_gall_results_prior_%s_model_%s_%s_%s_rollouts_%s_length_%s_level_%s_as_%01.02f_gs_%01.02f_gd_%03d.pkl' %(
                                    args.num_samples,
                                    args.prior_fn,
                                    args.model_type,
                                    vq_name,
                                    pcnn_name,
                                    args.num_playouts,
                                    args.rollout_steps,
                                    args.level, args.agent_max_speed, args.goal_speed,
                                    args.max_goal_distance)

    fpath = os.path.join(savedir, fname)
    if os.path.exists(fpath):
        print('loading previous results from %s' %fpath)
        try:
            ffile = open(fpath, 'rb')
            all_results = pickle.load(ffile)
            ffile.close()
            print('found %d runs in file' %(len(all_results.keys())-1))
        except EOFError, e:
            print('end of file', e)
            embed()
            print('unable to load ffile:%s' %fpath)
            all_results = {'args':args}
    else:
        all_results = {'args':args}

    for i in range(args.num_episodes):
        if ((seed in all_results.keys()) and args.save_pkl):
            try:
                rew = all_results[seed]['reward']
                print("seed %s already in results, score was %s" %(seed,rew))
                if rew>=0:
                    seed +=1
                    continue
                else:
                    print('rerunning since this one was lost')

            except:
                print("no reward - rerun")
        print("STARTING EPISODE %s seed %s" %(i,seed))
        print(args.save_pkl)
        st = time.time()
        r = run_trace(fname, seed=seed, ysize=args.ysize, xsize=args.xsize, level=args.level,
                      max_goal_distance=goal_dis, n_playouts=args.num_playouts,
                      max_rollout_length=args.rollout_steps,
                      prob_fn=prior, estimator=args.model_type,
                      history_size=history_size, do_render=args.render)

        et = time.time()
        r['full_end_time'] = et
        r['full_start_time'] = st
        all_results[seed] = r

        if args.save_pkl:
            ffile = open(fpath, 'w+')
            pickle.dump(all_results,ffile)
            print("saved seed %s"%seed)
            ffile.close()
        seed +=1
    embed()
    print("FINISHED")


