# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

import matplotlib.pyplot as plt
from imageio import imwrite
from gym_trajectories.envs.road import RoadEnv
import time
import numpy as np
from IPython import embed
from copy import deepcopy
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from gym_trajectories.envs.vq_vae import AutoEncoder, to_scalar
from gym_trajectories.envs.utils import discretized_mix_logistic_loss
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

def get_vq_from_road(road_state):
    road_state = Variable(transforms.ToTensor()(road_state[:,:,None].astype(np.float32)))
    x_d, z_e_x, z_q_x, latents = vmodel(road_state[None])
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    vroad_state = x_tilde[0,0].data.numpy() 
    uvroad_state = ((0.5*vroad_state+0.5)*255).astype(np.uint8)
    return uvroad_state

def get_vq_from_roads(road_states):
    print("precomputing road state estimates")
    road_states = Variable(transforms.ToTensor()(road_states.transpose(1,2,0).astype(np.float32)))[:,None]
    x_d, z_e_x, z_q_x, latents = vmodel(road_states)
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    vroad_state = x_tilde.data.numpy() 
    uvroad_states = ((0.5*vroad_state+0.5)*255).astype(np.uint8)[:,0]
    return uvroad_states


class PMCTS(object):
    def __init__(self, env, random_state, node_probs_fn, c_puct=1.4, 
            n_playouts=1000, rollout_length=300, use_est=False):
        # use estimator for planning, if false, use env
        self.use_est = use_est
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
        if self.use_est:
            self.road_map_ests = get_vq_from_roads(self.env.road_maps)

    def get_children(self, node):
        print('node name', node.name)
        for i in node.children_.keys():
            print(node.children_[i].__dict__)
        return [node.children_[i].__dict__ for i in node.children_.keys()]

    def playout(self, playout_num, state, state_index):
        # set new root of MCTS (we've taken a step in the real game)
        # only sets robot and goal states
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
        ry,rx = self.env.get_robot_state(state)
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
                if self.use_est:
                    state = next_vstate
                else:
                    state = next_state

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

        try:
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

                    if self.use_est:
                        state = next_vstate
                    else:
                        state = next_state

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
        except Exception, e:
            print(e)
            embed()
        return value, rframes


    def get_action_probs(self, state, state_index, temp=1e-2):
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0

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
            #embed()
        #if state_index == 8:
        #    print('state_index')
        #    embed()

        self.env.set_state(state, state_index)
        act_visits = [(act,float(node.n_visits_)) for act, node in self.root.children_.items()]
        try:
            actions, visits = zip(*act_visits)
        except Exception, e:
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

    def get_best_action(self, state, state_index):
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        self.env.set_state(state, state_index)
        acts, probs, playout_frames = self.get_action_probs(state, state_index, temp=1e-3)
        act = self.rdn.choice(acts, p=probs)
        logging.info("mcts chose action {} in state: {}".format(act,state_index))
        return act, probs, playout_frames, self.playout_states

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

def run_trace(seed=3432, ysize=40, xsize=40, level=5, max_goal_distance=100,
              n_playouts=300, max_rollout_length=50):

    # log params
    results = {'decision_ts':[], 'dis_to_goal':[], 'actions':[],
               'ysize':ysize, 'xsize':xsize, 'level':level,
               'n_playouts':n_playouts, 'seed':seed,
               'max_rollout_length':max_rollout_length}

    states = []
    # restart at same position every time
    rdn = np.random.RandomState(seed)
    true_env = RoadEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level)
    state = true_env.reset(experiment_name=seed, goal_distance=max_goal_distance)

    mcts_rdn = np.random.RandomState(seed+1)
    #pmcts = PMCTS(env=deepcopy(true_env),random_state=mcts_rdn,node_probs_fn=equal_node_probs_fn,
    #            n_playouts=n_playouts,rollout_length=max_rollout_length)
    pmcts = PMCTS(env=deepcopy(true_env),random_state=mcts_rdn,node_probs_fn=goal_node_probs_fn,
                n_playouts=n_playouts,rollout_length=max_rollout_length, use_est=True)

    t = 0
    finished = False
    # draw initial state
    #true_env.render(state)
    print("SEED", seed)
    frames = []
    sframes = []
    while not finished:
        states.append(state)
        ry,rx = true_env.get_robot_state(state)
        current_goal_distance = true_env.get_distance_to_goal()

        # search for best action
        st = time.time()
        action, action_probs, playout_frames, playout_states = pmcts.get_best_action(deepcopy(state), t)
        frames.append((true_env.get_state_plot(state), playout_frames))
        sframes.append((true_env.get_state_plot(state), playout_states))
        et = time.time()

        next_state, reward, finished, _ = true_env.step(state, t, action)

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
        #true_env.render(next_state)
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    if reward>0:
        print("robot won reward={} after {} steps".format(reward,t))
    else:
        print("robot died reward={} after {} steps".format(reward,t))
        embed()
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    #true_env.close_plot()

    plt.clf()
    plt.close()
    fpath = 'trials/road-vqvae/Eseed_{}'.format(seed)
# PLOT TRUE SCATTERS
#    try:
#
#        if not os.path.exists(fpath):
#            os.makedirs(fpath)
#        for ts in range(len(sframes)):
#            print("plotting true frame {}/{}".format(ts,t))
#            # true frame
#            actual_frame = sframes[ts][0]
#            fname = 'seed_%06d_tstep_%04d.png'%(seed, ts)
#            f,ax=plt.subplots(1,1, figsize=(3,3.5))
#            ax.imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
#            ax.set_title("true step:{}/{} reward:{}".format(ts,t,round(reward,2)))
#            plt.savefig(os.path.join(fpath,fname))
#            plt.close()
#        print("making gif")
#        gif_path = os.path.join(fpath, 'seed_{}_reward_{}.gif'.format(seed, int(reward)))
#        search = os.path.join(fpath, 'seed_*.png')
#        cmd = 'convert -delay 1/60 %s %s'%(search, gif_path)
#        os.system(cmd)
#
#    except Exception, e:
#        print(e)
#        embed()
# PLOT AGENT SCATTERS

    try:

        if not os.path.exists(fpath):
            os.makedirs(fpath)
        for ts in range(len(sframes)):
            print("plotting true frame {}/{}".format(ts,t))
            # true frame
            actual_frame = sframes[ts][0]
            # list of tuples with (real playout state, est playout state)
            playouts = sframes[ts][1]
            c = 0
            gap = 5 
            for pn, pframe in enumerate(playouts):
                    if not c%gap:
                        if c > 10:
                            if pframe.sum()<(4*true_env.robot.color-1):
                                continue

                        print(pframe.sum())
                        print("plotting step {}/{} playout step {}".format(ts,t,pn))
                        true_playout_frame = true_env.road_maps[pn]+pframe+true_env.goal_map
                        est_playout_frame = pmcts.road_map_ests[pn]+pframe+true_env.goal_map
                        est_err = np.sqrt((true_env.road_maps[pn]-pmcts.road_map_ests[pn]).astype(np.float)**2)

                        fname = 'seed_%06d_tstep_%04d_pstep_%04d.png'%(seed, ts, pn)
                        f,ax=plt.subplots(1,4, figsize=(14,3.5))
                        ax[0].imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
                        ax[0].set_title("true state step:{}/{}".format(ts,t))
                        ax[1].imshow(true_playout_frame, origin='lower', vmin=0, vmax=255 )
                        ax[1].set_title("rollout step:{}/{}".format(pn,pmcts.rollout_length))
                        ax[2].imshow(est_playout_frame, origin='lower', vmin=0, vmax=255 )
                        ax[2].set_title("rollout model:{}/{}".format(pn,pmcts.rollout_length))
                        ax[3].imshow(est_err, origin='lower')
                        ax[3].set_title("error in model:{}/{}".format(pn,pmcts.rollout_length))
                        plt.savefig(os.path.join(fpath,fname))
                        plt.close()
                    c+=1
        print("making gif")
        gif_path = os.path.join(fpath, 'seed_{}_reward_{}_gap_{}_error.gif'.format(seed, int(reward),gap))
        search = os.path.join(fpath, 'seed_*.png')
        cmd = 'convert -delay 1/1000 %s %s'%(search, gif_path)
        os.system(cmd)

    except Exception, e:
        print(e)
        embed()

# PLOT TRACES
#    try:
#        if not os.path.exists(fpath):
#            os.makedirs(fpath)
#        for ts in range(len(frames)):
#            print("plotting true frame {}/{}".format(ts,t))
#            # true frame
#            actual_frame = frames[ts][0]
#            # list of tuples with (real playout state, est playout state)
#            playouts = frames[ts][1]
#            keys = playouts.keys()
#            inds = range(len(keys))
#            selected_playouts = rdn.choice(inds, min(1, len(inds)), replace=False)
#            for pn in sorted(selected_playouts):
#                key = keys[pn]
#                score = key[1]
#                fs = playouts[key]
#                num = key[0]
#                print("plotting step {}/{} playout {}".format(ts,t,key))
#                for pf, (true_playout_frame,est_playout_frame) in enumerate(fs):
#                    if not pf:
#                        fname = 'seed_%06d_tstep_%04d_pnum_%04d_pstep_%04d.png'%(seed, ts, num, pf)
#                        f,ax=plt.subplots(1,3, figsize=(10,3.5))
#                        ax[0].imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
#                        ax[0].set_title("true state step: {}".format(ts))
#                        ax[1].imshow(true_playout_frame*0, origin='lower', vmin=0, vmax=255 )
#                        ax[1].set_title("rollout real num:{} step:{}".format(num,pf))
#                        ax[2].imshow(est_playout_frame*0, origin='lower', vmin=0, vmax=255 )
#                        ax[2].set_title("rollout model reward:{}".format(pf,round(score,2)))
#                        plt.savefig(os.path.join(fpath,fname))
#                        plt.close()
# 
#                    pf += 1
#                    fname = 'seed_%06d_tstep_%04d_pnum_%04d_pstep_%04d.png'%(seed, ts, num, pf)
#                    f,ax=plt.subplots(1,3, figsize=(10,3.5))
#                    ax[0].imshow(actual_frame, origin='lower', vmin=0, vmax=255 )
#                    ax[0].set_title("true state step: {}".format(ts))
#                    ax[1].imshow(true_playout_frame, origin='lower', vmin=0, vmax=255 )
#                    ax[1].set_title("sample rollout step:{}".format(pf))
#                    ax[2].imshow(est_playout_frame, origin='lower', vmin=0, vmax=255 )
#                    ax[2].set_title("sample model reward:{}".format(round(score,2)))
#                    plt.savefig(os.path.join(fpath,fname))
#                    plt.close()
#        gif_path = os.path.join(fpath, 'seed_{}_reward_{}.gif'.format(seed, int(reward)))
#        search = os.path.join(fpath, 'seed_*.png')
#        cmd = 'convert %s %s'%(search, gif_path)
#        os.system(cmd)
#
#    except Exception, e:
#        print(e)
#        embed()

    return results


if __name__ == "__main__":
    import argparse
    # this seems to work well
    #python roadway_pmcts.py -y 25 -x 25 --seed 45 -r 100  -p 100 -l 6

    default_base_datadir = '../gym_trajectories/saved/'
    default_model_savepath = os.path.join(default_base_datadir, 'cars_only_train.pkl')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=10, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=40, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=40, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=6, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=100, help='limit number of steps taken be random rollout')
    parser.add_argument('-m', '--model_loadpath', type=str, default=default_model_savepath)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='print debug info')

    args = parser.parse_args()
    use_cuda = args.cuda
    seed = args.seed
    num_z = 32
    nr_logistic_mix = 10

    if use_cuda:
        print("using gpu")
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=num_z).cuda()

    else:
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=num_z)
    opt = torch.optim.Adam(vmodel.parameters(), lr=1e-3)
    epoch = 0
    if os.path.exists(args.model_loadpath):
        vqvae_model_dict = torch.load(args.model_loadpath, map_location=lambda storage, loc: storage)
        vmodel.load_state_dict(vqvae_model_dict['state_dict'])
        vmodel.load_state_dict(vqvae_model_dict['state_dict'])
        opt.load_state_dict(vqvae_model_dict['optimizer'])
        epoch = vqvae_model_dict['epoch']
        print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
    else:
        print('could not find checkpoint at {}'.format(args.model_loadpath))
        sys.exit()

    goal_dis = args.max_goal_distance
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    all_results = []
    for i in range(args.num_episodes):
        r = run_trace(seed=seed, ysize=args.ysize, xsize=args.xsize, level=args.level,
                      max_goal_distance=goal_dis, n_playouts=args.num_playouts, max_rollout_length=args.rollout_steps)

        seed +=1
        all_results.append(r)
    print("FINISHED")
    embed()


