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
    best_angles[:len(env.action_space)/2] = 3
    best_angles[0] = 3.5
    best_angles = best_angles/float(best_angles.sum())

    unsorted_actions_and_probs = list(zip(best_actions, best_angles))
    actions_and_probs = sorted(unsorted_actions_and_probs, key=lambda tup: tup[0])

    print('bearing', bearing)
    print(actions_and_distances)
    print(actions_and_probs)
    return actions_and_probs

def equal_node_probs_fn(state, state_index, env):
    probs = np.ones(len(env.action_space))/float(len(env.action_space))
    actions_and_probs = list(zip(env.action_space, probs))
    return actions_and_probs

def get_vq_from_road(road_state):
    road_state = Variable(transforms.ToTensor()(road_state[:,:,None].astype(np.float32)))
    x_d, z_e_x, z_q_x, latents = vmodel(road_state[None])
    vroad_state = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    vroad_state = vroad_state[0,0].data.numpy() #.astype(np.uint8)
    vroad_state = (vroad_state*255).astype(np.uint8)
    return vroad_state


class PMCTS(object):
    def __init__(self, env, random_state, node_probs_fn, c_puct=1.4, n_playouts=1000, rollout_length=300):
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

    def get_children(self, node):
        print('node name', node.name)
        for i in node.children_.keys():
            print(node.children_[i].__dict__)
        return [node.children_[i].__dict__ for i in node.children_.keys()]

    def playout(self, playout_num, state, state_index):
        # set new root of MCTS (we've taken a step in the real game)
        # only sets robot and goal states
        finished,value = self.env.set_state(state, state_index)
        logging.debug('+++++++++++++START PLAYOUT NUM: {} FOR STATE: {}++++++++++++++'.format(playout_num,state_index))
        init_state = state
        init_state_index = state_index
        node = self.root
        actions = []
        won = False
 
        state = [state[0], state[1], get_vq_from_road(state[2])]
        while True:
            rs = self.env.get_robot_state(state)

            if node.is_leaf():
                if not finished:

                    logging.debug('PLAYOUT INIT STATE {}: expanding leaf at state {} robot: {}'.format(init_state_index, state_index, rs))
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action
                    actions_and_probs = self.node_probs_fn(state, state_index, self.env)
                    node.expand(actions_and_probs)
                    # if you have a neural network - use it here to bootstrap the value
                    # otherwise, playout till the end
                    # rollout one randomly
                    value, rand_actions, end_state, end_state_index = self.rollout_from_state(state, state_index)
                    actions += rand_actions
                    finished = True
                else:
                    end_state_index = state_index
                    end_state = state
                # finished the rollout
                node.update(value)
                actions.append(value)
                if value > 0:
                    node.n_wins+=1
                    won = True
                    logging.debug('won one with value:{} actions:{}'.format(value, actions))
                return won
            else:
                # greedy select
                # trys actions based on c_puct
                action, new_node = node.get_best(self.c_puct)
                actions.append(action)
                next_state, value, finished, _ = self.env.step(state, state_index, action)
                # time step
                state_index +=1
                state = [next_state[0], next_state[1], get_vq_from_road(next_state[2])]
                node = new_node

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
        init_state = state
        init_index = state_index
        rollout_actions = []
        rollout_states = []
        rollout_robot_positions = []

        try:
            finished,value = self.env.set_state(state, state_index)
            if finished:
                return value, rollout_actions, state, state_index
            c = 0
            while not finished:
                if c < self.rollout_length:
                    rs = self.env.get_robot_state(state)
                    a, action_probs = self.get_rollout_action(state)
                    rollout_robot_positions.append(rs)
                    rollout_states.append(state)
                    rollout_actions.append(a)
                    self.env.set_state(state)
                    next_state, reward, finished,_ = self.env.step(state, state_index, a)
                    state = [next_state[0], next_state[1], get_vq_from_road(next_state[2])]
                    state_index+=1
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
        return value, rollout_actions, state, state_index


    def get_action_probs(self, state, state_index, temp=1e-2):
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0

        finished,value = self.env.set_state(state, state_index)
        if not finished:
            for n in range(self.n_playouts):
                from_state = deepcopy(state)
                from_state_index = deepcopy(state_index)
                won+=self.playout(n, from_state, from_state_index)
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

    def get_best_action(self, state, state_index):
        logging.info("mcts starting search for action in state: {}".format(state_index))
        orig_state = deepcopy(state)
        self.env.set_state(state, state_index)
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
                n_playouts=n_playouts,rollout_length=max_rollout_length)

    t = 0
    finished = False
    # draw initial state
    true_env.render(state,t)
    print("SEED", seed)
    while not finished:
        states.append(state)
        ry,rx = true_env.get_robot_state(state)
        current_goal_distance = true_env.get_distance_to_goal()

        # search for best action
        st = time.time()
        action, action_probs = pmcts.get_best_action(deepcopy(state), t)
        et = time.time()

        next_state, reward, finished, _ = true_env.step(state, t, action)

        results['decision_ts'].append(et-st)
        results['dis_to_goal'].append(current_goal_distance)
        results['actions'].append(action)
        if not finished:
            pmcts.update_tree_move(action)
            #pmcts.reset_tree()
            state = next_state
            t+=1
        else:
            results['reward'] = reward
            states.append(next_state)
        true_env.render(next_state, t)
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

    time.sleep(1)
    true_env.close_plot()
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
    parser.add_argument('-l', '--level', type=int, default=4, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=100, help='limit number of steps taken be random rollout')
    parser.add_argument('-m', '--model_loadpath', type=str, default=default_model_savepath)
    parser.add_argument('-c', '--cuda', action='store_true', default=False)

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
    logging.basicConfig(level=logging.DEBUG)

    all_results = []
    for i in range(args.num_episodes):
        r = run_trace(seed=seed, ysize=args.ysize, xsize=args.xsize, level=args.level,
                      max_goal_distance=goal_dis, n_playouts=args.num_playouts, max_rollout_length=args.rollout_steps)

        seed +=1
        all_results.append(r)
    print("FINISHED")
    embed()



#    test_data = data_train_loader
#    if use_cuda:
#        x_test = Variable(test_data).cuda()
#    else:
#        x_test = Variable(test_data)


#def test(x,model,nr_logistic_mix,do_use_cuda=False,save_img_path=None):
#    x_d, z_e_x, z_q_x, latents = model(x)
#    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
#    loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,use_cuda=do_use_cuda)
#    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
#    loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
#    test_loss = to_scalar([loss_1, loss_2, loss_3])
#
#    if save_img_path is not None:
#        idx = np.random.randint(0, len(test_data))
#        x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
#        images = x_cat.cpu().data
#        oo = 0.5*np.array(x_tilde.cpu().data)[0,0]+0.5
#        ii = np.array(x.cpu().data)[0,0]
#        imwrite(save_img_path, oo)
#        imwrite(save_img_path.replace('.png', 'orig.png'), ii)
#    return test_loss


#
#    def get_mcts_action(self, state, ucb_weight=0.5):
#        init_y, init_x = self.env.robot.y, self.env.robot.x
#
#        actions = [(action_angle,action_speed) for action_angle in self.env.action_space[0] for action_speed in self.env.action_space[1]]
#        move_states = []
#        sdata = transforms.ToTensor()(state[:,:,None].astype(np.float32))
#        tstate = Variable(sdata)
#        x_tilde, z_e_x, z_q_x, latents = self.mcts_model['rep_model'](tstate[None])
#        # x_tilde would need to be sampled if we are going to use it because it
#        # is a mixture distribution in the 1th channel
#        # z_e_x is the output of the encoder
#        # z_q_x is the input into the decoder
#        # latents is the code book
#        # latents = latents.data.numpy()[0]
#        state_input = z_q_x.contiguous().view(z_q_x.shape[0],-1)
#
#        action_probs, state_value = self.mcts_model['zero_model'](state_input)
#
#
#        #for action in actions:
#        #    # reset robot back to starting position
#        #    # first step
#        #    self.env.robot.y = init_y
#        #    self.env.robot.x = init_x
#        #    self.env.steps = 0
#        #    action = [action_angle, action_speed]
#        #    next_state, reward, finished, _ = self.env.step(action)
#        #    move_states.append(next_state)




