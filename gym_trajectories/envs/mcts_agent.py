import matplotlib
matplotlib.use('TkAgg')
import sys
import os
from subprocess import Popen
import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import logging
import math
from imageio import imwrite
import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from vq_vae import AutoEncoder, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
from imageio import imread, imwrite
from PIL import Image
from collections import defaultdict, deque

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.u = 0.0
        self.P = prior_p
        self.Q = 0.0

    def expand(self, actions, probs):
        ''' expand to create new children. 
        action priors is the output from the policy function list '''
        for action, prob in zip(action,probs):
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def is_leaf(self):
        return self.children == {}

    def is_roof(self):
        return self.parent is None

    def update(self, value):
        self.n_visits+=1
        # instead of tracking W directly, we update Q only based on 
        # code from  junxiaosong/AlphaZero_Gomoku
        self.Q += (value-self.Q)/float(self.n_visits) 

    def update(self, leaf_value):
        ''' update node values from leaf evaluation 
        leaf_value is the value of subtree evaluation '''
        if self.parent is not None:
            self.parent.update(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        ''' calculate and return value for this node
        c_puct is a number in (0,inf) which controls the impact of values Q and prior P 
        on this nodes score'''
        U = c_puct * self.P * np.sqrt(self.parent.n_visits)/(self.n_visits+1)
        return self.Q+U

    def get_best(self, c_puct):
        best = max(self.children.iteritems(), key=lambda x: x[1].get_value(c_puct))
        return best


def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x-np.max(x))
    probs/=np.sum(probs)
    return probs

class NetMCTS(object):
    def __init__(self, policy_value_fn, env, rdn,
                       c_puct=1.4, n_playouts=1000, 
                       ):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.root = TreeNode(None,1.0)
        self.env = env
 
    def playout(self, state):
        node = self.root
        states = []
        while True:
            if node.is_leaf():
                finished = self.env.robot.alive()
                if not finished:
                    # forwards on network
                    actions_and_probs, value = self.mcts_model(state)
                    node.expand(actions, probs)
                else:
                    # determine scoring
                    value = end
                node.update(value)
                return value
            else:
                action,node = node.get_best(self.c_puct)
                state = self.state_manager.next_state(state, action)

    def get_move_probs(self, state, temp=1e-3):
        mgr = deepcopy(self.state_manager)
        for n in range(self.n_playout):
            self.playout(state)
            self.state_manager = deepcopy(mgr)
        act_visits = [(act,node.n_visits) for act, node in self.root.childrent.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1./temp*np.log(visits))
        return act, act_probs


    def get_action(self, state, temp=1e-3, add_noise=True, dirichlet_coeff1=0.25, dirichlet_coeff2=0.3):
        vsz = len(self.state_manager.valid_actions(state))
        move_probs = np.zeros((vsz,))
        acts, probs = self.get_move_probs(state, temp)
        move_probs[list(acts)] = probs
        if add_noise:
            dirichlet_probs=(1.0-dirichlet_coeff1)*probs+dirichlet_coeff1*self.rdn.dirichlet(dirichlet_coeff2*np.ones(len(probs)))
            move = self.rdn.choice(acts, p=dirichlet_probs)
        else:
            move = self.rnd.choice(acts, p=probs)
        return move, move_probs

    def update_to_move(self, move):
        # if you pass -1 then ... whole tree .....
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            print("move argument to update_to_move not in actions {}".format(move, self.root.children.keys())) 
            self.root = TreeNode(None, 1.0)




def policy_goto_goal(state):
    goal = state[0]
    agent = state[1]
    goal_y, goal_x = goal[0], goal[1]
    agent_y, agent_x = agent[0], agent[1]
    dy = goal_y-self.env.robot.y
    dx = goal_x-self.env.robot.x
    goal_angle = np.rad2deg(math.atan2(dy,dx))
    return [goal_angle, 1.0]


def get_trace(rdn, env, policy_value):
    # initial state
    # state consists of [agentyx, goalyx, pixelspace]
    state = env.reset()
    # TODO - not sure if this should be deepcopied
    mcts = NetMCTS(policy_value_fn, deepcopy(env), random_state=rdn)
    env_rdn = env.rdn
    temp = 1.0
    state, moves, mcts_probs = [], [], []
    steps = 0
    finished = False
    total_reward = 0
    while not finished:
        move, move_probs = mcts.get_action(env, state)
        # store real states/moves
        states.append(state)
        moves.append(move)
        mcts_probs.append(move_probs)
        # take actual step 
        next_state, reward, finished, _ = self.env.step(move)
        total_reward+=reward
        if not finished:
            steps +=1
            state = next_state
        else:
            if total_reward > 0:
                reached_goal = True
            else:
                reached_goal = False
            # add last state
            states.append(next_state)
    # mcts.make_full_sequence
    # states is one longer ----- 
    return states, mcts_probs, moves, total_reward, reached_goal
    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    default_base_datadir = 'saved'
    default_vqvae_savepath = os.path.join(default_base_datadir, 'frogger_model_40x40_level3.pkl')
    default_policy_savepath = os.path.join(default_base_datadir, 'policy_1.pkl')


    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-v', '--vqvae_loadpath', default=default_vqvae_savepath)
    parser.add_argument('-z', '--num_z', type=np.int, default=16)
    parser.add_argument('-l', '--model_loadpath', default=default_policy_savepath)
    parser.add_argument('-s', '--model_savepath', default=default_policy_savepath)
    parser.add_argument('-r', '--random_seed', type=np.int, default=233)
    parser.add_argument('-n', '--num_traces', type=np.int, default=1)
    args = parser.parse_args()
    use_cuda = args.cuda

    local_rdn = np.random.RandomState(args.seed)
    if use_cuda:
        vqvae_model = AutoEncoder(encoder_output_size=args.num_z).cuda()
    else:
        vqvae_model = AutoEncoder(encoder_output_size=args.num_z)
 
    if not os.path.exists(args.vqvae_loadpath):
        print('print _model path not found: {}'.format(args.vqvae_loadpath))
        sys.exit()
    else:
        vqvae_model_dict = torch.load(args.vqvae_loadpath, map_location=lambda storage, loc: storage)
        vqvae_model.load_state_dict(vqvae_model_dict['state_dict'])
        
        print('loaded checkpoint at epoch: {} from {}'.format(vqvae_model_dict['epoch'], args.vqvae_loadpath))

    room_env = RoomEnv(obstacle_type='frogger',seed=args.seed+233)
    positive_trace_data = []
    negative_trace_data = []
    for n in num_traces:
        # states, mcts_probs, moves, total_reward, reached_goal
        trace = get_trace(local_rdn, room_env, policy_goto_goal)
        if trace[-1]:
            positive_trace_data.append(trace[:-1])
        else:
            negative_trace_data.append(trace[:-1])















    #ba = BaseAgent(do_plot=False, train=True, n_episodes=5, seed=415)
    #ba.run()
#        if self.do_make_gif:
#            ne = int(self.episodes-1)
#            this_gif_path = self.gif_path %(ne, self.steps)
#            logging.info("starting gif creation for episode:{} file:{}".format(ne, this_gif_path))
#            search = os.path.join(self.img_path, 'episode_%06d_frame_*.png' %ne)
#            cmd = 'convert %s %s'%(search, this_gif_path)
#            # make gif
#            Popen(cmd.split(' '))
#
#    def get_goal_action(self, state):
#        goal_y, goal_x = self.env.goal.y, self.env.goal.x
#        dy = goal_y-self.env.robot.y
#        dx = goal_x-self.env.robot.x
#        goal_angle = np.rad2deg(math.atan2(dy,dx))
#        return [goal_angle, 1.0]
#
#    def get_random_action(self, state):
#        random_angle = self.rdn.choice(self.env.action_space[0] )
#        random_speed = self.rdn.choice(self.env.action_space[1] )
#        return [random_angle, random_speed]
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


#class Manager():
#    def __init__(self, do_plot=False, do_save_figs=False, save_fig_every=1,
#                       do_make_gif=False, save_path='saved', n_episodes=10, 
#                       seed=133, train=False):
#
#        self.agent = MCTSAgent()
#        self.do_plot = do_plot
#        self.do_save_figs = do_save_figs
#        self.save_fig_every=save_fig_every
#        self.save_path = save_path
#        self.do_make_gif = do_make_gif
#
#        if self.do_save_figs:
#            if train:
#                self.img_path = os.path.join(self.save_path, 'imgs_train') 
#            else:
#                self.img_path = os.path.join(self.save_path, 'imgs_test') 
#
#            if not os.path.exists(self.img_path):
#                os.makedirs(self.img_path)
#            self.plot_path = os.path.join(self.img_path, 'episode_%06d_frame_%05d_reward_%d.png')
#            if self.do_make_gif:
#                self.gif_path = os.path.join(self.img_path, 'episode_%06d_frames_%05d.gif')
#        # get last img epsidoe - in future should look for last in model or something
#        try:
#            past = glob(os.path.join(self.img_path, 'episode_*.png'))
#            # get name of episode
#            eps = [int(os.path.split(p)[1].split('_')[1]) for p in past]
#            last_episode = max(eps)
#        except Exception, e:
#            last_episode = 0
#        self.episodes = last_episode
#        # note - need new seed for randomness
#        ep_seed = seed+last_episode
#        print('-------------------------------------------------------------------------')
#        print('starting from episode: {} with seed {}'.format(last_episode, ep_seed))
#        raw_input('press any key to continue or ctrl+c to exit')
#        self.rdn = np.random.RandomState(ep_seed)
#        self.env = RoomEnv(obstacle_type='frogger',seed=ep_seed+1)
#        self.c_puct = 5
#        self.temp = 1.0
#        # num of simulations for each move
#        self.n_playout = 400
#        self.batch_size = 128
#        self.data_buffer =  deque(maxlen=10000)
# 
#    def run(self):
#        try:
#            game_batch_num = 1000
#            play_batch_size = 1
#            for i in range(n_games_batch):
#                self.collect_data(play_batch_size)
#                if len(self.data_buffer) > self.batch_size:
#                    loss, entropy = self.update_policy_model()
#                    if i%10:
#                        # TODO check performance
#                        self.save_policy_model()
#        except Exception, e:
#            print("Exception {}".format(e))
#            embed()
#
#    def collect_data(self, n_episodes):
#        for i in range(n_episodes):
#            reward, play_data = self.run_episode()
#            self.episode_length = len(play_data)
#            self.data_buffer.extend(play_data)
#
#    def run_episode(self):
#        print("starting episode {}".format(self.episodes))
#        self.steps = 0
#        state = self.env.reset()
#        finished = False
#        states, mcts_probs = [], []
#        while not finished:
#            action, action_probs = self.agent.get_action(deepcopy(self.env), temp, return_prob=1)
#            states.append(state)
#            mcts_probs.append(action_probs)
#            next_state, reward, finished, _ = self.env.step(action)
#            signal = 1 # one step
#            if finished: # episode over
#                if reward > 0:
#                    signal = 2 # won
#                else:
#                    signal = 0 # died
#            if self.do_plot:
#                self.env.render()
#
#            if self.do_save_figs:
#                if not self.steps%self.save_fig_every:
#                    this_plot_path = self.plot_path %(self.episodes, self.steps, signal)
#                    imwrite(this_plot_path, state)
#            if finished:
#                print("end game after {} steps- reward {}".format(self.steps,reward))
#                return reward, zip(states, mcts_probs)
#
#            else:
#                state = next_state
#                self.steps+=1
#
#        self.episodes+=1


