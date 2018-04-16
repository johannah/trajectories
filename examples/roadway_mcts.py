# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

# Key modifications from base MCTS described in survey paper
# use PUCT instead of base UCT
# Expand() expands *all* children, but only does rollout on 1 of them
# Selection will naturally bias toward the unexplored nodes
# so the "fresh" children will quickly be explored
# This is to closer match AlphaGo Zero, see appendix of the Nature paper

from gym_trajectories.envs.road import RoadEnv
import time
import numpy as np
from IPython import embed
from copy import deepcopy
def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

## simple state machine, counting
#class Env(object):
#    def __init__(self, size=5, seed=334):
#        self.size = size
#        self.action_space = tuple(range(self.size))
#        self.rdn = np.random.RandomState(seed)
#        self.states = range(size)
#
#    def reset(self, start_state_index=0):
#        # return state
#        return self.states[start_state_index]
#
#    def step(self, state_index, action):
#        print(self.states)
#        print(state_index, action)
#        # step to state index with action
#        # return next_state, reward, finished, _
#        finished = False
#        state = self.states[state_index]
#        if action == state:
#            # if you choose action == state, progress, else stuck
#            next_state_index = state_index+1
#            next_state = self.states[next_state_index]
#            reward = +1
#            if next_state == self.size-1:
#                finished = True
#                reward += self.size + 10
#        else:
#            # lose
#            next_state_index = 0
#            next_state = self.states[next_state_index]
#            finished = True
#            reward = -self.size + -10
#
#        return next_state, reward, finished, {}

            
def perfect_policy_fn(state, valid_actions):
    action_probs = np.zeros_like(valid_actions).astype(np.float)
    action_probs[state] = 1.0
    comb = tuple(zip(valid_actions, action_probs))
    return comb, 0

class TreeNode(object):
    def __init__(self, parent, name='unk'):
        self.name = name
        self.parent = parent
        self.W_ = 0.0
        # action -> tree node
        self.children_ = {}
        self.n_visits_ = 0
        self.past_actions = []
        self.n_wins = 0

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                child_name = (self.name[0],action)
                self.children_[action] = TreeNode(self, name=child_name)

    def is_leaf(self):
        return self.children_ == {}

    def is_root(self):
        return self.parent is None

    def _update(self, value):
        self.n_visits_ += 1
        self.W_ += value

    def update(self, value):
        if self.parent != None:
            self.parent.update(value)
        self._update(value)

    def get_value(self, c_uct):
        if self.n_visits_ == 0:
            lp=0.0
            rp = np.inf
        else:
            lp = self.W_/float(self.n_visits_)
            rp = c_uct*np.sqrt(2*np.log(self.parent.n_visits_)/float(self.n_visits_))
        return lp+rp

    def get_best(self, c_uct):
        best = max(self.children_.iteritems(), key=lambda x: x[1].get_value(c_uct))
        return best


class MCTS(object):
    def __init__(self, 
                       random_state, c_uct=1.4, n_playouts=1000):
        self.rdn = random_state
        self.root = TreeNode(None,  name=(0,-1))
        self.c_uct = c_uct
        self.n_playouts = n_playouts
        self.tree_subs_ = []
        self.warn_at_tree_size = 1000
        self.tree_subs_ = []
        self.step = 0
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
        print('+++++++++++++START PLAYOUT{}++++++++++++++++'.format(playout_num))
        print(self.env.get_robot_state(), state[:2], state_index)
        init_state = state
        init_state_index = state_index
        node = self.root
        state_indexes = [state_index]
        actions = []
        self.playout_end_states = []
        self.playout_actions = []
        self.playout_rrs = []
        won = False
        while True:
            #print(".... playout state:{} nodename{}".format(state_index, node.name))
            rs = self.env.get_robot_state()
            self.playout_rrs.append(rs)

            if node.is_leaf():
                if not finished:
                    print('expanding leaf at state {} robot: {}'.format(state_index, rs))
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action
                    probs = np.ones(len(self.env.action_space))/float(len(self.env.action_space))
                    actions_and_probs = list(zip(self.env.action_space, probs))
                    node.expand(actions_and_probs)
                    # if you have a neural network - use it here to bootstrap the value
                    # otherwise, playout till the end
                    # rollout one randomly
                    # _____ THIS MIGHT BE WRONG
                    value, rand_actions, end_state, end_state_index = self.rollout_from_state(state, state_index)
                    actions += rand_actions
                    finished = True 
                else:
                    end_state_index = state_index
                    end_state = state
                #print("-----finished from init_state_index {}".format(init_state_index))
                #print("FINISHED value {} actions {}".format(value, actions))
                #self.playout_end_states.append(end_state)
                #self.playout_actions.append(actions)
                # finished the rollout 
                node.update(value)
                actions.append(value)
                if value>0:
                    node.n_wins+=1
                    won = True
                #node.past_actions.append(actions)

                if value == 0:
                    print("VALUE 0")
                    raise
                if value > 0:
                    print('won one with value:{} actions:{}'.format(value, actions))
                if abs(value) == 1:
                    print("VALUE 1")
                    embed()
                return won
            else:
                # greedy select
                # trys actions based on c_uct
                # COULD THIS BE FKD?
                action, new_node = node.get_best(self.c_uct)
                actions.append(action)
                # reward should be zero unless terminal. it should never be terminal here
                next_state, value, finished, _ = self.env.step(state, state_index, action)
                #print("NONLEAF state_index {} action {} finished {} reward {}".format(state_index,action,finished,value))
           
                
                # SO APPARENTLY IT IS NEVER SUPPOSED TO END HERE WHICH SEEMS UNREASONABLE TO ME
                # MINE TOTALLY ENDS HERE 
                state_indexes.append(state_index)
                # time step
                state_index +=1
                state = next_state
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
        print('-------------------------------------------')
        print('starting random rollout from state: {}'.format(state_index))
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
                if c < 10:
                    rs = self.env.get_robot_state()
                    a, action_probs = self.get_rollout_action(state)
                    rollout_robot_positions.append(rs)
                    rollout_states.append(state)
                    rollout_actions.append(a)
                    state, reward, finished,_ = self.env.step(state, state_index, a)
                    state_index+=1
                    c+=1
                    if finished:
                        print('finished rollout after {} steps with value {}'.format(c,value))
                        value = reward
                else:
                    # stop early
                    value = self.env.get_lose_reward(state_index)
                    print('stopping rollout after {} steps with value {}'.format(c,value))
                    finished = True

            print('-------------------------------------------')
        except Exception, e:
            print(e)
            raise
            embed()
        return value, rollout_actions, state, state_index


    def get_action_probs(self, state, state_index, temp=1e-3):
        # low temp -->> argmax
        self.nodes_seen[state_index] = []
        won = 0

        finished,value = self.env.set_state(state, state_index)
        if not finished:
            for n in range(self.n_playouts): 
                from_state = deepcopy(state)
                from_state_index = deepcopy(state_index)
                won+=self.playout(n, from_state, from_state_index)
            print("NUMBER WON", won)
        else:
            print("GIVEN STATE WHICH WILL DIE")
            embed()
            
        self.env.set_state(state, state_index)
        act_visits = [(act,node.n_visits_) for act, node in self.root.children_.items()]
        actions, visits = zip(*act_visits)
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
        orig_state = deepcopy(state)
        self.env.set_state(state, state_index)
        acts, probs = self.get_action_probs(state, state_index, temp=1e-3)
        #act_probs = np.zeros_like((self.env.action_space)).astype(np.float)
        #act_probs[list(acts)] = probs
        #maxes = np.max(act_probs)
        #opts = np.where(act_probs == maxes)[0]
        #if len(opts)>1:
        #    self.rdn.shuffle(opts)
        #act = opts[0]
        act = self.rdn.choice(acts, p=probs)
        return act, probs

    def update_tree_move(self, action):
        # keep previous info
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_tree_size:
                print("WARNING: over {} tree_subs_ detected".format(len(self.tree_subs_)))
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            print("Move argument {} to update_to_move not in actions {}, resetting".format(action, self.root.children_.keys()))

    def reset_tree(self):
        print("Resetting tree")
        self.root = TreeNode(None)
        self.tree_subs_ = []

def run_trace(max_goal_distance=100):
    rrs = []
    states_trace = []
    actions_trace = []
    finished = False
    # restart at same position every time
    state = true_env.reset(max_goal_distance)
    v = 0
    t = 0

    mcts = MCTS(rdn,n_playouts=500)
    mcts.env = deepcopy(true_env)
    true_env.render(state,t)
    while not finished:
        ry,rx = true_env.get_robot_state()
        rrs.append((ry,rx))
        action, action_probs = mcts.get_best_action(state, t)
        next_state, reward, finished, _ = true_env.step(state, t, action)
        mcts.update_tree_move(action)
        v+=reward
        states_trace.append(state)
        actions_trace.append(action)
        true_env.render(next_state,t)
        if not finished:
            state = next_state
            t+=1
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print(rrs) 
    print(actions_trace)
    if v>0:
        print("robot won after {} steps".format(t))
    else:
        print("robot died after {} steps".format(t))
        embed()

    print(v)
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")

    return states_trace, actions_trace, v


if __name__ == "__main__":
    ss = []
    aa = []
    rr = []
    true_env = RoadEnv(ysize=40,xsize=20, level=4)
    rdn = np.random.RandomState(343)
    goal_dis = 15
    for i in range(10):
        s, a, v = run_trace(goal_dis)
        goal_dis+=2
        ss.append(s)
        aa.append(a)
        rr.append(v)
        time.sleep(1)
        true_env.close_plot()
    print("FINISHED")
    embed()
