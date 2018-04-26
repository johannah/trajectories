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

import numpy as np
from IPython import embed
def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

# simple state machine, counting
class Env(object):
    def __init__(self, size=5, seed=334):
        self.size = size
        self.action_space = tuple(range(self.size))
        self.rdn = np.random.RandomState(seed)
        self.states = range(size)

    def reset(self, start_state_index=0):
        # return state
        return self.states[start_state_index]

    def step(self, state_index, action):
        print(self.states)
        print(state_index, action)
        # step to state index with action
        # return next_state, reward, finished, _
        finished = False
        state = self.states[state_index]
        if action == state:
            # if you choose action == state, progress, else stuck
            next_state_index = state_index+1
            next_state = self.states[next_state_index]
            reward = +1
            if next_state == self.size-1:
                finished = True
                reward += self.size + 10
        else:
            # lose
            next_state_index = 0
            next_state = self.states[next_state_index]
            finished = True
            reward = -self.size + -10

        return next_state, reward, finished, {}

            
def perfect_policy_fn(state, valid_actions):
    action_probs = np.zeros_like(valid_actions).astype(np.float)
    action_probs[state] = 1.0
    comb = tuple(zip(valid_actions, action_probs))
    return comb, 0

def random_policy_fn(state, valid_actions):
    action_probs = rdn.rand(len(valid_actions))
    action_probs = action_probs / np.sum(action_probs)
    return tuple(zip(valid_actions, action_probs)), 0

class TreeNode(object):
    def __init__(self, parent, prior_p=1, name='unk'):
        self.name = name
        self.parent = parent
        self.W_ = 0.0
        # action -> tree node
        self.children_ = {}
        self.n_visits_ = 0
        self.P_ = prior_p

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                self.children_[action] = TreeNode(self, prior_p=prob, name=action)

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
    def __init__(self, rollout_policy_fn,
                        env, random_state, c_uct=1.4, n_playouts=10):
        self.env = env
        self.rdn = random_state
        self.root = TreeNode(None, prior_p=1.0, name='root')

        self.rollout_fn = rollout_policy_fn
        self.c_uct = c_uct
        self.n_playouts = n_playouts
        self.tree_subs_ = []
        self.warn_at_tree_size = 1000
        self.tree_subs_ = []
        self.step = 0

    def playout(self, state_index):
        print("starting playout from {}".format(state_index))
        # set new root of MCTS (we've taken a step in the real game)
        state = self.env.reset(state_index)
        init_state = state
        init_state_index = state_index
        node = self.root
        finished = False
        value = 0
        state_indexes = [state_index]
        while not finished:
            print(".... playout state:{} value:{} nodename{}".format(state_index, value, node.name))
            if node.is_leaf():
                print('expanding leaf at state {}'.format(state))
                # add all unexpanded action nodes and initialize them
                # assign equal action to each action
                probs = np.ones(len(self.env.action_space))/float(len(self.env.action_space))
                actions_and_probs = list(zip(self.env.action_space, probs))
                node.expand(actions_and_probs)
                # if you have a neural network - use it here to bootstrap the value
                # otherwise, playout till the end
                # rollout one randomly
                value += self.rollout_from_state(state_index)
                print("-----rollout from state {} value {}".format(state_index, value))
                finished = True
            else:
                # greedy select
                # trys actions based on c_uct
                action, new_node = node.get_best(self.c_uct)
                next_state, reward, finished, _ = self.env.step(state_index, action)
                state_indexes.append(state_index)
                # time step
                state_index +=1
                value += reward
                print("playout state {} action BEST {} next_state {} reward {} value {} finished {}".format(state,action,next_state,reward,value,finished))
                state = next_state
                node = new_node
        node.update(value)

    def get_rollout_action(self, state):
        act_probs,_ = self.rollout_fn(state, self.env.action_space)
        acts, act_probs = zip(*act_probs)   
        act = self.rdn.choice(acts, p=act_probs)
        return act, act_probs


    def rollout_from_state(self, state_index):
        print('starting random rollout from state: {}'.format(state_index))
        c = 0
        state = self.env.reset(state_index)
        value = 0
        rollout_actions = []
        rollout_states = []
        finished = False
        while not finished:
            a, action_probs = mcts.get_rollout_action(state)
            rollout_states.append(state)
            rollout_actions.append(a)
            state, reward, finished,_ = self.env.step(state_index, a)
            state_index+=1
            value +=reward
            c+=1
            if c > 1000:
                break
        print('finished rollout', value)
        print(zip(rollout_states, rollout_actions))
        return value


    def get_action_probs(self, state_index, temp=1e-3):
        # low temp -->> argmax
        for n in range(self.n_playouts):
            self.playout(state_index)
        act_visits = [(act,node.n_visits_) for act, node in self.root.children_.items()]
        actions, visits = zip(*act_visits)
        action_probs = softmax(1.0/temp*np.log(visits))
        return actions, action_probs

    def evaluate(self, state_index, limit=1600):
        orig_state_index = state_index
        state = self.env.reset(state_index)
        states_trace = [state]
        value = 0
        finished = False
        for i in range(limit):
            if not finished:
                actions_and_probs,_ = self.rollout_fn(state, self.env.action_space)
                max_action = max(actions_and_probs, key=lambda x: x[1])[0]
                next_state, reward, finished, _ = self.env.step(state_index, max_action)
                state_index +=1
                value+=reward
                states_trace.append(next_state)
                state = next_state
            else:
                state_traces.append(state)
                break
        return value

    def get_best_action(self, state_index):
        acts, probs = self.get_action_probs(state_index, temp=1e-3)
        act_probs = np.zeros_like((self.env.action_space)).astype(np.float)
        act_probs[list(acts)] = probs
        maxes = np.max(act_probs)
        opts = np.where(act_probs == maxes)[0]
        if len(opts)>1:
            self.rdn.shuffle(opts)
        act = opts[0]
        return act, act_probs

    def update_tree_move(self, action):
        # keep previous info
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_tree_size:
                print("WARNING: over {} tree_subs_ detected".format(len(self.tree_subs_)))
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            print("Move argument {} to update_to_move not in actions {}, resetting".format(move, self.root.children_.keys()))
            embed()

def run_trace():
    states_trace = []
    actions_trace = []
    finished = False
    state = aenv.reset()
    true_env = Env(size=6)
    state = true_env.reset(0)
    v = 0
    t=0
    while not finished:

        action, action_probs = mcts.get_best_action(state)
        next_state, reward, finished, _ = true_env.step(t,action)
        t+=1
        v+=reward
        states_trace.append(state)
        actions_trace.append(action)
        mcts.update_tree_move(action)
        print('F:state:{} action:{} next_state:{}'.format(state, action, next_state))
        if not finished:
            state = next_state
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print(states_trace) 
    print(actions_trace)
    print(v)
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    print("_____________________________________________________________________")
    return states_trace, actions_trace, v


if __name__ == "__main__":
    rdn = np.random.RandomState(343)
    aenv = Env(size=6)
    mcts = MCTS(random_policy_fn, aenv, rdn)
    #mcts = MCTS(perfect_policy_fn, aenv, rdn)
    ss = []
    aa = []
    rr = []
    for i in range(1):
        s, a, v = run_trace()
        ss.append(s)
        aa.append(a)
        rr.append(v)
    embed()
