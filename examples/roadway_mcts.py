# Author: Kyle Kastner & Johanna Hansen
# License: BSD 3-Clause
# http://mcts.ai/pubs/mcts-survey-master.pdf
# https://github.com/junxiaosong/AlphaZero_Gomoku

from gym_trajectories.envs.road import RoadEnv
import time
import numpy as np
from IPython import embed
from copy import deepcopy
import logging 

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
    def __init__(self, env, random_state, c_uct=1.4, n_playouts=1000, rollout_length=300):
        self.env = env
        self.rdn = random_state
        self.root = TreeNode(None,  name=(0,-1))
        self.c_uct = c_uct
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
        while True:
            rs = self.env.get_robot_state(state)
            if node.is_leaf():
                if not finished:
                    logging.debug('PLAYOUT INIT STATE {}: expanding leaf at state {} robot: {}'.format(init_state_index, state_index, rs))
                    # add all unexpanded action nodes and initialize them
                    # assign equal action to each action
                    probs = np.ones(len(self.env.action_space))/float(len(self.env.action_space))
                    actions_and_probs = list(zip(self.env.action_space, probs))
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
                # trys actions based on c_uct
                action, new_node = node.get_best(self.c_uct)
                actions.append(action)
                next_state, value, finished, _ = self.env.step(state, state_index, action)
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
        logging.debug('-------------------------------------------')
        logging.debug('starting random rollout from state: {}'.format(state_index))
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
                    state, reward, finished,_ = self.env.step(state, state_index, a)
                    state_index+=1
                    c+=1
                    if finished:
                        logging.debug('finished rollout after {} steps with value {}'.format(c,value))
                        value = reward
                else:
                    # stop early
                    value = self.env.get_timeout_reward(state_index)
                    logging.debug('stopping rollout after {} steps with value {}'.format(c,value))
                    finished = True

            logging.debug('-------------------------------------------')
        except Exception, e:
            print(e)
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
        else:
            logging.info("GIVEN STATE WHICH WILL DIE - state index {} max env {}".format(state_index, self.env.max_steps))
            #embed()
            
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
        self.root = TreeNode(None)
        self.tree_subs_ = []

def run_trace(seed=3432, ysize=40, xsize=40, level=5, max_goal_distance=100, 
              n_playouts=300, max_rollout_length=50):

    # log params
    results = {'decision_ts':[], 'dis_to_goal':[], 'actions':[], 
               'ysize':ysize, 'xsize':xsize, 'level':level, 
               'n_playouts':n_playouts, 'seed':seed,
               'max_rollout_length':max_rollout_length}

    # restart at same position every time
    rdn = np.random.RandomState(seed)
    true_env = RoadEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level)
    state = true_env.reset(max_goal_distance)

    mcts_rdn = np.random.RandomState(seed+1)
    mcts = MCTS(env=deepcopy(true_env),random_state=mcts_rdn,
                n_playouts=n_playouts,rollout_length=max_rollout_length)

    t = 0
    finished = False
    # draw initial state
    true_env.render(state,t)
    while not finished:
        ry,rx = true_env.get_robot_state(state)
        current_goal_distance = true_env.get_distance_to_goal()

        # search for best action
        st = time.time()
        action, action_probs = mcts.get_best_action(deepcopy(state), t)
        #mcts.reset_tree()
        mcts.update_tree_move(action)
        et = time.time()

        next_state, reward, finished, _ = true_env.step(state, t, action)
        true_env.render(next_state,t)

        results['decision_ts'].append(et-st)
        results['dis_to_goal'].append(current_goal_distance)
        results['actions'].append(action)
        if not finished:
            state = next_state
            t+=1
        else:
            results['reward'] = reward
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=35, help='random seed to start with')
    parser.add_argument('-e', '--num_episodes', type=int, default=10, help='num traces to run')
    parser.add_argument('-y', '--ysize', type=int, default=40, help='pixel size of game in y direction')
    parser.add_argument('-x', '--xsize', type=int, default=40, help='pixel size of game in x direction')
    parser.add_argument('-g', '--max_goal_distance', type=int, default=1000, help='limit goal distance to within this many pixels of the agent')
    parser.add_argument('-l', '--level', type=int, default=4, help='game playout level. level 0--> no cars, level 10-->nearly all cars')
    parser.add_argument('-p', '--num_playouts', type=int, default=200, help='number of playouts for each step')
    parser.add_argument('-r', '--rollout_steps', type=int, default=100, help='limit number of steps taken be random rollout')

    args = parser.parse_args()
    seed = args.seed
    goal_dis = args.max_goal_distance
    logging.basicConfig(level=logging.INFO)

    all_results = []
    for i in range(args.num_episodes):
        r = run_trace(seed=seed, ysize=args.ysize, xsize=args.xsize, level=args.level,
                      max_goal_distance=goal_dis, n_playouts=args.num_playouts, max_rollout_length=args.rollout_steps)

        seed +=1
        all_results.append(r)
    print("FINISHED")
    embed()
