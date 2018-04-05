import matplotlib
matplotlib.use('TkAgg')
from glob import glob
import os
from subprocess import Popen
import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import logging
import math
from imageio import imwrite
lose_reward=-100
step_reward=-1
win_reward=100

class Particle():
    def __init__(self, world, name, local_map, init_y, init_x,
                 angle, speed, bounce=True, bounce_angle=0,
                 color=12, ymarkersize=3, xmarkersize=3):

        self.bounce_angle=bounce_angle
        self.steps = 0
        self.world = world
        self.name = name
        self.angle = angle
        self.speed = speed
        self.color = color
        self.bounce = bounce
        # always add positive number to size
        self.ymarkersize = abs(ymarkersize)
        self.xmarkersize = abs(xmarkersize)
        self.y = init_y
        self.x = init_x
        self.alive = True
        self.local_map = local_map
        self.step(0)

    def wall_bounce(self, hit_wall):
        if hit_wall:
            # some agents will bounce off of the walls, others should die
            if self.bounce:
                self.angle+=np.sign(self.angle)*self.bounce_angle
            else:
                self.alive = False

    def step(self, timestep):
        # (meters/second) * second
        # TODO angle
        rads = np.deg2rad(self.angle) 
        newy = self.y + self.speed*np.sin(rads)*timestep
        newx = self.x + self.speed*np.cos(rads)*timestep
        self.y,self.x,newyplus,newxplus = self.collide_with_walls(newy,newx)
        if self.alive:
            y = range(int(self.y), int(newyplus))
            x = range(int(self.x), int(newxplus))
            if self.name == 'robot':
                if not len(y):
                    y = [int(self.y)]
                if not len(x):
                    x = [int(self.x)]
                inds = np.array([(yy,xx) for yy in y for xx in x]).T
            inds = np.array([(yy,xx) for yy in y for xx in x]).T
            self.local_map[inds[0,:], inds[1,:]] = self.color
            self.steps +=1
        return self.alive

    def collide_with_walls(self,newy,newx):
        hit_wall = False
        # markersize is always positive so only need to check x/ysize
        newyplus = int(newy+self.ymarkersize)
        newxplus = int(newx+self.xmarkersize)
        # leading edge hit
        if (newyplus>= self.world.ysize-1):
            newyplus = self.world.ysize-1
        if (newxplus>= self.world.xsize-1):
            newxplus = self.world.xsize-1
        if (newy>=self.world.ysize-1):
            newy = float(self.world.ysize-1)
            hit_wall = True
        if (newx>=self.world.xsize-1):
            newx = float(self.world.xsize-1)
            hit_wall = True
        if (newyplus<= 0):
            newyplus = 0.0
        if (newxplus<= 0):
            newxplus = 0.0
        if (newy <= 0):
            newy = 0.0
            hit_wall = True
        if (newx <= 0):
            newx = 0.0
            hit_wall = True
        self.wall_bounce(hit_wall)
        return newy, newx, newyplus, newxplus

def check_state(ry, rx, room_map, goal_map):
   # if particle is able to collide with other agents
   if room_map[ry,rx].sum()>0:
       print('robot collided')
       return True, lose_reward
   elif goal_map[ry,rx].sum()>0:
       print('robot WON!!')
       return True, win_reward
   else:
       return False, step_reward



class RoomEnv():
    def __init__(self, obstacle_type='walkers', ysize=40, xsize=40, seed=10, 
                 timestep=1,collision_distance=2):
        # TODO - what if episode already exists in savedir
        self.steps = 0
        self.obstacle_type = obstacle_type
        self.ysize = ysize
        self.xsize = xsize
        self.timestep = timestep
        self.max_speed = 2.0
        # average speed
        self.average_speed = 1.0
        # make max steps twice the steps required to cross diagonally across the room
        self.max_steps = int(2*(np.sqrt(self.ysize**2 + self.xsize**2)/float(self.average_speed))/float(self.timestep))
        #      90
        #      | 
        # 180 --- 0 
        #      |
        #     360
        self.angles = np.linspace(0, 180, 5)[::-1]
        self.speeds = np.linspace(0,self.max_speed,4)
        self.action_space = [self.angles,self.speeds]
        self.collision_distance=collision_distance
        self.rdn = np.random.RandomState(seed)
        self.room_map = np.zeros((self.ysize, self.xsize), np.uint8)

    def get_data_from_fig(self):
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def configure_cars(self, max_xcarsize):
        self.car_ysize=1
        median1 = (self.ysize/2)-(self.safezone/2)
        median2 = (self.ysize/2)+(self.safezone/2)

        # make stripes around safe spaces
        self.human_markers = np.zeros_like(self.room_map)
        s = 0
        self.human_markers[median1,:] = s
        self.human_markers[median2,:] = s
        self.human_markers[self.safezone,:] = s
        self.human_markers[self.ysize-self.safezone,:] = s

        lanes = list(np.arange(self.safezone+1, median1-self.car_ysize, self.car_ysize))
        lanes.extend(list(np.arange(median2+1, self.ysize-self.safezone-self.car_ysize, self.car_ysize)))
        
        mx = max(max_xcarsize,5)
        cars = {
                'truck':{'color':45, 'speed':np.linspace(.2,.7,10),    'xsize':mx, 'angles':[], 'lanes':[]},
                'wagon':{'color':85, 'speed':np.linspace(1.,2.7,10),   'xsize':max(mx-1,3), 'angles':[], 'lanes':[]},
                'tesla':{'color':100, 'speed':np.linspace(1,5.7,10),   'xsize':max(mx-2,3), 'angles':[], 'lanes':[]},
                'sport':{'color':130, 'speed':np.linspace(3.2,4.7,10), 'xsize':max(mx-3,3), 'angles':[], 'lanes':[]},
                'sedan':{'color':155, 'speed':np.linspace(1.2,3.3,10), 'xsize':max(mx-4,3), 'angles':[], 'lanes':[]},
                }


        self.rdn.shuffle(lanes)
        lanes = list(lanes)
        assign = True
        while assign:
            for name, car in cars.iteritems():
                if not len(lanes):
                    assign=False
                    break
                if not self.rdn.randint(1000)%2:
                    angle = 0.0
                else:
                    angle = 180
                cars[name]['lanes'].append(lanes.pop())
                cars[name]['angles'].append(angle)

        self.cars = cars


    def reset(self):
        max_xcarsize = int(self.xsize*.15)
        self.plotted = False
        plt.close()
       
        # robot shape
        yrsize,xrsize=2,2
        self.safezone = yrsize*3
        init_ys = [self.rdn.randint(yrsize,self.safezone-yrsize), 
                   self.rdn.randint(self.ysize-self.safezone+yrsize, self.ysize-yrsize),
                   self.rdn.randint((self.ysize/2)-self.safezone+yrsize, 
                                    (self.ysize/2)+self.safezone-yrsize)]
        init_y = float(self.rdn.choice(init_ys))
        init_x = float(self.rdn.randint(max_xcarsize,self.xsize-max_xcarsize))
        
        self.robot_map = np.zeros((self.ysize, self.xsize), np.uint8)

        self.robot = Particle(world=self,  name='robot',  
                              local_map=self.robot_map,
                              init_y=init_y, init_x=init_x, 
                              angle=0, speed=0.0, bounce=False, 
                              xmarkersize=xrsize, ymarkersize=yrsize,
                              color=200)

        goal_y = float(self.rdn.randint(2,self.ysize-2))
        goal_x = float(self.rdn.randint(2,self.xsize-2))
        self.goal_map = np.zeros((self.ysize, self.xsize), np.uint8)
        self.goal = Particle(world=self, name='goal', 
                              local_map=self.goal_map,
                              init_y=goal_y, init_x=goal_x, 
                              angle=0, speed=0.0, bounce=True, 
                              ymarkersize=1,xmarkersize=1,
                              color=254)


 
        self.configure_cars(max_xcarsize)
        self.obstacles = {}
        self.cnt = 0
        # initialize map
        [self.step_obstacles() for i  in range(self.max_steps)]
        self.room_maps = np.zeros((self.max_steps, self.ysize, self.xsize), np.uint8)
        # make all of the cars
        for t in range(self.max_steps):
            self.step_obstacles()
            self.room_maps[t,:,:] = self.room_map

        # set steps to zero for the agent
        self.steps = 0
        state = self.goal_map+self.robot_map+self.room_maps[self.steps]
        if not self.robot.alive:
            embed()
        return state

    def step_obstacles(self):
        self.room_map *=0
        dead_obstacles = []
        for n,o in self.obstacles.iteritems():
            alive = o.step(self.timestep) 
            if not alive:
                dead_obstacles.append(n)

        for n in dead_obstacles:
            del self.obstacles[n]
        self.add_frogger_obstacles()
      
    def add_frogger_obstacles(self, level=3):
        # 1.4 is pretty easy
        # 1.2 is much harder
        for name, car in self.cars.iteritems():
            pp = self.rdn.poisson(car['xsize']*2.5, len(car['lanes']))
            for p, angle, lane in zip(pp, car['angles'], car['lanes']):
                if p<car['xsize']+level:
                    if int(angle) == 0:
                        init_x = 0
                    else:
                        init_x = self.xsize-1-car['xsize']

                    self.obstacles[self.cnt] = Particle(world=self, name=self.cnt, 
                                                 local_map=self.room_map,
                                                 init_y=lane, 
                                                 init_x=init_x, 
                                                 angle=angle, 
                                                 speed=self.rdn.choice(car['speed']),
                                                 bounce=False,
                                                 color=car['color'],
                                                 ymarkersize=self.car_ysize,
                                                 xmarkersize=car['xsize']) 
                    self.cnt +=1



    def check_goal_progress(self):
        goal_dis = np.sqrt((self.goal.y-self.robot.y)**2 + (self.goal.x-self.robot.x)**2)
        if goal_dis < self.collision_distance:
            print("reached goal at distance of {} m".format(goal_dis))
            self.robot.alive = False
 

    def step(self, action):
        ''' step agent '''
        if self.robot.alive:
            assert(len(action)==2)
            # room_maps is max_steps long
            room_map = self.room_maps[self.steps]
            self.robot_map *= 0
            self.robot.angle = action[0]
            self.robot.speed = action[1]
            alive = self.robot.step(self.timestep)
            ry, rx = np.where(self.robot_map>0)
            state = room_map+self.robot_map+self.goal_map
            game_over,reward=check_state(ry,rx,room_map,self.goal_map)
            if game_over:
                self.robot.alive = False

            self.steps +=1
            if self.steps > self.max_steps-1:
                # might be off by one
                print("ran out of steps")
                self.robot.alive = False
            if not self.robot.alive:
                print('reward: {}'.format(reward))
            return state, reward, not self.robot.alive, ''
        else:
            print('robot has died')
            raise

    def render(self):
        if not self.goal_map.sum() > 0:
            print("no goal")
            embed()
        if not self.plotted:
            # reset environment
            plt.ion()
            self.fig, self.ax = plt.subplots(1,1)
            self.shown = self.ax.matshow(self.room_map, vmin=0, vmax=255)
            self.ax.set_aspect('equal')
            self.ax.set_ylim(0,self.ysize)
            self.ax.set_xlim(0,self.xsize)
 
        self.shown.set_data(self.room_maps[self.steps]+self.robot_map+self.goal_map)
        plt.show()
        plt.pause(.0001)
      
class BaseAgent():
    def __init__(self, do_plot=False, do_save_figs=False, save_fig_every=3,
                       do_make_gif=False, save_path='saved', n_episodes=10, 
                       seed=133, train=False):

        self.do_plot = do_plot
        self.do_save_figs = do_save_figs
        self.save_fig_every=save_fig_every
        self.save_path = save_path
        self.do_make_gif = do_make_gif

        if self.do_save_figs:
            if train:
                self.img_path = os.path.join(self.save_path, 'imgs_train') 
            else:
                self.img_path = os.path.join(self.save_path, 'imgs_test') 

            if not os.path.exists(self.img_path):
                os.makedirs(self.img_path)
            self.plot_path = os.path.join(self.img_path, 'episode_%06d_frame_%05d_reward_%d.png')
            if self.do_make_gif:
                self.gif_path = os.path.join(self.img_path, 'episode_%06d_frames_%05d.gif')
        # get last img epsidoe - in future should look for last in model or something
        try:
            past = glob(os.path.join(self.img_path, 'episode_*.png'))
            # get name of episode
            eps = [int(os.path.split(p)[1].split('_')[1]) for p in past]
            last_episode = max(eps)
        except Exception, e:
            last_episode = 0
        self.episodes = last_episode
        # note - need new seed for randomness
        ep_seed = seed+last_episode
        print('-------------------------------------------------------------------------')
        print('starting from episode: {} with seed {}'.format(last_episode, ep_seed))
        raw_input('press any key to continue or ctrl+c to exit')
        self.rdn = np.random.RandomState(ep_seed)
        self.env = RoomEnv(obstacle_type='frogger',seed=ep_seed+1)
        self.n_episodes = self.episodes+n_episodes
 
    def run(self):
        while self.episodes < self.n_episodes:
            if self.do_make_gif and (self.episodes==self.n_episodes-1):
                self.do_plot=True
            self.run_episode()

        if self.do_make_gif:
            ne = int(self.episodes-1)
            this_gif_path = self.gif_path %(ne, self.steps)
            logging.info("starting gif creation for episode:{} file:{}".format(ne, this_gif_path))
            search = os.path.join(self.img_path, 'episode_%04d_frame_*.png' %ne)
            cmd = 'convert %s %s'%(search, this_gif_path)
            # make gif
            Popen(cmd.split(' '))

    def get_goal_action(self, state):
        goal_y, goal_x = self.env.goal.y, self.env.goal.x
        dy = goal_y-self.env.robot.y
        dx = goal_x-self.env.robot.x
        goal_angle = np.rad2deg(math.atan2(dy,dx))
        return [goal_angle, 1.0]

    def get_random_action(self, state):
        random_angle = self.rdn.choice(self.env.action_space[0] )
        random_speed = self.rdn.choice(self.env.action_space[1] )
        return [random_angle, random_speed]

#    def get_mcts_action(self, state, ucb_weight=0.5):
#        init_y, init_x = self.env.robot.y, self.env.robot.x
#
#        actions = [(action_angle,action_speed) for action_angle in self.env.action_space[0] for action_speed in self.env.action_space[1]]
#        move_states = []
#
#        for action in actions:
#            # reset robot back to starting position
#            # first step
#            self.env.robot.y = init_y
#            self.env.robot.x = init_x
#            self.env.steps = 0
#            action = [action_angle, action_speed]
#            next_state, reward, finished, _ = self.env.step(action)
#            move_states.append(next_state)

    def run_episode(self):
        print("starting episode {}".format(self.episodes))
        self.steps = 0
        state = self.env.reset()
        finished = False
        while not finished:
            action = self.get_goal_action(state)
            #action = self.get_random_action(state)
            next_state, reward, finished, _ = self.env.step(action)
            self.steps+=1
            state = next_state
            if self.do_plot:
                self.env.render()
            if self.do_save_figs:
                if not self.steps%self.save_fig_every:
                    signal = 1 # one step
                    if finished: # episode over
                        if reward > 0:
                            signal = 2 # won
                        else:
                            signal = 0 # died

                    this_plot_path = self.plot_path %(self.episodes, self.steps, signal)
                    imwrite(this_plot_path, state)
 
        self.episodes+=1

if __name__ == '__main__':
    #env = RoomEnv(obstacle_type='walkers')
    ba = BaseAgent(do_plot=False, do_save_figs=True, train=True, n_episodes=20000, seed=415)
    ba.run()

