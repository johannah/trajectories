import matplotlib
matplotlib.use('TkAgg')
import os
from subprocess import Popen
import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import logging
import math

class Particle():
    def __init__(self, world, name, local_map, init_y, init_x,
                 angle, speed, bounce=True, 
                 color=12, ymarkersize=3, xmarkersize=3):

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
                self.angle+=np.sign(self.angle)*40
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
            y = range(int(self.y), newyplus)
            x = range(int(self.x), newxplus)
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

        if (newy <= 0):
            newy = 0.0
            hit_wall = True
        if (newx <= 0):
            newx = 0.0
            hit_wall = True
        self.wall_bounce(hit_wall)
        return newy, newx, newyplus, newxplus

 

class RoomEnv():
    def __init__(self, obstacle_type='walkers', ysize=128, xsize=100, seed=10, 
                 timestep=1,collision_distance=2):
        # TODO - what if episode already exists in savedir
        self.car_every = 10
        self.safezone = collision_distance*3
        self.steps = 0
        self.obstacle_type = obstacle_type
        self.ysize = ysize
        self.xsize = xsize
        self.timestep = timestep
        self.max_speed = 2.0
        # average speed
        self.average_speed = 1.0
        # make max steps twice the steps required to cross diagonally across the room
        self.max_steps = 2*(np.sqrt(self.ysize**2 + self.xsize**2)/float(self.average_speed))/float(self.timestep)
        self.action_space = [np.linspace(-1, 1, 5), np.linspace(0, self.max_speed, 4)]
        self.collision_distance=collision_distance
        self.rdn = np.random.RandomState(seed)
        self.room_map = np.zeros((self.ysize, self.xsize), np.uint8)

    def get_data_from_fig(self):
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def configure_cars(self):
        self.car_ysize=2
        median1 = (self.ysize/2)-(self.safezone/2)
        median2 = (self.ysize/2)+(self.safezone/2)

        self.human_markers = np.zeros_like(self.room_map)
        self.human_markers[median1,:] = 255
        self.human_markers[median2,:] = 255

        self.human_markers[self.safezone,:] = 255
        self.human_markers[self.ysize-self.safezone,:] = 255

        lanes = list(np.arange(self.safezone+1, median1-self.car_ysize, self.car_ysize))
        lanes.extend(list(np.arange(median2+1, self.ysize-self.safezone-self.car_ysize, self.car_ysize)))
        
        cars = {
                'truck':{'color':45, 'speed':np.linspace(.2,.7,10),    'xsize':9, 'angles':[], 'lanes':[]},
                'wagon':{'color':85, 'speed':np.linspace(1.,2.7,10),  'xsize':7, 'angles':[], 'lanes':[]},
                'tesla':{'color':100, 'speed':np.linspace(1,5.7,10),    'xsize':4, 'angles':[], 'lanes':[]},
                'sport':{'color':130, 'speed':np.linspace(3.2,4.7,10),  'xsize':5, 'angles':[], 'lanes':[]},
                'sedan':{'color':155, 'speed':np.linspace(1.2,3.3,10), 'xsize':6, 'angles':[], 'lanes':[]},
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
        plt.close()
        # reset environment
        plt.ion()
        self.fig, self.ax = plt.subplots(1,1)
        self.shown = self.ax.matshow(self.room_map, vmin=0, vmax=255)
        self.ax.set_aspect('equal')
        self.ax.set_ylim(0,self.ysize)
        self.ax.set_xlim(0,self.xsize)
        
        init_ys = [self.rdn.randint(0,self.safezone), 
                   self.rdn.randint(self.ysize-self.safezone, self.ysize-1),
                   self.rdn.randint((self.ysize/2)-self.safezone, 
                                    (self.ysize/2)+self.safezone)]
        init_y = float(self.rdn.choice(init_ys, 1))
        init_x = float(self.rdn.randint(0,self.xsize))
        goal_y = float(self.rdn.randint(0,self.ysize))
        goal_x = float(self.rdn.randint(0,self.xsize))
        self.reward = 0
        self.goal_map = np.zeros((self.ysize, self.xsize), np.uint8)
        self.robot_map = np.zeros((self.ysize, self.xsize), np.uint8)
        self.goal = Particle(world=self, name='goal', 
                              local_map=self.goal_map,
                              init_y=goal_y, init_x=goal_x, 
                              angle=0, speed=0.0, bounce=False, 
                              ymarkersize=1,xmarkersize=1,
                              color=254)

        self.robot = Particle(world=self,  name='robot',  
                              local_map=self.robot_map,
                              init_y=init_y, init_x=init_x, 
                              angle=0, speed=0.1, bounce=False, 
                              xmarkersize=2, ymarkersize=2,
                              color=200)

        self.configure_cars()
        self.human_markers+=self.goal_map
        self.obstacles = {}
        self.cnt = 0
        # start a few cars
        if self.obstacle_type == 'frogger':
            [self.step([0,0]) for i in range(self.xsize)]
        self.steps = 0
        #state = self.get_data_from_fig()
        state = self.goal_map+self.robot_map+self.room_map
        return state

    def add_frogger_obstacles(self, level=1):
        # 1.4 is pretty easy
        # 1.2 is much harder
        for name, car in self.cars.iteritems():
            pp = self.rdn.poisson(car['xsize']*1.7, len(car['lanes']))
            print(pp)
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



    def add_walker_obstacles(self):
        for n in self.rdn.randint(0,self.ysize/2,self.ysize/4):
            self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                         init_y=n, 
                                         init_x=0, 
                                         angle=0, speed=1.5,
                                         bounce=False,
                                         color=12)
            self.cnt +=1
        for n in self.rdn.randint(self.ysize/2,self.ysize,self.ysize/4):
            self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                         init_y=n, 
                                         init_x=self.xsize-1, 
                                         angle=180, speed=1.5,
                                         bounce=False,
                                         color=34) 
            self.cnt +=1


    def check_for_collisions(self):
        # if particle is able to collide with other agents
        ds = []
        for n,o in self.obstacles.iteritems():
            dis = np.sqrt((o.y-self.robot.y)**2 + (o.x-self.robot.x)**2)
            ds.append(dis)
            if dis < self.collision_distance:
                print("robot collided with obstacle {} at distance of {} m".format(o.name,dis))
                #self.robot.alive = False

    def check_goal_progress(self):
        goal_dis = np.sqrt((self.goal.y-self.robot.y)**2 + (self.goal.x-self.robot.x)**2)
        if goal_dis < self.collision_distance:
            print("reached goal at distance of {} m".format(goal_dis))
            self.robot.alive = False
 

    def step(self, action):
        ''' step agent '''
        self.room_map *= 0
        self.robot_map *= 0
        assert(len(action)==2)
        self.steps +=1
        dead_obstacles = []
        if self.steps < self.max_steps:
            for n,o in self.obstacles.iteritems():
                alive = o.step(self.timestep) 
                if not alive:
                    dead_obstacles.append(n)
        else:
            self.robot.alive = False

        for n in dead_obstacles:
            del self.obstacles[n]
        self.robot.angle = action[0]
        self.robot.speed = action[1]
        self.robot.step(self.timestep)
        #next_state = self.get_data_from_fig()
        print('goal',np.argmax(self.goal_map))
        state = self.room_map+self.robot_map+self.goal_map
        self.check_for_collisions()
        self.check_goal_progress()
        if not self.steps%self.car_every:
            self.add_frogger_obstacles()
        return state, self.reward, not self.robot.alive, ''

    def render(self):
        self.shown.set_data(self.room_map+self.human_markers+self.robot_map)
        plt.show()
        plt.pause(.0001)
      
class BaseAgent():
    def __init__(self, env, do_plot=False, do_save_figs=False, save_fig_every=10,
                       do_make_gif=False, save_path='saved', n_episodes=10):

        self.env = env
        self.episodes = 0
        self.n_episodes = n_episodes
        self.do_plot = do_plot
        self.do_save_figs = do_save_figs
        self.save_fig_every=save_fig_every
        self.save_path = save_path
        self.do_make_gif = do_make_gif

        if self.do_save_figs:
            self.img_path = os.path.join(self.save_path, 'imgs') 
            if not os.path.exists(self.img_path):
                os.makedirs(self.img_path)
            self.plot_path = os.path.join(self.img_path, 'episode_%04d_frame_%05d.png')
            if self.do_make_gif:
                self.gif_path = os.path.join(self.img_path, 'episode_%04d_frames_%05d.gif')


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

    def get_action(self, state):
        goal_y, goal_x = self.env.goal.y, self.env.goal.x
        dy = goal_y-self.env.robot.y
        dx = goal_x-self.env.robot.x
        goal_angle = np.rad2deg(math.atan2(dy,dx))
        return [goal_angle, 1.0]

    def run_episode(self):
        print("starting episode {}".format(self.episodes))
        self.steps = 0
        state = self.env.reset()
        finished = False
        while not finished:
            action = self.get_action(state)
            next_state, reward, finished, _ = self.env.step(action)
            if self.do_plot:
                self.env.render()
                if self.do_save_figs:
                    if not self.steps%self.save_fig_every:
                        this_plot_path = self.plot_path %(self.episodes, self.steps)
                        plt.savefig(this_plot_path)
            self.steps+=1
            state = next_state
        self.episodes+=1


if __name__ == '__main__':
    #env = RoomEnv(obstacle_type='walkers')
    env = RoomEnv(obstacle_type='frogger')
    ba = BaseAgent(env, do_plot=True, n_episodes=2)
    ba.run()

