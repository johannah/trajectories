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
    def __init__(self, world, name, init_y, init_x,
                 angle, speed, bounce=True, 
                 color='r', marker='o', markersize=7):

        self.steps = 0
        self.world = world
        self.y = init_y 
        self.x = init_x
        self.name = name
        self.angle = angle
        self.speed = speed
        self.color = color
        self.alive = True
        self.bounce = bounce

        self.points = self.world.ax.plot([self.x], [self.y],
                                    linestyle='None', 
                                    marker=marker, c=self.color, 
                                    markersize=markersize)[0]

    def wall_bounce(self):
        self.angle+=np.sign(self.angle)*40

    def step(self, timestep):
        # (meters/second) * second
        # TODO angle
        rads = np.deg2rad(self.angle) 
        newy = self.y + self.speed*np.sin(rads)*timestep
        newx = self.x + self.speed*np.cos(rads)*timestep

        self.y,self.x,bounce = self.world.collide_with_walls(newy,newx)
        if bounce:
            if self.bounce:
                self.wall_bounce()
            else:
                self.alive = False

        self.steps +=1
        if self.alive: 
            self.points.set_data([self.x], [self.y])
        return self.alive


class RoomEnv():
    def __init__(self, obstacle_type='walkers', ysize=128, xsize=100, seed=10, 
                 timestep=1,collision_distance=2):
        # TODO - what if episode already exists in savedir
        self.car_every = 10
        self.safezone = collision_distance*2
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

    def get_data_from_fig(self):
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def reset(self):
        plt.close()
        # reset environment
        plt.ion()
        self.fig, self.ax = plt.subplots(1,1)
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
        
        self.goal = Particle(world=self, name='goal', init_y=goal_y, init_x=goal_x, 
                              angle=0, speed=0.0, bounce=False, 
                              color='r', marker='x')

        self.robot = Particle(world=self, name='robot', 
                              init_y=init_y, init_x=init_x, 
                              angle=0, speed=0.1, bounce=False, 
                              color='g', marker='o')

        self.obstacles = {}
        self.cnt = 0
        if self.obstacle_type == 'frogger':
            [self.step([0,0]) for i in range(self.car_every*5)]
        if self.obstacle_type == 'walker':
            self.add_walker_obstacles()
        self.steps = 0

        state = self.get_data_from_fig()
        return state

    def add_frogger_obstacles(self):
        cars ={'blue':1.5, 'cornflowerblue':.5, 'teal':2}
        for color, speed in cars.iteritems():
            num_cars = self.rdn.randint(1,1+self.ysize*.02)
            median1 = (self.ysize/2)-(self.safezone/2)
            median2 = (self.ysize/2)+(self.safezone/2)

            one_layer = self.rdn.randint(self.safezone, median1, num_cars)
            two_layer = self.rdn.randint(median2,self.ysize-self.safezone,num_cars)

            for n in one_layer:
                self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                             init_y=n, 
                                             init_x=0, 
                                             angle=0.0, speed=speed,
                                             bounce=False,
                                             color=color, marker='s', markersize=5) 
                self.cnt +=1
            for n in two_layer:
                self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                             init_y=n, 
                                             init_x=self.xsize-1, 
                                             angle=180, speed=speed,
                                             bounce=False,
                                             color=color, marker='s', markersize=5) 
                self.cnt +=1



    def add_walker_obstacles(self):
        for n in self.rdn.randint(0,self.ysize/2,self.ysize/4):
            self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                         init_y=n, 
                                         init_x=0, 
                                         angle=0, speed=1.5,
                                         bounce=False,
                                         color='b', marker='s', markersize=10) 
            self.cnt +=1
        for n in self.rdn.randint(self.ysize/2,self.ysize,self.ysize/4):
            self.obstacles[self.cnt] = Particle(world=self, name=n, 
                                         init_y=n, 
                                         init_x=self.xsize-1, 
                                         angle=180, speed=1.5,
                                         bounce=False,
                                         color='b', marker='s', markersize=10) 
            self.cnt +=1


    def check_for_collisions(self):
        # if particle is able to collide with other agents
        ds = []
        for n,o in self.obstacles.iteritems():
            dis = np.sqrt((o.y-self.robot.y)**2 + (o.x-self.robot.x)**2)
            ds.append(dis)
            if dis < self.collision_distance:
                print("robot collided with obstacle {} at distance of {} m".format(o.name,dis))
                self.robot.alive = False

    def check_goal_progress(self):
        goal_dis = np.sqrt((self.goal.y-self.robot.y)**2 + (self.goal.x-self.robot.x)**2)
        if goal_dis < self.collision_distance:
            print("reached goal at distance of {} m".format(goal_dis))
            self.robot.alive = False
 
    def collide_with_walls(self,newy,newx):
        bounce = False
        if (newy >= self.ysize-1):
            # TODO bounce smarter in the correct direction
            newy = float(self.ysize-1)
            bounce = True
        if (newx >= self.xsize-1):
            newx = float(self.xsize-1)
            bounce = True
        if (newy <= 0):
            newy = 0.0
            bounce = True
        if (newx <= 0):
            newx = 0.0
            bounce = True
        return newy, newx, bounce
 
    def step(self, action):
        ''' step agent '''
        print('step', self.steps)
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

        for d in dead_obstacles: 
            self.obstacles[d].points.remove()
            del self.obstacles[d].points
            del self.obstacles[d]

        self.robot.angle = action[0]
        self.robot.speed = action[1]
        self.robot.step(self.timestep)
        next_state = self.get_data_from_fig()
        self.check_for_collisions()
        self.check_goal_progress()
        if not self.steps%self.car_every:
            self.add_frogger_obstacles()
        return next_state, self.reward, not self.robot.alive, ''

    def render(self):
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

    def get_action(self,state):
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
    ba = BaseAgent(env, do_plot=False, n_episodes=6)
    ba.run()

