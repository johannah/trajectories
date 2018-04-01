import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import os
from subprocess import Popen
import gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import logging

class Particle():
    def __init__(self, world, name, init_y, init_x,
                 angle, speed, collide=False, bounce=True, 
                 color='r', marker='o', markersize=7):

        self.world = world
        self.y = init_y 
        self.x = init_x
        self.name = name
        self.angle = angle
        self.speed = speed
        self.color = color
        self.alive = True
        self.collide = collide
        self.bounce = bounce
        self.steps = 0

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
        self.points.set_data([self.x], [self.y])



class RoomEnv():
    def __init__(self, ysize=120, xsize=100, n_obstacles=20, seed=10, 
                 timestep=1,collision_distance=3):
        # TODO - what if episode already exists in savedir
        self.ysize = ysize
        self.xsize = xsize
        self.timestep = timestep
        self.max_speed = 2.0
        # average speed
        self.average_speed = 1.0
        # make max steps twice the steps required to cross diagonally across the room
        self.max_steps = 2*(np.sqrt(self.ysize**2 + self.xsize**2)/float(self.average_speed))/float(self.timestep)
        self.n_obstacles = n_obstacles
        self.action_space = [np.linspace(-1, 1, 5), np.linspace(0, 1, 3)]
        self.collision_distance=collision_distance
        self.rdn = np.random.RandomState(seed)

    def get_data_from_fig(self):
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def reset(self):
        # reset environment
        plt.ion()
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_aspect('equal')
        self.ax.set_ylim(0,self.ysize)
        self.ax.set_xlim(0,self.xsize)
        plt.draw()

        init_y = float(self.rdn.randint(0,self.ysize))
        init_x = float(self.rdn.randint(0,self.xsize))
        goal_y = float(self.rdn.randint(0,self.ysize))
        goal_x = float(self.rdn.randint(0,self.xsize))
        self.reward = 0
        
        self.goal = Particle(world=self, name='goal', init_y=goal_y, init_x=goal_x, 
                              angle=0, speed=0.0, collide=True, bounce=False, 
                              color='r', marker='x')

        self.robot = Particle(world=self, name='robot', 
                              init_y=init_y, init_x=init_x, 
                              angle=0, speed=0.1, collide=True, bounce=False, 
                              color='g', marker='s')

        self.obstacles = {}
        for n in range(self.n_obstacles):
            self.obstacles[n] = Particle(world=self, name=n, 
                                         init_y=self.rdn.randint(0,self.ysize), 
                                         init_x=self.rdn.randint(0,self.xsize), 
                                         angle=self.rdn.randint(0,360), speed=1.5,
                                         collide=False,bounce=True,
                                         color='b', marker='o', markersize=10) 
        self.steps = 0
        state = self.get_data_from_fig()
        return state

    def check_for_collisions(self):
        # if particle is able to collide with other agents
        ds = []
        for n,o in self.obstacles.iteritems():
            dis = np.sqrt((o.y-self.robot.y)**2 + (o.x-self.robot.x)**2)
            ds.append(dis)
            if dis < self.collision_distance:
                print("robot collided with obstacle {} at distance of {} m".format(o.name,dis))
                self.robot.alive = False
        print(min(ds))

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
        self.steps +=1
        if self.steps < self.max_steps:
            for n,o in self.obstacles.iteritems():
                o.step(self.timestep) 
        else:
            self.robot.alive = False

        next_state = self.get_data_from_fig()
        self.check_for_collisions()
        self.check_goal_progress()
        return next_state, self.reward, not self.robot.alive, ''

    def render(self):
        plt.show()
        plt.pause(.0001)
      
class Agent():
    def __init__(self):
        pass
if __name__ == '__main__':
    env = RoomEnv()
    s = env.reset()
    finished = False
    while not finished:
        next_state, reward, finished, _ = env.step(0.5)
        env.render()

#     #do_plot=True,do_save_figs=True,save_fig_every=10,do_make_gif=False,
#                       save_path='saved'):
#
#        else:
#            if self.do_make_gif:
#                this_gif_path = self.gif_path %(self.episode, self.steps)
#                logging.info("starting gif creation for episode:{} file:{}".format(self.episode, this_gif_path))
#                search = os.path.join(self.img_path, 'episode_%04d_*.png' %self.episode) 
#                cmd = 'convert %s %s'%(search, this_gif_path)
#                # make gif
#                Popen(cmd.split(' '))
#            if self.do_save_figs:
#                if not self.steps%self.save_fig_every:
#                    this_plot_path = self.plot_path %(self.episode, self.steps)
#                    plt.savefig(this_plot_path)
#
#
#        if self.do_save_figs:
#             this_plot_path = self.plot_path %(self.episode, self.steps)
#             plt.savefig(this_plot_path)
#
#        self.episode = 0
#        self.do_plot = do_plot
#        self.do_save_figs = do_save_figs
#        self.save_fig_every=save_fig_every
#        self.save_path = save_path
#        self.do_make_gif = do_make_gif
#        if self.do_save_figs:
#            self.img_path = os.path.join(self.save_path, 'imgs') 
#            if not os.path.exists(self.img_path):
#                os.makedirs(self.img_path)
#            self.plot_path = os.path.join(self.img_path, 'episode_%04d_frame_%05d.png')
#            if self.do_make_gif:
#                self.gif_path = os.path.join(self.img_path, 'episode_%04d_frames_%05d.gif')
# 
#        if self.do_plot:
#            plt.show()
#            plt.pause(.0001)
#
#
