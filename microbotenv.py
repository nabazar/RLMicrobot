# -*- coding: utf-8 -*-
"""MicrobotEnv - Corrected Version"""

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.matlib

from magneticfieldsim import MagneticFieldSim
from microrobotmodel import MicroRobotModel

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def rotationmatrix(thx, thy, thz):
    Rotx = np.array([[1, 0, 0, 0],
                     [0, np.cos(thx), -np.sin(thx), 0],
                     [0, np.sin(thx), np.cos(thx), 0],
                     [0, 0, 0, 1]])
    Roty = np.array([[np.cos(thy), 0, np.sin(thy), 0],
                     [0, 1, 0, 0],
                     [-np.sin(thy), 0, np.cos(thy), 0],
                     [0, 0, 0, 1]])
    Rotz = np.array([[np.cos(thz), -np.sin(thz), 0, 0],
                     [np.sin(thz), np.cos(thz), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    Rot = Rotz @ Roty @ Rotx
    return Rot

class Microrobot_Env(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.done = False
        self.reward = 0
        self.dt = 0.01
        
        # Initialize microbot model
        self.microbot = MicroRobotModel([0, 0, 0], np.zeros((3, 3)), 0)
        self.rm = self.microbot.rm
        self.r1 = self.microbot.r1
        self.r2 = self.microbot.r2
        self.Rh = self.microbot.Rh

        # Full state: [theta, theta_dot, x, y, vx, vy]
        self.state = np.zeros(6, dtype=np.float32)
        self.next_state = np.zeros(6, dtype=np.float32)
        self.start = 0
        self.theta = 0
        self.theta_dot = 0
        self.goal_distance = np.deg2rad(3)
        self.goal = self.start + self.goal_distance
        self.goal = self.correct_for_wrap_rad(self.goal)
        
        self.x = self.rm * np.cos(self.theta)
        self.y = self.rm * np.sin(self.theta)
        self.vx = 0
        self.vy = 0
        self.P = np.array([self.x, self.y, 0, 1])
        
        xt, yt = pol2cart(self.rm, self.goal)
        self.target_cart = [xt, yt, 0]
        self.start_cart = [self.x, self.y, 0]
        
        self.path = [[self.x, self.y, 0]]
        self.step_rate = 1
        num_actions = 3
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
        
        # Observation space matches state (6 values)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10, -0.1, -0.1, -1, -1], dtype=np.float32),
            high=np.array([np.pi, 10, 0.1, 0.1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.axs = 0
        self.testmode = 0
        self.time = 0

    def step(self, action):
        self.goal = self.start + self.goal_distance
        self.goal = self.correct_for_wrap_rad(self.goal)
        
        theta = self.state[0]
        x = self.rm * np.cos(theta)
        y = self.rm * np.sin(theta)
        
        # Apply actions to generate currents
        mu1 = 30
        I = [
            mu1 * action[0],
            mu1 * action[1],
            mu1 * action[2],
            -mu1 * action[0],
            -mu1 * action[1],
            -mu1 * action[2]
        ]
        
        self.info = I
        P = np.array([x, y, 0, 1])
        self.P = P
        
        # Calculate magnetic field
        mag = MagneticFieldSim(P, I)
        self.mag = mag
        B, J = mag.IntegrationOfBiotSavar()
        
        # Update microbot model
        self.microbot.B = B
        self.microbot.J = J
        self.microbot.P = P
        self.microbot.th = theta
        
        # Get dynamics
        v, w, M = self.microbot.MicroRobotDyn()
        
        # Update position
        dx = v[0]
        dy = v[1]
        self.vx = dx
        self.vy = dy
        x = x + self.dt * dx
        y = y + self.dt * dy
        
        # Convert to polar coordinates
        new_r, new_theta = cart2pol(x, y)
        new_theta = self.correct_for_wrap_rad(new_theta)
        
        # Update theta_dot
        self.theta_dot = (new_theta - theta) / self.dt
        
        # Calculate reward
        xt, yt, _ = self.target_cart
        d_target = np.sqrt((xt - x)**2 + (yt - y)**2)
        dltg = np.rad2deg(self.correct_for_wrap_rad(self.goal - new_theta))
        
        loss = -0.5 * abs(1e2 * d_target)**2 - 0.5 * abs(dltg)**2
        self.reward = 0.001 * loss
        
        # Check if goal is reached
        if abs(d_target) < 5e-4 and abs(dltg) < 1:
            self.reward += 100
            self.done = True
        else:
            self.reward -= 1
            self.done = False
        
        # Update full state
        self.theta = new_theta
        self.next_state = np.array([new_theta, self.theta_dot, x, y, self.vx, self.vy], dtype=np.float32)
        self.state = self.next_state
        
        self.path.append([x, y, 0])
        self.time += self.dt
        
        return self.next_state, self.reward, self.done, {}

    def reset(self):
        self.done = False
        self.reward = 0
        self.start = np.random.uniform(0, 2 * np.pi)
        self.theta = self.start
        self.theta_dot = 0
        self.goal = self.start + self.goal_distance
        self.goal = self.correct_for_wrap_rad(self.goal)
        
        xt, yt = pol2cart(self.rm, self.goal)
        self.target_cart = [xt, yt, 0]
        
        x, y = pol2cart(self.rm, self.start)
        self.start_cart = [x, y, 0]
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        
        self.state = np.array([self.theta, self.theta_dot, x, y, self.vx, self.vy], dtype=np.float32)
        self.path = [[x, y, 0]]
        self.time = 0
        
        return self.state

    def resetForTest(self):
        self.done = False
        self.reward = 0
        self.start = 0
        self.theta_dot = 0
        self.goal = self.start + self.goal_distance
        self.goal = self.correct_for_wrap_rad(self.goal)
        self.theta = self.start
        
        xt, yt = pol2cart(self.rm, self.goal)
        self.target_cart = [xt, yt, 0]
        
        x, y = pol2cart(self.rm, self.start)
        self.start_cart = [x, y, 0]
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        
        self.state = np.array([self.theta, self.theta_dot, x, y, self.vx, self.vy], dtype=np.float32)
        self.path = [[x, y, 0]]
        self.time = 0
        
        return self.state

    def EnvPlot(self):
        fig = plt.figure(figsize=(10, 10))
        axs = plt.axes(projection='3d')
        
        # Plot workspace boundaries
        angles = np.linspace(0, 2 * np.pi, 100)
        axs.plot(self.r1 * np.cos(angles), self.r1 * np.sin(angles), np.zeros_like(angles), 'b')
        axs.plot(self.r2 * np.cos(angles), self.r2 * np.sin(angles), np.zeros_like(angles), 'b')
        axs.plot(self.rm * np.cos(angles), self.rm * np.sin(angles), np.zeros_like(angles), 'c', linestyle='dashed')
        
        # Plot coils if available
        if hasattr(self, 'mag'):
            xx, yy, zz = self.mag.xx, self.mag.yy, self.mag.zz
            colors = ['b', 'g', 'r', 'b', 'g', 'r']
            for j in range(6):
                axs.plot3D(xx[j, :], yy[j, :], zz[j, :], colors[j])
        
        # Plot current position
        P = self.P
        axs.plot3D([P[0]], [P[1]], [0], 'go', markersize=10)
        
        # Plot target
        xt, yt, _ = self.target_cart
        axs.plot3D([xt], [yt], [0], 'ro', markersize=10)
        
        # Plot path
        path_array = np.array(self.path)
        if len(path_array) > 1:
            axs.plot3D(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'g-', alpha=0.5)
        
        axs.set_xlabel('X')
        axs.set_ylabel('Y')
        axs.set_zlabel('Z')
        axs.set_title('Microbot Environment')
        
        self.axs = axs
        return fig, axs

    def render(self, mode='human'):
        fig, axs = self.EnvPlot()
        plt.show()
        return fig, axs

    def correct_for_wrap_rad(self, angle):
        angle = angle % (2 * np.pi)
        return angle
