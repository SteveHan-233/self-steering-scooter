import os

# set rendering engine
os.environ['MUJOCO_GL'] = 'egl'

# mujoco wrapper
from dm_control import mujoco

#RL Libraries
import gym

#PyMJCF
from dm_control import mjcf

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# General
import copy
import os
import itertools
import numpy as np
import math

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import PIL.Image

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
def display_video(frames, render_name="render", framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    fig.show()
    anim.save(render_name + ".gif")

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qw, qx, qy, qz]

scooter_model="""
<mujoco>
    <option gravity="0 0 -9.8">
    </option>
    <visual>
        <global />
    </visual>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
         rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
        <material name="scooterBody" rgba=".3 .3 .3 1"/>
        <material name="scooterWheel" rgba=".3 .3 .3 1"/>
    </asset>
    <worldbody>
        <geom type="plane" material="grid" pos="-80 0 0" size="100 100 .1"/>
        <body name="scooter" pos="0 0 .1" euler="0 0 0">
            <camera name="third_person" mode="track" pos="-1.5 0 1.6" xyaxes="0 -1 0 0.707 0 0.707"/>
            <joint name="free_joint" type="free"/>
            <geom type="box" size=".3 .1 .02" material="scooterBody"/>
            <geom type="box" pos=".37 .08 0" size=".07 .02 .02" material="scooterBody"/>
            <geom type="box" pos=".37 -.08 0" size=".07 .02 .02" material="scooterBody"/>
            <body name="back_wheel" pos=".42 0 0">
                <joint name="axle_back" frictionloss=".001" type="hinge" axis="0 -1 0" />
                <geom type="ellipsoid" euler="90 0 0" friction="2 .005 .0001" size=".1 .1 .05" material="scooterWheel" />
            </body>
            <body pos="-.41 0 0" name="steering_pole" euler="0 10 0">
                <joint limited="true" frictionloss="5.0" damping="0.1" armature="0.002" range="-30.58 30.58" name="steering_axle" type="hinge" axis="0 0 1" />
                <geom pos="0 0 .52" type="cylinder" size=".03 .4"  /> 
                <geom name="handle_bar" pos="0 0 0.88" type="cylinder" size=".02 .2" euler="90 0 0" /> 
                <body name="front_wheel" pos="0 0 0">
                    <joint name="axle_front" type="hinge" axis="0 -1 0" frictionloss=".1" damping=".01" armature=".01" />
                    <geom type="ellipsoid" euler="90 0 0" friction="2 .005 .0001" size=".1 .1 .05" material="scooterWheel" />
                </body>
            </body>
        </body>
        <!-- <body> 
            <geom type="cylinder" pos="-2 0 -.8" size="1 3" euler="90 0 0" />
        </body> -->
    </worldbody>
    <actuator>
        <velocity gear=".34"  name="forwardMotor" joint="axle_front" kv="100"/>
        <position name="steering_pos" kp="50.0" joint="steering_axle" />
    </actuator>
    <sensor>
        <jointpos name="steering_sensor" joint="steering_axle" />
    </sensor>
</mujoco>
"""

duration = 10
framerate = 30

class Sim(gym.Env):
    def __init__(self, hertz):
        root = mjcf.from_xml_string(scooter_model)

        high = np.array([.1, np.finfo(np.float32).max, .5, .5])

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-high, high=high)

        self.hertz = hertz
        self.timestep = 0

        for i in range(5):
            root.worldbody.add('light', diffuse=[.5, .5, .5], pos=[i * -15, 0, 10], dir=[0, 0, -1])

        self.physics = mjcf.Physics.from_mjcf_model(root)

        self.scene_option = mujoco.wrapper.core.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = False

        self.frames = []
    
    def reset(self):
        self.physics.reset()
        self.timestep = 0
        self.frames=[]
        angle = np.random.uniform(-.1, .1)
        self.physics.named.data.qpos['free_joint'][3:] = get_quaternion_from_euler(angle, 0, 0)
        self.physics.named.data.ctrl['forwardMotor'] = 20
        self.physics.named.data.qvel['free_joint'] = [-5.8, 0, 0, 0, 0, 0]
        self.physics.step()
        return self.get_state()
    
    def step(self, action):
        if (abs(self.get_steer_goal()) < .49):
            steer = 0
            if action == 0: 
                steer = -.01
            elif action == 1:
                steer = -.005
            elif action == 3:
                steer = .005
            elif action == 4:
                steer = .01
            self.physics.named.data.ctrl['steering_pos'] += steer
        while True:
            self.physics.step()
            if self.timestep < self.physics.data.time * self.hertz:
                self.timestep += 1
                break
        tilt_angle = self.get_tilt_angle()
        done = False
        reward = 1
        if abs(tilt_angle) > .5: 
            done = True
            reward = 0 
        info = {}
        return self.get_state(), reward, done, info


    def get_tilt_angle(self):
        quat = self.physics.named.data.qpos["free_joint"][3:7]
        sinr_cosp = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        cosr_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2])
        return min(max(math.atan2(sinr_cosp, cosr_cosp), -1), 1)

    def get_tilt_velocity(self):
        return self.physics.named.data.qvel["free_joint"][3]
    
    def render_episode(self, model, render_name, max_time=999999):
        obs = self.reset()
        done = False
        pixels = self.render_pixels()
        self.frames.append(pixels)
        while not done and self.physics.data.time < max_time:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.step(action)
            print(action)
            if len(self.frames) < self.physics.data.time * framerate:
                pixels = self.render_pixels()
                self.frames.append(pixels)
        display_video(self.frames, render_name, framerate)
        self.reset()

    def render_pixels(self):
        return self.physics.render(camera_id=0, scene_option=self.scene_option)

    def display_frame(self):
        img = PIL.Image.fromarray(self.render_pixels())
        img.show()


    def get_steer_angle(self):
        return min(max(self.physics.named.data.sensordata['steering_sensor'][0], -.5), .5)

    def get_steer_goal(self):
        return self.physics.named.data.ctrl['steering_pos']

    def get_state(self):
        return np.array([self.get_tilt_angle(), self.get_tilt_velocity(), self.get_steer_angle(), self.get_steer_goal()], dtype=np.float32)
        
    def simulate(self):
        self.reset()
        while self.physics.data.time < duration:
            self.physics.step()
            if len(self.frames) < self.physics.data.time * framerate:
                pixels = self.physics.render(camera_id=0, scene_option=self.scene_option)
                self.frames.append(pixels)
                #print(self.get_steer_angle(), self.get_tilt_angle())
                #print(self.get_steer_goal(), self.get_steer_angle())
        display_video(self.frames, framerate)
