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
def display_video(frames, render_name="render", framerate=30, lines=[]):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    #fig, ax = plt.subplots(1, 1)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    line, = ax.plot([], [], '-')
    def update(i):
      im.set_data(frames[i])
      line.set_data(lines[i][0], lines[i][1])
      return im, line
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(frames),
                                   interval=interval, blit=True, repeat=False)
    fig.show()
    anim.save(render_name + ".mp4")

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
        <global offwidth="1920" offheight="1080" />
        <scale actuatorlength="2" actuatorwidth=".07" />
        <rgba actuator=".1 .1 .1 .8"/>
        <quality shadowsize="4096"/>
    </visual>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
         rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
        <material name="scooterBody" rgba=".3 .3 .3 1"/>
        <material name="scooterWheel" rgba=".3 .3 .3 1"/>
    </asset>
    <worldbody>
        <body name="camera" mocap="true" pos="-10 0 5" euler="0 -50 -90">
            <camera name="top_view" mode="fixed" />
        </body>
        <!--<body name="camera" pos="-25 0 40" euler="0 0 0">
            <camera name="top_view" mode="fixed" />
        </body>-->
        <geom type="plane" material="grid" pos="-80 0 0" size="100 100 .1"/>
        <body name="scooter" pos="0 0 .1" euler="0 0 0">
            <!--<camera name="third_person" mode="track" pos="2 0 .8" euler="0 80 90"/>-->
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
        <body>
            <geom type="cylinder" pos="-50 0 1" size=".8 1" rgba="1 0.2 0.2 1"/>
        </body>
        <!--
        <body> 
            <geom type="cylinder" pos="-5 0 -.95" size="1 20" euler="90 10 30" />
            <geom type="cylinder" pos="-15 0 -.98" size="1 20" euler="90 -20 -30" />
        </body>
        -->
    </worldbody>
    <actuator>
        <velocity gear=".34"  name="forwardMotor" joint="axle_front" kv="100" group="3"/>
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
    def __init__(self, hertz, max_timestep=2000):
        root = mjcf.from_xml_string(scooter_model)

        high = np.array([.1, np.finfo(np.float32).max, .5, .5, 1])

        self.action_space = gym.spaces.Discrete(5)
        #self.action_space = gym.spaces.Box(low=np.array([-.5]), high=np.array([.5]))
        self.observation_space = gym.spaces.Box(low=-high, high=high)

        self.hertz = hertz
        self.timestep = 0
        self.max_timestep = max_timestep

        for i in range(5):
            root.worldbody.add('light', diffuse=[.5, .5, .5], pos=[i * -15, 0, 10], dir=[0, 0, -1])

        self.physics = mjcf.Physics.from_mjcf_model(root)

        self.scene_option = mujoco.wrapper.core.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_ACTUATOR] = False

        self.frames = []
        self.past_pos = []
        self.lines = []
    
    def reset(self):
        self.physics.reset()
        self.timestep = 0
        self.frames=[]
        self.lines = []
        self.goal_pos = [-50, 0]
        self.done_next = False # to give reward after reaching goal
        #angle = np.random.uniform(0, 2 * math.pi)
        angle = -1.14
        self.physics.named.data.qpos['free_joint'][3:] = get_quaternion_from_euler(0, 0, angle)
        self.physics.named.data.ctrl['forwardMotor'] = 20
        self.physics.named.data.qvel['free_joint'] = [-5.8 * math.cos(angle), -5.8 * math.sin(angle), 0, 0, 0, 0]
        self.physics.step()
        return self.get_state()
    
    def step(self, action):
        steer_reward = 0
        if (abs(self.get_steer_goal()) < .49):
            steer = 0
            if action == 0: 
                steer = -.1
                steer_reward = -.05
            elif action == 1:
                steer = -.01
                steer_reward = -.02
            elif action == 3:
                steer = .01
                steer_reward = -.02
            elif action == 4:
                steer = .1
                steer_reward = -.05
            self.physics.named.data.ctrl['steering_pos'] += steer
        while True:
            self.physics.step()
            if self.timestep < self.physics.data.time * self.hertz:
                self.timestep += 1
                break
        done = False
        state = self.get_state()
        tilt_angle = state[0]
        tilt_velocity = state[1]
        goal_angle = state[4]
        reward = -tilt_angle**2 - .1 * tilt_velocity **2 - 2 * goal_angle **2 + steer_reward
        if self.timestep > self.max_timestep or self.done_next: 
            done = True
        elif abs(tilt_angle) > .6: 
            done = True
            reward = reward * (self.max_timestep - self.timestep)
        elif self.goal_reached():
            self.done_next = True
            reward = 40000
            print("let's go")
        info = {}
        return state, reward, done, info


    def get_tilt_angle(self):
        quat = self.physics.named.data.qpos["free_joint"][3:7]
        sinr_cosp = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        cosr_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2])
        return min(max(math.atan2(sinr_cosp, cosr_cosp), -1), 1)

    def get_tilt_velocity(self):
        return self.physics.named.data.qvel["free_joint"][3]
    
    def render_episode(self, model, render_name, max_time=10):
        obs = self.reset()
        done = False
        #pixels = self.render_pixels()
        #self.frames.append(pixels)
        while self.physics.data.time < max_time and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.step(action)
            #print(obs, reward, done)
            if len(self.frames) < self.physics.data.time * framerate:
                pixels = self.render_pixels()
                self.frames.append(pixels)
                #draw the path that the scooter has taken
                pos_homo = np.ones(4, dtype=float)
                pos_homo[:3] = self.physics.named.data.xpos['front_wheel'][:3]
                self.past_pos.append(pos_homo)
                camera = mujoco.Camera(self.physics, camera_id="top_view", height=1080, width=1920)
                camera_matrix = camera.matrix
                line = np.zeros([2, len(self.past_pos)], dtype=float)
                t = len(self.frames)
                self.physics.named.data.mocap_pos['camera'] = np.array([-10 - t * 15 /100, 0, 5 + t * 35/100])
                self.physics.named.data.mocap_quat['camera'] = get_quaternion_from_euler(0, -50 + t * 50 /100, -90 + t * 90/100)
                for i in range(len(self.past_pos)):
                    xs, ys, s = camera_matrix @ self.past_pos[i]
                    x = xs/s
                    y = ys/s
                    line[:,i] = np.array([x, y])
                self.lines.append(line)
        #print(self.lines)
        display_video(self.frames, render_name, framerate, lines=self.lines)
        self.reset()

    def render_pixels(self):
    #        return self.physics.render(camera_id=0, scene_option=self.scene_option)
        return self.physics.render(camera_id='top_view', scene_option=self.scene_option, width=1920, height=1080)
    
    def display_frame(self):
        img = PIL.Image.fromarray(self.render_pixels())
        img.show()


    def get_steer_angle(self):
        return min(max(self.physics.named.data.sensordata['steering_sensor'][0], -.5), .5)

    def get_steer_goal(self):
        return self.physics.named.data.ctrl['steering_pos']

    # gets the angle between the scooter's direction and the direction to the goal. 
    def get_goal_angle(self):
        scooter_backwheel_pos = self.physics.named.data.xpos['back_wheel'][:2]
        scooter_frontwheel_pos = self.physics.named.data.xpos['front_wheel'][:2]
        scooter_vector = scooter_frontwheel_pos - scooter_backwheel_pos
        normalized_scooter_vector = scooter_vector / np.linalg.norm(scooter_vector)
        goal_vector = self.goal_pos - scooter_frontwheel_pos
        normalized_goal_vector = goal_vector / np.linalg.norm(goal_vector)
        angle = np.arccos(np.dot(normalized_scooter_vector, normalized_goal_vector))
        if (normalized_goal_vector[0] * normalized_scooter_vector[1] - normalized_goal_vector[1] * normalized_scooter_vector[0] < 0):
            angle = -angle
        return angle

    def goal_reached(self):
        scooter_frontwheel_pos = self.physics.named.data.xpos['front_wheel'][:2]
        if np.linalg.norm(self.goal_pos - scooter_frontwheel_pos) < 3:
            return True
        return False


    def get_state(self):
        return np.array([self.get_tilt_angle(), self.get_tilt_velocity(), self.get_steer_angle(), self.get_steer_goal(), self.get_goal_angle()], dtype=np.float32)
        
    def simulate(self):
        self.reset()
        while self.physics.data.time < duration:
            self.physics.step()
            if len(self.frames) < self.physics.data.time * framerate:
                pixels = self.physics.render(camera_id=0, scene_option=self.scene_option)
                self.frames.append(pixels)
        display_video(self.frames, "test", framerate)

#sim = Sim(30)
#sim.simulate()
