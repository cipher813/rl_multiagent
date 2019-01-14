"""
Project 3: Multiagent
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

BUG due to bug in built version of Unity, cannot load two unity environments in same loop
https://github.com/Unity-Technologies/ml-agents/issues/1167
This may be fixed in mlagents.envs but have not verified.
"""
from agents.MADDPG import MADDPG
from util import *

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_multiagent/"
# PATH = "/home/cipher813/rl_continuous_control/"
# PATH = "/home/bcm822_gmail_com/rl_continuous_control/"
RESULT_PATH = PATH + "results/"

# Number of agents to train.  Must be "single" or "multi."
# BUG cannot run more than one unity environment in same loop (so cannot be "both")
TRAIN_MODE = "multi"

env_dict = {
            "Tennis":["unity","Tennis.app","multi",0.0],
            # "Crawler":["unity","Crawler.app","multi",400.0], #400 dynamic, 2000 static
            # "Crawler":["unity","Crawler_Linux_NoVis/Crawler.x86_64","multi",400.0], #400 dynamic, 2000 static
            # "Reacher20":["unity","Reacher20.app","multi",0.0], # 30.0
            # "Reacher20":["unity","Reacher_Linux_NoVis2/Reacher.x86_64","multi",30.0],
            "Reacher1":["unity","Reacher1.app","single",30.0], # 30.0
            # "Reacher1":["unity","Reacher_Linux_NoVis1/Reacher.x86_64","single",30.0],
            "Pendulum":["gym","Pendulum-v0","single",2000.0],
            "BipedalWalker":["gym","BipedalWalker-v2","single",300.0] # 300.0
            }

agent_dict = {
              "MADDPG":[MADDPG,"multi"],
              # "DDPGplus":[DDPGplus,"both"],
              # "D4PG":[D4PG,"single"],
              # "TD3":[TD3,"both"]
             }

results = train_envs(PATH, env_dict, agent_dict,TRAIN_MODE)
print(results)
