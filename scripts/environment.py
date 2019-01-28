"""
Project 3: Multiagent Collaboration and Competition
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

Code inspired by implementation at https://github.com/danielnbarbosa/drlnd_collaboration_and_competition

Reimplementation of environment class.
"""
import platform
from unityagents import UnityEnvironment

class Unity_Multiagent():
    """Implementation of environment class."""
    def __init__(self, evaluation_only=False, seed=0):
        """Load platform specific file and initialize environment."""
        os = platform.system()
        if os=="Darwin":
            fn = "Tennis.app"
        elif os == "Linux":
            fn = "Tennis_Linux_NoVis/Tennis.x86_64" # NoVis/Headless required for cloud training
        else:
            print("Specified platform not available.")
            raise NotImplementedError
        self.env = UnityEnvironment(file_name="../data/" + fn, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.evaluation_only = evaluation_only

    def reset(self):
        """Reset Environment."""
        info = self.env.reset(train_mode=not self.evaluation_only)[self.brain_name]
        state = info.vector_observations
        return state

    def step(self, action):
        """Take a step in environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return state, reward, done
