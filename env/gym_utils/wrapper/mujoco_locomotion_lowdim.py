"""
Environment wrapper for Gym environments (MuJoCo locomotion tasks) with state observations.

For consistency, we will use Dict{} for the observation space, with the key "state" for the state observation.
"""

import numpy as np
import gym
from gym import spaces
import imageio


class MujocoLocomotionLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
        render_fps=30,
    ):
        self.env = env
        self.render_fps = render_fps
        self.video_writer = None

        # setup spaces
        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        obs_example = self.env.reset()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, **kwargs):
        """Ignore passed-in arguments like seed"""
        options = kwargs.get("options", {})

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if "video_path" in options:
          self.video_writer = imageio.get_writer(
              options["video_path"],
              format="FFMPEG",
              mode="I",
              fps=self.render_fps,
              macro_block_size=None,
          )

        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        raw_obs = self.env.reset()

        # normalize
        obs = self.normalize_obs(raw_obs)
        return {"state": obs}

    def normalize_obs(self, obs):
        obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        return obs.astype(np.float32, copy=False)

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(raw_action)

        # normalize
        obs = self.normalize_obs(raw_obs)
        if self.video_writer is not None:
            self.video_writer.append_data(self.render(mode="rgb_array"))
        return {"state": obs}, reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()
