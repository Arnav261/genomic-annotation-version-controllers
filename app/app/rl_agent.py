"""
RL scaffold for conflict resolution. Use offline training with Stable-Baselines3 (PPO).
"""
from typing import List, Dict, Any, Tuple, Optional
import gym
import numpy as np
from gym import spaces
import logging
import os

logger = logging.getLogger(__name__)
MAX_ANN = 12
EMB_DIM = 768

class AnnotationEnv(gym.Env):
    def __init__(self, embeddings: np.ndarray, evidence_scores: List[float], annotation_ids: List[str]):
        super().__init__()
        self.embeddings = embeddings
        self.n = embeddings.shape[0]
        self.evidence_scores = np.array(evidence_scores + [0.0] * (MAX_ANN - len(evidence_scores)))[:MAX_ANN]
        self.annotation_ids = annotation_ids + ["pad"] * (MAX_ANN - len(annotation_ids))
        obs_dim = MAX_ANN * EMB_DIM + MAX_ANN
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_ANN * MAX_ANN + MAX_ANN)
        self._build_obs()
        self._external_reward = None
        self.done = False

    def _build_obs(self):
        emb_padded = np.zeros((MAX_ANN, EMB_DIM), dtype=np.float32)
        for i in range(min(self.n, MAX_ANN)):
            emb_padded[i, :] = self.embeddings[i, :EMB_DIM].astype(np.float32)
        flat = emb_padded.flatten()
        obs = np.concatenate([flat, self.evidence_scores.astype(np.float32)])
        self._obs = obs

    def reset(self):
        self.done = False
        self._external_reward = None
        self._build_obs()
        return self._obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        info = {}
        reward = 0.0
        if action < MAX_ANN * MAX_ANN:
            i = action // MAX_ANN
            j = action % MAX_ANN
            info["action"] = f"merge_{i}_{j}"
        else:
            idx = action - MAX_ANN * MAX_ANN
            info["action"] = f"reject_{idx}"
        if self._external_reward is not None:
            reward = self._external_reward
            self._external_reward = None
            self.done = True
        reward = reward - 0.01
        return self._obs, float(reward), bool(self.done), info

    def set_feedback_reward(self, reward: float):
        self._external_reward = float(reward)

def train_agent(env_creator, total_timesteps: int = 10000, save_path: str = "app/model_data/rl_agent.zip"):
    try:
        from stable_baselines3 import PPO
    except Exception:
        raise RuntimeError("stable-baselines3 required for training")
    env = env_creator()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info(f"Saved RL agent to {save_path}")
    return save_path