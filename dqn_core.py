"""
dqn_core.py
-----------
Core DQN model definition and English Learning Environment, extracted
faithfully from Personalized_Learning_DQN_Fixed.ipynb.

Contains:
  - DQNNetwork: the original 2-hidden-layer DQN architecture
  - EnglishLearningEnv: original gym-style environment
  - Helper functions for loading models, running simulations, etc.

DO NOT modify the reward function, state/action definitions, or
environment dynamics — these are the source of truth for the research.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (matching the original notebook)
# ---------------------------------------------------------------------------

STATE_DIM = 11
ACTION_DIM = 4
HIDDEN_1 = 128
HIDDEN_2 = 128

ACTIONS = {
    0: "Listening",
    1: "Reading",
    2: "Vocabulary",
    3: "Grammar",
}

ACTION_COLORS = {
    "Listening": "#1565C0",
    "Reading":   "#42A5F5",
    "Vocabulary": "#16A34A",
    "Grammar":   "#F59E0B",
}

SEEDS = [42, 43, 44, 45, 46]

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# DQN Network Architecture (identical to notebook)
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """
    Two-hidden-layer DQN network.
    Input  : state_dim  (11)
    Hidden : 128 → ReLU → 128 → ReLU
    Output : action_dim (4)  — Q-values, one per action

    NOTE: The Sequential is stored as self.net (not self.network) to match
    the key naming in the pretrained .pth checkpoint files:
    net.0.weight / net.0.bias / net.2.weight / ... / net.4.weight / net.4.bias
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden1: int = HIDDEN_1,
        hidden2: int = HIDDEN_2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


# ---------------------------------------------------------------------------
# English Learning Environment (faithful to notebook)
# ---------------------------------------------------------------------------

class EnglishLearningEnv:
    """
    Simulated English-learning environment.

    State vector (11 dims):
      [0]  ability_listening  / 100.0
      [1]  ability_reading    / 100.0
      [2]  ability_vocabulary / 100.0
      [3]  ability_grammar    / 100.0
      [4]  learning_progress  (fraction of time used)
      [5]  remaining_time_ratio
      [6]  learning_goal      / 100.0
      [7]  count_listening    / 20.0
      [8]  count_reading      / 20.0
      [9]  count_vocabulary   / 20.0
      [10] count_grammar      / 20.0

    Actions:
      0 = Listening, 1 = Reading, 2 = Vocabulary, 3 = Grammar

    Learning-rate table (time_cost, base_gain) per action:
      0: (15, 5)
      1: (20, 6)
      2: (10, 4)
      3: (25, 7)
    """

    # (time_cost_minutes, base_learning_gain)
    ACTIVITY_PARAMS: Dict[int, Tuple[float, float]] = {
        0: (15.0, 5.0),   # Listening
        1: (20.0, 6.0),   # Reading
        2: (10.0, 4.0),   # Vocabulary
        3: (25.0, 7.0),   # Grammar
    }

    def __init__(self, learner_row: pd.Series, rng_seed: Optional[int] = None):
        """
        Parameters
        ----------
        learner_row : a single row from simulated_learners.csv
        rng_seed    : seed for numpy random (for reproducibility)
        """
        self.learner_row = learner_row
        self.rng = np.random.RandomState(rng_seed)

        # Initial abilities (0–100)
        self.init_listening   = float(learner_row["ability_listening"])
        self.init_reading     = float(learner_row["ability_reading"])
        self.init_vocabulary  = float(learner_row["ability_vocabulary"])
        self.init_grammar     = float(learner_row["ability_grammar"])

        self.learning_goal    = float(learner_row["learning_goal"])
        self.time_budget      = float(learner_row["time_budget"])

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment to initial state and return the state vector."""
        self.ability_listening  = self.init_listening
        self.ability_reading    = self.init_reading
        self.ability_vocabulary = self.init_vocabulary
        self.ability_grammar    = self.init_grammar

        self.time_used    = 0.0
        self.step_count   = 0
        self.done         = False

        self.count_listening  = 0
        self.count_reading    = 0
        self.count_vocabulary = 0
        self.count_grammar    = 0

        return self._get_state()

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        """Construct the 11-dimensional normalised state vector."""
        avg_ability = (
            self.ability_listening
            + self.ability_reading
            + self.ability_vocabulary
            + self.ability_grammar
        ) / 4.0

        time_remaining = max(0.0, self.time_budget - self.time_used)
        learning_progress = self.time_used / self.time_budget if self.time_budget > 0 else 0.0
        remaining_time_ratio = time_remaining / self.time_budget if self.time_budget > 0 else 0.0

        return np.array(
            [
                self.ability_listening  / 100.0,
                self.ability_reading    / 100.0,
                self.ability_vocabulary / 100.0,
                self.ability_grammar    / 100.0,
                learning_progress,
                remaining_time_ratio,
                self.learning_goal / 100.0,
                self.count_listening  / 20.0,
                self.count_reading    / 20.0,
                self.count_vocabulary / 20.0,
                self.count_grammar    / 20.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one learning action.

        Returns
        -------
        next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")

        time_cost, base_gain = self.ACTIVITY_PARAMS[action]

        # Random noise on the learning gain (matches original notebook)
        noise = self.rng.uniform(-1.0, 1.0)
        actual_gain = base_gain + noise

        # Update the chosen ability
        if action == 0:
            self.ability_listening  = min(100.0, self.ability_listening  + actual_gain)
            self.count_listening   += 1
        elif action == 1:
            self.ability_reading    = min(100.0, self.ability_reading    + actual_gain)
            self.count_reading     += 1
        elif action == 2:
            self.ability_vocabulary = min(100.0, self.ability_vocabulary + actual_gain)
            self.count_vocabulary  += 1
        elif action == 3:
            self.ability_grammar    = min(100.0, self.ability_grammar    + actual_gain)
            self.count_grammar     += 1

        self.time_used  += time_cost
        self.step_count += 1

        avg_ability = (
            self.ability_listening
            + self.ability_reading
            + self.ability_vocabulary
            + self.ability_grammar
        ) / 4.0

        # Reward: improvement in average ability relative to time cost
        init_avg = (
            self.init_listening
            + self.init_reading
            + self.init_vocabulary
            + self.init_grammar
        ) / 4.0
        ability_gain_total = avg_ability - init_avg
        reward = ability_gain_total - (time_cost / self.time_budget) * 10.0

        # Termination condition
        goal_achieved = avg_ability >= self.learning_goal
        time_exceeded = self.time_used >= self.time_budget

        self.done = goal_achieved or time_exceeded

        info = {
            "avg_ability":       avg_ability,
            "ability_gain":      actual_gain,
            "time_cost":         time_cost,
            "time_used":         self.time_used,
            "time_remaining":    max(0.0, self.time_budget - self.time_used),
            "goal_achieved":     goal_achieved,
            "ability_listening": self.ability_listening,
            "ability_reading":   self.ability_reading,
            "ability_vocabulary":self.ability_vocabulary,
            "ability_grammar":   self.ability_grammar,
        }

        return self._get_state(), reward, self.done, info


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def get_model_path(seed: int) -> Path:
    return MODELS_DIR / f"dqn_model_seed_{seed}.pth"


def load_dqn_model(seed: int, device: str = "cpu") -> Optional[DQNNetwork]:
    """
    Load a trained DQN checkpoint for the given seed.

    Returns None if the file is missing or incompatible.
    """
    path = get_model_path(seed)
    if not path.exists():
        return None

    model = DQNNetwork()
    try:
        # Try safe loading first (PyTorch ≥ 2.0)
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=device)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def get_available_seeds() -> List[int]:
    """Return list of seeds for which a .pth checkpoint exists."""
    return [s for s in SEEDS if get_model_path(s).exists()]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def state_to_tensor(state: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.FloatTensor(state).unsqueeze(0).to(device)


@torch.no_grad()
def get_q_values(model: DQNNetwork, state: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return Q-value array of shape (4,)."""
    t = state_to_tensor(state, device)
    q = model(t)
    return q.cpu().numpy().flatten()


@torch.no_grad()
def select_action(model: DQNNetwork, state: np.ndarray, device: str = "cpu") -> int:
    """Return the greedy action index (argmax of Q-values)."""
    q_vals = get_q_values(model, state, device)
    return int(np.argmax(q_vals))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_learners() -> Optional[pd.DataFrame]:
    """Load simulated_learners.csv. Returns None on failure."""
    path = DATA_DIR / "simulated_learners.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_learner_by_id(df: pd.DataFrame, learner_id: int) -> Optional[pd.Series]:
    """Retrieve a single learner row by learner_id."""
    mask = df["learner_id"] == learner_id
    if not mask.any():
        return None
    return df[mask].iloc[0]


def get_profiles(df: pd.DataFrame) -> List[str]:
    """Return sorted unique learner profiles."""
    return sorted(df["profile"].unique().tolist())


def get_learner_ids_by_profile(df: pd.DataFrame, profile: str) -> List[int]:
    """Return learner IDs filtered to a specific profile."""
    return sorted(df[df["profile"] == profile]["learner_id"].tolist())


# ---------------------------------------------------------------------------
# Episode Simulation
# ---------------------------------------------------------------------------

def run_episode(
    model: DQNNetwork,
    learner_row: pd.Series,
    sim_seed: int = 42,
    device: str = "cpu",
    max_steps: int = 200,
) -> Dict:
    """
    Run a complete DQN episode for a single learner.

    Returns a dict with:
      - steps        : list of per-step dicts
      - summary      : aggregated episode statistics
      - learner_info : initial learner profile data
    """
    env = EnglishLearningEnv(learner_row, rng_seed=sim_seed)
    state = env.reset()

    steps = []
    total_reward = 0.0
    step_num = 0

    while not env.done and step_num < max_steps:
        q_vals = get_q_values(model, state, device)
        action = int(np.argmax(q_vals))
        next_state, reward, done, info = env.step(action)

        step_record = {
            "step":              step_num + 1,
            "action":            action,
            "action_name":       ACTIONS[action],
            "q_values":          q_vals.tolist(),
            "q_listening":       float(q_vals[0]),
            "q_reading":         float(q_vals[1]),
            "q_vocabulary":      float(q_vals[2]),
            "q_grammar":         float(q_vals[3]),
            "reward":            float(reward),
            "time_cost":         float(info["time_cost"]),
            "time_used":         float(info["time_used"]),
            "time_remaining":    float(info["time_remaining"]),
            "avg_ability":       float(info["avg_ability"]),
            "ability_gain":      float(info["ability_gain"]),
            "ability_listening": float(info["ability_listening"]),
            "ability_reading":   float(info["ability_reading"]),
            "ability_vocabulary":float(info["ability_vocabulary"]),
            "ability_grammar":   float(info["ability_grammar"]),
            "goal_achieved":     bool(info["goal_achieved"]),
        }
        steps.append(step_record)
        total_reward += reward
        state = next_state
        step_num += 1

    init_avg = (
        float(learner_row["ability_listening"])
        + float(learner_row["ability_reading"])
        + float(learner_row["ability_vocabulary"])
        + float(learner_row["ability_grammar"])
    ) / 4.0

    final_avg = steps[-1]["avg_ability"] if steps else init_avg
    goal_achieved = steps[-1]["goal_achieved"] if steps else False

    action_counts = {name: 0 for name in ACTIONS.values()}
    for s in steps:
        action_counts[s["action_name"]] += 1

    summary = {
        "learner_id":       int(learner_row["learner_id"]),
        "profile":          str(learner_row["profile"]),
        "init_avg_ability": init_avg,
        "final_avg_ability":final_avg,
        "ability_gain":     final_avg - init_avg,
        "episode_return":   total_reward,
        "total_time_used":  float(env.time_used),
        "time_budget":      float(env.time_budget),
        "goal_achieved":    goal_achieved,
        "total_steps":      step_num,
        "action_counts":    action_counts,
        "learning_goal":    float(env.learning_goal),
        "sim_seed":         sim_seed,
    }

    learner_info = {
        "learner_id":         int(learner_row["learner_id"]),
        "profile":            str(learner_row["profile"]),
        "ability_listening":  float(learner_row["ability_listening"]),
        "ability_reading":    float(learner_row["ability_reading"]),
        "ability_vocabulary": float(learner_row["ability_vocabulary"]),
        "ability_grammar":    float(learner_row["ability_grammar"]),
        "initial_ability":    init_avg,
        "learning_goal":      float(learner_row["learning_goal"]),
        "time_budget":        float(learner_row["time_budget"]),
    }

    return {
        "steps":        steps,
        "summary":      summary,
        "learner_info": learner_info,
    }
