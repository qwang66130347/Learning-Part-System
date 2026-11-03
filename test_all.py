import sys
import os
from pathlib import Path

ROOT = Path("c:/Users/warisara.ch/Desktop/DQN_Streamlit")
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

print("=" * 60)
print("TEST 1: Check Required Project Structure")
print("=" * 60)
required_files = [
    "app.py",
    "dqn_core.py",
    "shared_ui.py",
    "requirements.txt",
    "README.md",
    "data/simulated_learners.csv",
    "data/training_history_all_seeds.csv",
    "data/evaluation_results_all_seeds.csv",
    "data/evaluation_summary_by_seed.csv",
    "data/statistical_results_all_seeds.csv",
    "data/robustness_results_all_seeds.csv",
    "data/robustness_summary_by_seed.csv",
    "models/dqn_model_seed_42.pth",
    "models/dqn_model_seed_43.pth",
    "models/dqn_model_seed_44.pth",
    "models/dqn_model_seed_45.pth",
    "models/dqn_model_seed_46.pth",
    "pages/1_Learner_Simulation.py",
    "pages/2_Recommended_Learning_Path.py",
    "pages/3_Model_Evaluation.py",
    "pages/4_Training_Analytics.py",
]

missing = []
for f in required_files:
    p = ROOT / f
    if not p.exists():
        missing.append(f)
        print(f"  [MISSING] {f}")
    else:
        print(f"  [OK] {f} ({p.stat().st_size:,} bytes)")

assert not missing, f"Missing files: {missing}"
print("All required files exist!")

print()
print("=" * 60)
print("TEST 2: Verify CSV Loading and Schemas")
print("=" * 60)
import pandas as pd

csv_checks = {
    "data/simulated_learners.csv": ["learner_id", "profile", "ability_listening", "ability_reading", "ability_vocabulary", "ability_grammar", "learning_goal", "time_budget"],
    "data/training_history_all_seeds.csv": ["training_seed", "episode", "reward", "epsilon", "avg_ability_gain"],
    "data/evaluation_results_all_seeds.csv": ["learner_id", "agent", "episode_return", "ability_gain", "time_cost", "goal_achieved", "training_seed"],
    "data/evaluation_summary_by_seed.csv": ["training_seed", "agent", "episode_return", "ability_gain", "time_cost", "goal_achieved"],
    "data/statistical_results_all_seeds.csv": ["Metric", "Friedman_p", "Wilcoxon_DQN_vs_Random_p", "Wilcoxon_DQN_vs_Weakest_p"],
    "data/robustness_results_all_seeds.csv": ["scenario", "training_seed", "agent", "episode_return"],
    "data/robustness_summary_by_seed.csv": ["scenario", "training_seed", "agent", "episode_return"],
}

for csv_path, req_cols in csv_checks.items():
    df = pd.read_csv(ROOT / csv_path)
    print(f"Checking {csv_path}: {len(df):,} rows, {len(df.columns)} cols")
    for c in req_cols:
        assert c in df.columns, f"Missing column {c} in {csv_path}"
    print(f"  -> All required columns present: {req_cols[:4]}...")

print()
print("=" * 60)
print("TEST 3: Load DQN Checkpoints")
print("=" * 60)
from dqn_core import load_dqn_model, SEEDS

for seed in SEEDS:
    model = load_dqn_model(seed)
    assert model is not None, f"Failed to load seed {seed}"
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Seed {seed}: successfully loaded, parameters = {param_count}")

print()
print("=" * 60)
print("TEST 4: DQN Inference & Full Episode Simulation")
print("=" * 60)
from dqn_core import load_learners, get_learner_by_id, run_episode, select_action, get_q_values
import numpy as np

learners_df = load_learners()
learner = get_learner_by_id(learners_df, 1)
model_42 = load_dqn_model(42)

# Check state dimension and Q-value calculation
sample_state = np.zeros(11, dtype=np.float32)
q_vals = get_q_values(model_42, sample_state)
assert len(q_vals) == 4, f"Expected 4 Q-values, got {len(q_vals)}"
action = select_action(model_42, sample_state)
assert 0 <= action < 4, f"Invalid action: {action}"
print(f"  Single step Q-values: {q_vals}, action: {action}")

# Run episode
ep = run_episode(model_42, learner, sim_seed=123)
summary = ep["summary"]
print(f"  Learner 1 simulated: steps={summary['total_steps']}, gain={summary['ability_gain']:.2f}, return={summary['episode_return']:.2f}")
assert summary['total_steps'] > 0
assert len(ep['steps']) == summary['total_steps']

# Verify reproducibility
ep_rep = run_episode(model_42, learner, sim_seed=123)
assert abs(ep["summary"]["episode_return"] - ep_rep["summary"]["episode_return"]) < 1e-6
print("  Reproducibility check: PASSED (identical episode returns for same sim_seed)")

print()
print("=" * 60)
print("TEST 5: Verify Streamlit App & Pages Syntax")
print("=" * 60)
import py_compile
scripts = [
    "app.py",
    "shared_ui.py",
    "dqn_core.py",
    "pages/1_Learner_Simulation.py",
    "pages/2_Recommended_Learning_Path.py",
    "pages/3_Model_Evaluation.py",
    "pages/4_Training_Analytics.py",
]

for s in scripts:
    py_compile.compile(str(ROOT / s), doraise=True)
    print(f"  Syntax OK: {s}")

print()
print("=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
