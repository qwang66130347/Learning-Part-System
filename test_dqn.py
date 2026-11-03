import sys
sys.path.insert(0, 'c:/Users/warisara.ch/Desktop/DQN_Streamlit')
import os
os.chdir('c:/Users/warisara.ch/Desktop/DQN_Streamlit')

from dqn_core import *
import numpy as np

print('=== TEST A: Model Architecture ===')
model = DQNNetwork()
print(f'Network keys: {[n for n, _ in model.named_parameters()]}')

print()
print('=== TEST B: Model Loading ===')
for seed in SEEDS:
    m = load_dqn_model(seed)
    status = 'OK' if m is not None else 'FAILED'
    print(f'  seed {seed}: {status}')

print()
print('=== TEST C: DQN Inference ===')
test_state = np.array([0.5, 0.4, 0.3, 0.5, 0.0, 1.0, 0.85, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
model = load_dqn_model(42)
if model:
    q_vals = get_q_values(model, test_state)
    action = select_action(model, test_state)
    print(f'  Q-values shape: {q_vals.shape}')
    print(f'  Q-values: {q_vals}')
    action_name = ACTIONS[action]
    print(f'  Selected action: {action} ({action_name})')

print()
print('=== TEST D: Dataset Loading ===')
df = load_learners()
if df is not None:
    print(f'  Learners: {len(df)} rows, {len(df.columns)} cols')
    profiles = get_profiles(df)
    print(f'  Profiles: {profiles}')

print()
print('=== TEST E: Full Episode Simulation ===')
df = load_learners()
learner = get_learner_by_id(df, 1)
profile = str(learner['profile'])
print(f'  Learner 1 profile: {profile}')
model = load_dqn_model(42)
result = run_episode(model, learner, sim_seed=42)
summary = result['summary']
print(f'  Steps: {summary["total_steps"]}')
print(f'  Init ability: {summary["init_avg_ability"]:.2f}')
print(f'  Final ability: {summary["final_avg_ability"]:.2f}')
print(f'  Ability gain: {summary["ability_gain"]:.2f}')
print(f'  Episode return: {summary["episode_return"]:.2f}')
print(f'  Goal achieved: {summary["goal_achieved"]}')
print(f'  Action counts: {summary["action_counts"]}')

result2 = run_episode(model, learner, sim_seed=42)
reproducible = abs(result['summary']['episode_return'] - result2['summary']['episode_return']) < 1e-6
print(f'  Reproducible (same seed): {reproducible}')

print()
print('ALL TESTS PASSED')
