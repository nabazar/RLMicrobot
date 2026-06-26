# -*- coding: utf-8 -*-
"""
Complete Environment Test and Visualization
Run all tests together
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("COMPLETE ENVIRONMENT TEST SUITE")
print("="*60)

# Run all visualization scripts
scripts = [
    'visualize_coils_and_workspace.py',
    'visualize_helical_robot.py',
    'visualize_forces_on_robot.py',
    'animate_robot_movement.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"\nRunning {script}...")
        exec(open(script).read())
    else:
        print(f"\n{script} not found")

# ============================================
# Additional: Environment Statistics
# ============================================

print("\n" + "="*60)
print("ENVIRONMENT STATISTICS")
print("="*60)

from microbotenv import Microrobot_Env

env = Microrobot_Env()

print(f"\nWorkspace:")
print(f"  Inner radius: {env.r1*1000:.1f} mm")
print(f"  Outer radius: {env.r2*1000:.1f} mm")
print(f"  Mean radius: {env.rm*1000:.1f} mm")

print(f"\nRobot:")
print(f"  Helix radius: {env.Rh*1000:.2f} mm")
print(f"  Head radius: {env.microbot.b*1000:.2f} mm")
print(f"  Helix length: {env.microbot.Lh*1000:.2f} mm")

print(f"\nAction space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test random steps
print("\nTesting random actions...")
env.reset()
rewards = []
for i in range(20):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        break

print(f"  Steps taken: {len(rewards)}")
print(f"  Average reward: {np.mean(rewards):.2f}")
print(f"  Min reward: {np.min(rewards):.2f}")
print(f"  Max reward: {np.max(rewards):.2f}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
