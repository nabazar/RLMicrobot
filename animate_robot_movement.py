# -*- coding: utf-8 -*-
"""
Animate Robot Movement in Workspace
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

from microbotenv import Microrobot_Env

print("="*60)
print("ANIMATING ROBOT MOVEMENT")
print("="*60)

# Create environment
env = Microrobot_Env()
env.resetForTest()

# Run simulation and store data
positions = []
rewards = []
states = []

print("Running simulation...")
for i in range(100):
    error = env.goal - env.theta
    error = env.correct_for_wrap_rad(error)
    action = np.array([np.clip(error * 0.5, -1, 1), 0.0, 0.0])
    
    next_state, reward, done, info = env.step(action)
    positions.append([env.x, env.y])
    rewards.append(reward)
    states.append(next_state)
    
    if done:
        print(f"Goal reached at step {i+1}")
        break

positions = np.array(positions)
print(f"Total steps: {len(positions)}")

# ============================================
# 1. Static Path Plot
# ============================================

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Path
ax1.plot(positions[:,0]*1000, positions[:,1]*1000, 'g-', linewidth=2, label='Path')
ax1.scatter(positions[0,0]*1000, positions[0,1]*1000, color='blue', s=100, label='Start')
ax1.scatter(positions[-1,0]*1000, positions[-1,1]*1000, color='red', s=100, label='End')

# Workspace
angles = np.linspace(0, 2*np.pi, 100)
ax1.plot(env.r1*np.cos(angles)*1000, env.r1*np.sin(angles)*1000, 'b--', alpha=0.5, label='r1')
ax1.plot(env.r2*np.cos(angles)*1000, env.r2*np.sin(angles)*1000, 'b--', alpha=0.5, label='r2')
ax1.plot(env.rm*np.cos(angles)*1000, env.rm*np.sin(angles)*1000, 'g--', alpha=0.3, label='rm')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_title('Robot Path')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Reward
ax2.plot(rewards, 'b-', linewidth=2)
ax2.set_xlabel('Step')
ax2.set_ylabel('Reward')
ax2.set_title('Reward vs Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('robot_path_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. Animate Movement
# ============================================

print("\nCreating animation...")

fig2, ax3 = plt.subplots(figsize=(8, 8))

# Setup workspace
angles = np.linspace(0, 2*np.pi, 100)
ax3.plot(env.r1*np.cos(angles)*1000, env.r1*np.sin(angles)*1000, 'b--', alpha=0.5, label='r1')
ax3.plot(env.r2*np.cos(angles)*1000, env.r2*np.sin(angles)*1000, 'b--', alpha=0.5, label='r2')
ax3.plot(env.rm*np.cos(angles)*1000, env.rm*np.sin(angles)*1000, 'g--', alpha=0.3, label='rm')

# Plot target
target_x = env.target_cart[0] * 1000
target_y = env.target_cart[1] * 1000
ax3.scatter(target_x, target_y, color='red', s=200, marker='*', label='Target', zorder=5)

# Initialize robot and path
robot_point, = ax3.plot([], [], 'go', markersize=10, label='Robot')
path_line, = ax3.plot([], [], 'g-', alpha=0.5, linewidth=2)

ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_title('Robot Movement Animation')
ax3.set_aspect('equal')
ax3.set_xlim([-30, 30])
ax3.set_ylim([-30, 30])
ax3.legend()
ax3.grid(True, alpha=0.3)

# Animation function
def animate(i):
    if i < len(positions):
        x = positions[:i+1, 0] * 1000
        y = positions[:i+1, 1] * 1000
        robot_point.set_data([x[-1]], [y[-1]])
        path_line.set_data(x, y)
    return robot_point, path_line

# Create animation
anim = FuncAnimation(fig2, animate, frames=len(positions), 
                    interval=100, blit=True, repeat=False)

# Save animation
anim.save('robot_animation.gif', writer='pillow', fps=10)
print("Animation saved as robot_animation.gif")

# Display in Jupyter
from IPython.display import HTML
HTML(anim.to_jshtml())

print("\n✓ Animation created successfully!")
