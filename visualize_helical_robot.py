# -*- coding: utf-8 -*-
"""
Visualize Helical Microrobot in 3D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

from microrobotmodel import MicroRobotModel

print("="*60)
print("HELICAL MICROROBOT VISUALIZATION")
print("="*60)

# Create robot model
B = np.array([0.01, 0, 0.02])
J = np.zeros((3, 3))
th = 0
robot = MicroRobotModel(B, J, th)

# ============================================
# 1. Helical Shape
# ============================================

shape = robot.spiralMicroRobot()

fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot helix
ax1.plot(shape[:, 0]*1000, shape[:, 1]*1000, shape[:, 2]*1000, 
         'b-', linewidth=2, label='Helical Body')

# Plot head (sphere approximation)
u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, np.pi, 20)
head_radius = robot.b
head_x = head_radius * np.outer(np.cos(u), np.sin(v)) * 1000
head_y = head_radius * np.outer(np.sin(u), np.sin(v)) * 1000
head_z = (head_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + robot.Lh) * 1000
ax1.plot_surface(head_x, head_y, head_z, color='red', alpha=0.5)

# Plot tail
tail_x = [0, shape[0, 0]*1000]
tail_y = [0, shape[0, 1]*1000]
tail_z = [0, shape[0, 2]*1000]
ax1.plot(tail_x, tail_y, tail_z, 'g-', linewidth=2, label='Tail')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('Helical Microrobot Model')
ax1.legend()
ax1.set_aspect('equal')

plt.tight_layout()
plt.savefig('helical_robot_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. Robot at Different Orientations
# ============================================

fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

for idx, angle in enumerate(angles):
    row = idx // 3
    col = idx % 3
    
    # Rotate the shape
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_shape = shape[:, :3] @ rotation_matrix.T
    
    axes[row, col].plot(rotated_shape[:, 0]*1000, rotated_shape[:, 1]*1000, 
                       rotated_shape[:, 2]*1000, 'b-', linewidth=2)
    axes[row, col].set_xlabel('X (mm)')
    axes[row, col].set_ylabel('Y (mm)')
    axes[row, col].set_zlabel('Z (mm)')
    axes[row, col].set_title(f'Angle: {np.degrees(angle):.0f}°')
    axes[row, col].set_aspect('equal')

plt.tight_layout()
plt.savefig('robot_orientations.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Robot visualizations saved")
