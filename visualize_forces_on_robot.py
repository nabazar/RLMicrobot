# -*- coding: utf-8 -*-
"""
Visualize Forces and Torques on Microrobot
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

from magneticfieldsim import MagneticFieldSim
from microrobotmodel import MicroRobotModel

print("="*60)
print("FORCES AND TORQUES ON MICROROBOT")
print("="*60)

# ============================================
# 1. Force and Torque at Different Positions
# ============================================

positions = []
forces = []
torques = []
field_mags = []

for angle in np.linspace(0, 2*np.pi, 12):
    rm = 22.5e-3
    P = np.array([rm * np.cos(angle), rm * np.sin(angle), 0])
    positions.append(P)
    
    # Calculate field
    mag = MagneticFieldSim(P, np.ones(6) * 1.0)
    B, J = mag.IntegrationOfBiotSavar()
    field_mags.append(np.linalg.norm(B))
    
    # Create robot at this position
    robot = MicroRobotModel(B, J, angle)
    v, w, M = robot.MicroRobotDyn()
    forces.append(v)
    torques.append(w)

forces = np.array(forces)
torques = np.array(torques)
field_mags = np.array(field_mags)

fig1, axes = plt.subplots(2, 2, figsize=(12, 10))

# Force magnitude vs position
axes[0, 0].plot(np.degrees(np.linspace(0, 360, 12)), 
               np.linalg.norm(forces, axis=1)*1000, 'bo-', linewidth=2)
axes[0, 0].set_xlabel('Position Angle (degrees)')
axes[0, 0].set_ylabel('Force Magnitude (mN)')
axes[0, 0].set_title('Force Magnitude vs Position')
axes[0, 0].grid(True, alpha=0.3)

# Torque magnitude vs position
axes[0, 1].plot(np.degrees(np.linspace(0, 360, 12)), 
               np.linalg.norm(torques, axis=1)*1000, 'ro-', linewidth=2)
axes[0, 1].set_xlabel('Position Angle (degrees)')
axes[0, 1].set_ylabel('Torque Magnitude (mN·m)')
axes[0, 1].set_title('Torque Magnitude vs Position')
axes[0, 1].grid(True, alpha=0.3)

# Field magnitude vs position
axes[1, 0].plot(np.degrees(np.linspace(0, 360, 12)), 
               field_mags*1000, 'go-', linewidth=2)
axes[1, 0].set_xlabel('Position Angle (degrees)')
axes[1, 0].set_ylabel('Field Magnitude (mT)')
axes[1, 0].set_title('Field Magnitude vs Position')
axes[1, 0].grid(True, alpha=0.3)

# Force vs field correlation
axes[1, 1].scatter(field_mags*1000, np.linalg.norm(forces, axis=1)*1000, s=50)
axes[1, 1].set_xlabel('Field Magnitude (mT)')
axes[1, 1].set_ylabel('Force Magnitude (mN)')
axes[1, 1].set_title('Force vs Field Correlation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forces_torques_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. 3D Force Vector Visualization
# ============================================

fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot workspace
angles = np.linspace(0, 2*np.pi, 100)
rm = 22.5e-3
ax2.plot(rm * np.cos(angles), rm * np.sin(angles), np.zeros_like(angles), 'g--', alpha=0.5)

# Plot force vectors
for i, pos in enumerate(positions):
    force = forces[i]
    force_norm = np.linalg.norm(force)
    if force_norm > 0:
        # Normalize force for visualization
        force_dir = force / force_norm
        scale = 0.005
        ax2.quiver(pos[0], pos[1], 0, 
                  force_dir[0]*scale, force_dir[1]*scale, 0,
                  color='red', arrow_length_ratio=0.3, alpha=0.8, 
                  label='Force' if i == 0 else "")

# Plot torque vectors (along z)
for i, pos in enumerate(positions):
    torque = torques[i]
    torque_z = torque[2]
    if abs(torque_z) > 0:
        ax2.quiver(pos[0], pos[1], 0, 
                  0, 0, torque_z * 0.001,
                  color='blue', arrow_length_ratio=0.3, alpha=0.8,
                  label='Torque' if i == 0 else "")

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title('Force and Torque Vectors on Robot')
ax2.legend()
ax2.set_xlim([-0.03, 0.03])
ax2.set_ylim([-0.03, 0.03])

plt.tight_layout()
plt.savefig('force_torque_vectors_3d.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Force and torque visualizations saved")
