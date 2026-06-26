# -*- coding: utf-8 -*-
"""
Visualize Coils, Workspace, and Magnetic Field
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

from magneticfieldsim import MagneticFieldSim
from microbotenv import Microrobot_Env

print("="*60)
print("COILS, WORKSPACE, AND MAGNETIC FIELD VISUALIZATION")
print("="*60)

# Create environment
env = Microrobot_Env()
env.resetForTest()

# Create magnetic field object
P = np.array([0, 0, 0])
I = np.ones(6) * 1.0
mag = MagneticFieldSim(P, I)

# ============================================
# 1. 3D Coil System with Workspace
# ============================================

fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')

# Plot coils
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
coil_names = ['Coil X+', 'Coil Y+', 'Coil Z+', 'Coil X-', 'Coil Y-', 'Coil Z-']

for j in range(6):
    ax1.plot3D(mag.xx[j, :], mag.yy[j, :], mag.zz[j, :], 
              color=colors[j], linewidth=1.5, label=coil_names[j], alpha=0.7)

# Workspace circles
angles = np.linspace(0, 2*np.pi, 100)

# Inner circle (r1)
x1 = env.r1 * np.cos(angles)
y1 = env.r1 * np.sin(angles)
ax1.plot3D(x1, y1, np.zeros_like(angles), 'b-', linewidth=2, label='Inner (r1)')

# Outer circle (r2)
x2 = env.r2 * np.cos(angles)
y2 = env.r2 * np.sin(angles)
ax1.plot3D(x2, y2, np.zeros_like(angles), 'b-', linewidth=2, label='Outer (r2)')

# Mean path (rm)
xm = env.rm * np.cos(angles)
ym = env.rm * np.sin(angles)
ax1.plot3D(xm, ym, np.zeros_like(angles), 'g-', linewidth=3, label='Robot Path (rm)')

# Show robot position
ax1.scatter([env.x], [env.y], [0], color='green', s=150, label='Robot', edgecolors='black', linewidth=2)

# Show target
ax1.scatter([env.target_cart[0]], [env.target_cart[1]], [0], 
           color='red', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)

# Set labels
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_zlabel('Z (m)', fontsize=12)
ax1.set_title('6-Coil System with Robot Workspace', fontsize=14)
ax1.set_xlim([-env.r2*1.2, env.r2*1.2])
ax1.set_ylim([-env.r2*1.2, env.r2*1.2])
ax1.set_zlim([-env.r2*0.5, env.r2*0.5])
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coil_workspace_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. Magnetic Field Vector Field
# ============================================

print("\n2. Computing magnetic field vector field...")

n_points = 20
x = np.linspace(-15e-3, 15e-3, n_points)
y = np.linspace(-15e-3, 15e-3, n_points)
X, Y = np.meshgrid(x, y)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
B_mag = np.zeros_like(X)

for i in range(n_points):
    for j in range(n_points):
        P_test = np.array([X[i, j], Y[i, j], 0])
        mag_test = MagneticFieldSim(P_test, np.ones(6) * 1.0)
        B, _ = mag_test.IntegrationOfBiotSavar()
        Bx[i, j] = B[0]
        By[i, j] = B[1]
        B_mag[i, j] = np.linalg.norm(B)

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

# Vector field
scale = 8000
quiver = ax2.quiver(X*1000, Y*1000, Bx*scale, By*scale, B_mag*1000,
                   cmap='viridis', width=0.002)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_title('Magnetic Field Vector Field (XY Plane)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
plt.colorbar(quiver, ax=ax2, label='|B| (mT)')

# Contour
contour = ax3.contourf(X*1000, Y*1000, B_mag*1000, 20, cmap='hot')
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_title('Field Magnitude Contour (mT)')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax3)

plt.tight_layout()
plt.savefig('magnetic_field_vectors.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Vector field saved")
