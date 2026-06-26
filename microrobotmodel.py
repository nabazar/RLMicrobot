# -*- coding: utf-8 -*-
"""MicroRobotModel - Corrected Version"""

import numpy as np

class MicroRobotModel:
    def __init__(self, B, J, th):
        """
        Initialize the microrobot model.
        
        Args:
            B (np.ndarray): Magnetic field vector (3,)
            J (np.ndarray): Magnetic field gradient (3,3)
            th (float): Initial orientation angle (radians)
        """
        # Helical parameters
        self.nh = 5                       # Number of helical turns
        self.ths = np.linspace(0, self.nh * 2 * np.pi, 100)
        self.Rh = 1e-3                    # Helix radius (m)
        self.b = 1.1 * self.Rh            # Head radius (m)
        self.Lh = 4.4e-3                  # Helix length (m)
        self.ph = self.Lh / (self.nh * 2 * np.pi)  # Helix pitch parameter
        self.etah = 2.5e-3                # Fluid viscosity (Pa·s)
        self.ah = 0.05e-3                 # Filament radius (m)
        
        # State variables
        self.B = np.asarray(B)            # Magnetic field
        self.J = np.asarray(J)            # Field gradient
        self.th = th                      # Orientation angle
        
        # Workspace parameters
        self.r1 = 20e-3
        self.r2 = 25e-3
        self.rm = (self.r1 + self.r2) / 2
        
        # Physical constants
        self.phi = np.pi / 4              # Helix angle
        self.freq = 0.8                   # Operating frequency (Hz)
        self.mu0 = 4 * np.pi * 1e-7       # Permeability of free space
        
        # Magnetic properties
        self.chi = 2e-2                   # Magnetic susceptibility
        self.T_ratio = 1.0                # Volume fraction of magnetic material
        
        # Add dt for dynamics update
        self.dt = 0.01
        
    def spiralMicroRobot(self):
        """Generate the 3D shape of the helical microrobot."""
        a = self.ph
        r = self.Rh
        ths = self.ths
        shape = np.zeros((len(ths), 4))
        
        for i in range(len(ths)):
            if i < 2 * len(ths) / 3:
                r = 0.99 * r
            else:
                r = 0.95 * r
            shape[i, :] = [r * np.cos(ths[i]), a * ths[i], r * np.sin(ths[i]), 1]
        
        return shape
    
    def helicalMatrix(self):
        """
        Compute the resistive coupling matrix for a helical swimmer.
        
        Returns:
            np.ndarray: 2x2 coupling matrix relating force/torque to velocity.
        """
        n = self.nh
        R = self.Rh
        eta = self.etah
        t = self.ah
        phi = self.phi
        b = 1.1 * R
        
        # Viscous drag on the head (approximated as sphere)
        Dv = 6 * np.pi * eta * b
        Dw = 8 * np.pi * eta * b**3
        
        # Drag coefficients for the helix
        eps = 1e-12
        denom = np.log(0.36 * np.pi * R / (t * np.sin(phi) + eps)) + 0.5
        em = 4 * np.pi * eta / denom
        denom = np.log(0.36 * np.pi * R / (t * np.sin(phi) + eps))
        en = 2 * np.pi * eta / denom
        
        # Coupling matrix components
        A = (2 * np.pi * n * R * (em * np.cos(phi)**2 + en * np.sin(phi)**2) / 
             np.sin(phi)) - Dv
        Bmat = 2 * np.pi * n * (R**2) * (em - en) * np.cos(phi)
        D = (2 * np.pi * n * (R**3) * (en * np.cos(phi)**2 + em * np.sin(phi)**2) / 
             np.sin(phi)) + Dw
        
        H = np.array([[A, Bmat], [Bmat, D]])
        return H
    
    def MicroRobotDyn(self):
        """
        Compute the robot's dynamics given magnetic field and gradient.
        
        Returns:
            tuple: (v, w, M) where:
                v (np.ndarray): Translational velocity (3,)
                w (np.ndarray): Angular velocity (3,)
                M (np.ndarray): Magnetic moment (3,)
        """
        phi = self.phi
        
        # Calculate magnetic moment
        V = (4/3) * np.pi * self.b**3
        M_mag = V * self.T_ratio * (self.chi / (self.mu0 * (1.0 + self.chi)))
        
        # Moment vector along field direction
        B_norm = np.linalg.norm(self.B)
        if B_norm > 1e-12:
            M = M_mag * self.B / B_norm
        else:
            M = np.zeros(3)
        
        # Force and torque
        F = np.dot(M, self.J)                    # Magnetic force
        tau = np.cross(M, self.B)                # Magnetic torque
        
        # Coupling matrix (2x2)
        H = self.helicalMatrix()
        
        # Check if H is invertible
        if np.abs(np.linalg.det(H)) < 1e-12:
            H_inv = np.linalg.pinv(H)
        else:
            H_inv = np.linalg.inv(H)
        
        # For simplicity, use the z-component of force and torque
        # In a full 3D model, this would be more complex
        Fz = F[2] if len(F) > 2 else 0
        Tauz = tau[2] if len(tau) > 2 else 0
        
        # Solve for velocity and angular velocity along z
        FT = np.array([[Fz], [Tauz]])
        vz_wz = np.matmul(H_inv, FT)
        
        # Create full 3D velocity and angular velocity vectors
        v = np.zeros(3)
        w = np.zeros(3)
        v[0] = F[0] / (6 * np.pi * self.etah * self.b)  # Approximate drag
        v[1] = F[1] / (6 * np.pi * self.etah * self.b)
        v[2] = vz_wz[0, 0] if vz_wz.shape == (2, 1) else vz_wz[0]
        w[2] = vz_wz[1, 0] if vz_wz.shape == (2, 1) else vz_wz[1]
        
        return v, w, M
