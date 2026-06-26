# -*- coding: utf-8 -*-
"""
Magnetic Field Simulation - Hybrid Approach
Combines analytical solution + interpolation for speed and accuracy
"""
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.interpolate import RegularGridInterpolator
import pickle
import os

class MagneticFieldSim:
    """Hybrid magnetic field solver with interpolation for RL"""
    
    def __init__(self, P=None, I=None, use_interpolation=True):
        """
        Initialize field simulator
        
        Parameters:
        P: observation point (optional)
        I: coil currents (optional)
        use_interpolation: if True, use pre-computed grid (fast)
                           if False, use direct calculation (accurate)
        """
        self.P = P if P is not None else np.zeros(3)
        self.I = I if I is not None else np.ones(6)
        
        # Coil parameters
        self.Rc = 30e-3  # coil radius
        self.nl = 5      # number of layers
        self.Lw = 2 * self.Rc
        self.CoilPositions = np.array([
            [self.Lw/2, 0, 0], [0, self.Lw/2, 0], [0, 0, self.Lw/2],
            [-self.Lw/2, 0, 0], [0, -self.Lw/2, 0], [0, 0, -self.Lw/2]
        ])
        
        self.use_interpolation = use_interpolation
        
        # Load or create interpolation grid
        self.grid_file = 'magnetic_field_grid.pkl'
        if use_interpolation and os.path.exists(self.grid_file):
            self._load_grid()
        elif use_interpolation:
            self._create_grid()
            self._save_grid()
    
    def _coil_field_analytical(self, P, coil_idx, I):
        """
        Exact analytical field from a circular coil using elliptic integrals
        """
        mu0 = 4 * np.pi * 1e-7
        Rc = self.Rc
        center = self.CoilPositions[coil_idx]
        
        # Determine coil orientation
        axis = np.argmax(np.abs(center))
        
        # Transform to coil coordinate system
        if axis == 0:  # Coil in YZ plane
            # Rotate coordinates
            r = np.sqrt((P[1] - center[1])**2 + (P[2] - center[2])**2)
            z = P[0] - center[0]
            
            if r < 1e-12 and abs(z) < 1e-12:
                return np.array([mu0 * I / (2 * Rc), 0, 0])
            
            k2 = 4 * Rc * r / ((Rc + r)**2 + z**2)
            if k2 >= 1:  # Numerical stability
                k2 = 0.999999
            k = np.sqrt(k2)
            
            try:
                K = ellipk(k2)
                E = ellipe(k2)
            except:
                K = np.pi/2  # Fallback
                E = np.pi/2
            
            # Cylindrical components
            Br = (mu0 * I / (2 * np.pi)) * (z / (r * np.sqrt((Rc + r)**2 + z**2))) * \
                 (-K + (Rc**2 + r**2 + z**2) / ((Rc - r)**2 + z**2) * E)
            
            Bz = (mu0 * I / (2 * np.pi)) * (1 / np.sqrt((Rc + r)**2 + z**2)) * \
                 (K + (Rc**2 - r**2 - z**2) / ((Rc - r)**2 + z**2) * E)
            
            # Convert to Cartesian
            if r > 0:
                By = Br * (P[1] - center[1]) / r
                Bz_cart = Br * (P[2] - center[2]) / r
            else:
                By = 0
                Bz_cart = 0
            
            return np.array([Bz, By, Bz_cart])
        
        elif axis == 1:  # Coil in XZ plane
            r = np.sqrt((P[0] - center[0])**2 + (P[2] - center[2])**2)
            z = P[1] - center[1]
            
            if r < 1e-12 and abs(z) < 1e-12:
                return np.array([0, mu0 * I / (2 * Rc), 0])
            
            k2 = 4 * Rc * r / ((Rc + r)**2 + z**2)
            if k2 >= 1:
                k2 = 0.999999
            
            try:
                K = ellipk(k2)
                E = ellipe(k2)
            except:
                K = np.pi/2
                E = np.pi/2
            
            Br = (mu0 * I / (2 * np.pi)) * (z / (r * np.sqrt((Rc + r)**2 + z**2))) * \
                 (-K + (Rc**2 + r**2 + z**2) / ((Rc - r)**2 + z**2) * E)
            
            Bz = (mu0 * I / (2 * np.pi)) * (1 / np.sqrt((Rc + r)**2 + z**2)) * \
                 (K + (Rc**2 - r**2 - z**2) / ((Rc - r)**2 + z**2) * E)
            
            if r > 0:
                Bx = Br * (P[0] - center[0]) / r
                Bz_cart = Br * (P[2] - center[2]) / r
            else:
                Bx = 0
                Bz_cart = 0
            
            return np.array([Bx, Bz, Bz_cart])
        
        else:  # axis == 2, Coil in XY plane
            r = np.sqrt((P[0] - center[0])**2 + (P[1] - center[1])**2)
            z = P[2] - center[2]
            
            if r < 1e-12 and abs(z) < 1e-12:
                return np.array([0, 0, mu0 * I / (2 * Rc)])
            
            k2 = 4 * Rc * r / ((Rc + r)**2 + z**2)
            if k2 >= 1:
                k2 = 0.999999
            
            try:
                K = ellipk(k2)
                E = ellipe(k2)
            except:
                K = np.pi/2
                E = np.pi/2
            
            Br = (mu0 * I / (2 * np.pi)) * (z / (r * np.sqrt((Rc + r)**2 + z**2))) * \
                 (-K + (Rc**2 + r**2 + z**2) / ((Rc - r)**2 + z**2) * E)
            
            Bz = (mu0 * I / (2 * np.pi)) * (1 / np.sqrt((Rc + r)**2 + z**2)) * \
                 (K + (Rc**2 - r**2 - z**2) / ((Rc - r)**2 + z**2) * E)
            
            if r > 0:
                Bx = Br * (P[0] - center[0]) / r
                By = Br * (P[1] - center[1]) / r
            else:
                Bx = 0
                By = 0
            
            return np.array([Bx, By, Bz])
    
    def _create_grid(self, grid_size=31):
        """Create interpolation grid using analytical solution"""
        print("Creating magnetic field interpolation grid...")
        
        grid_points = np.linspace(-self.Lw/2, self.Lw/2, grid_size)
        self.grid_points = grid_points
        
        # Initialize grid
        self.field_grid = np.zeros((grid_size, grid_size, grid_size, 3))
        
        # Calculate field at each grid point
        total = grid_size**3
        count = 0
        
        for i, x in enumerate(grid_points):
            for j, y in enumerate(grid_points):
                for k, z in enumerate(grid_points):
                    count += 1
                    if count % 1000 == 0:
                        print(f"  {count}/{total} points computed")
                    
                    P = np.array([x, y, z])
                    B = np.zeros(3)
                    
                    # Sum contributions from all coils
                    for coil_idx in range(6):
                        B += self._coil_field_analytical(P, coil_idx, 1.0)
                    
                    # Multiply by layers
                    B *= self.nl
                    self.field_grid[i, j, k, :] = B
        
        print(f"Grid creation complete! {total} points computed")
        
        # Create interpolators
        self.Bx_interp = RegularGridInterpolator(
            (grid_points, grid_points, grid_points),
            self.field_grid[:, :, :, 0],
            bounds_error=False,
            fill_value=None
        )
        self.By_interp = RegularGridInterpolator(
            (grid_points, grid_points, grid_points),
            self.field_grid[:, :, :, 1],
            bounds_error=False,
            fill_value=None
        )
        self.Bz_interp = RegularGridInterpolator(
            (grid_points, grid_points, grid_points),
            self.field_grid[:, :, :, 2],
            bounds_error=False,
            fill_value=None
        )
    
    def _save_grid(self):
        """Save interpolation grid to file"""
        with open(self.grid_file, 'wb') as f:
            pickle.dump({
                'grid_points': self.grid_points,
                'field_grid': self.field_grid
            }, f)
        print(f"Grid saved to {self.grid_file}")
    
    def _load_grid(self):
        """Load interpolation grid from file"""
        with open(self.grid_file, 'rb') as f:
            data = pickle.load(f)
        
        self.grid_points = data['grid_points']
        self.field_grid = data['field_grid']
        
        # Recreate interpolators
        self.Bx_interp = RegularGridInterpolator(
            (self.grid_points, self.grid_points, self.grid_points),
            self.field_grid[:, :, :, 0],
            bounds_error=False,
            fill_value=None
        )
        self.By_interp = RegularGridInterpolator(
            (self.grid_points, self.grid_points, self.grid_points),
            self.field_grid[:, :, :, 1],
            bounds_error=False,
            fill_value=None
        )
        self.Bz_interp = RegularGridInterpolator(
            (self.grid_points, self.grid_points, self.grid_points),
            self.field_grid[:, :, :, 2],
            bounds_error=False,
            fill_value=None
        )
        print(f"Grid loaded from {self.grid_file}")
    
    def get_field_fast(self, P, I=None):
        """
        Fast field calculation using interpolation
        """
        if I is not None:
            self.I = I
        
        # Ensure P is within bounds
        P_clipped = np.clip(P, -self.Lw/2, self.Lw/2)
        
        # Interpolate base field (for 1A per coil)
        B_base = np.array([
            self.Bx_interp(P_clipped)[0] if hasattr(self.Bx_interp(P_clipped), '__len__') else self.Bx_interp(P_clipped),
            self.By_interp(P_clipped)[0] if hasattr(self.By_interp(P_clipped), '__len__') else self.By_interp(P_clipped),
            self.Bz_interp(P_clipped)[0] if hasattr(self.Bz_interp(P_clipped), '__len__') else self.Bz_interp(P_clipped)
        ])
        
        # Scale by average current (assuming linearity)
        # For more accuracy, would need separate grids for each coil
        avg_I = np.mean(self.I)
        return B_base * avg_I
    
    def get_field_accurate(self, P, I=None):
        """
        Accurate field calculation using analytical solution
        Use for validation and offline calculations
        """
        if I is not None:
            self.I = I
        
        B = np.zeros(3)
        for coil_idx in range(6):
            B += self._coil_field_analytical(P, coil_idx, self.I[coil_idx])
        
        return B * self.nl
    
    def IntegrationOfBiotSavar(self):
        """
        Main interface - returns B and J
        """
        if self.use_interpolation:
            B = self.get_field_fast(self.P, self.I)
        else:
            B = self.get_field_accurate(self.P, self.I)
        
        # Calculate gradient using finite differences
        J = self._compute_gradient_fast()
        
        return B, J
    
    def _compute_gradient_fast(self):
        """Compute gradient using finite differences (fast)"""
        h = 0.1e-3  # 0.1mm step
        P0 = self.P
        
        # Calculate field at 6 neighboring points
        B0 = self.get_field_fast(P0) if self.use_interpolation else self.get_field_accurate(P0)
        
        Bx_plus = self.get_field_fast(P0 + np.array([h, 0, 0])) if self.use_interpolation else self.get_field_accurate(P0 + np.array([h, 0, 0]))
        Bx_minus = self.get_field_fast(P0 - np.array([h, 0, 0])) if self.use_interpolation else self.get_field_accurate(P0 - np.array([h, 0, 0]))
        By_plus = self.get_field_fast(P0 + np.array([0, h, 0])) if self.use_interpolation else self.get_field_accurate(P0 + np.array([0, h, 0]))
        By_minus = self.get_field_fast(P0 - np.array([0, h, 0])) if self.use_interpolation else self.get_field_accurate(P0 - np.array([0, h, 0]))
        Bz_plus = self.get_field_fast(P0 + np.array([0, 0, h])) if self.use_interpolation else self.get_field_accurate(P0 + np.array([0, 0, h]))
        Bz_minus = self.get_field_fast(P0 - np.array([0, 0, h])) if self.use_interpolation else self.get_field_accurate(P0 - np.array([0, 0, h]))
        
        # Build Jacobian
        J = np.zeros((3, 3))
        J[0, :] = (Bx_plus - Bx_minus) / (2 * h)
        J[1, :] = (By_plus - By_minus) / (2 * h)
        J[2, :] = (Bz_plus - Bz_minus) / (2 * h)
        
        return J
