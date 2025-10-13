# -*- coding: utf-8 -*-
"""Magnetic Field and its Gradient Coding"""
import numpy as np
import numpy.matlib 

class MagneticFieldSim():

    def __init__(self, P, I):
        self.P = P  # microbot position
        self.nl = 5  # number of coil layers
        self.ntpl = 500  # number of turns per layer
        self.Lc = 20e-3  # length of any coil
        self.p = self.Lc / (2 * self.ntpl * np.pi)
        self.nd = 1000
        self.Rc = 30e-3  # radius of any coil
        self.I = I

        ## Workspace Specifications
        self.Lw = 2 * self.Rc  # length, width and height of the workspace
        self.CoilPositions = np.array([
            [self.Lw/2, 0, 0], 
            [0, self.Lw/2, 0], 
            [0, 0, self.Lw/2],
            [-self.Lw/2, 0, 0], 
            [0, -self.Lw/2, 0], 
            [0, 0, -self.Lw/2]
        ])  # coils positions
        
        C = self.CoilPositions
        n = 10000
        betah = np.linspace(0, 2 * self.ntpl * np.pi, n)
        self.xx = np.ndarray((6, n))
        self.yy = np.ndarray((6, n))
        self.zz = np.ndarray((6, n))
        
        for j in range(6):  # the number of coils is equal to 6
            if j == 0 or j == 3:
                zz = np.full((1, n), C[j, 2]) + self.Rc * np.cos(betah)
                yy = np.full((1, n), C[j, 1]) + self.Rc * np.sin(betah)
                xx = np.full((1, n), C[j, 0]) + np.sign(C[j, 0]) * 2 * self.p * betah
            elif j == 1 or j == 4:
                xx = np.full((1, n), C[j, 0]) + self.Rc * np.cos(betah)
                zz = np.full((1, n), C[j, 2]) + self.Rc * np.sin(betah)
                yy = np.full((1, n), C[j, 1]) + np.sign(C[j, 1]) * 2 * self.p * betah
            elif j == 2 or j == 5:
                xx = np.full((1, n), C[j, 0]) + self.Rc * np.cos(betah)
                yy = np.full((1, n), C[j, 1]) + self.Rc * np.sin(betah)
                zz = np.full((1, n), C[j, 2]) + np.sign(C[j, 2]) * 2 * self.p * betah
            
            self.xx[j, :] = xx
            self.yy[j, :] = yy
            self.zz[j, :] = zz

    def BiotSavar(self):
        beta = self.beta
        dbeta = self.dbeta
        Rc = self.Rc
        p = self.p
        C = self.Cj
        P = self.P
        I = self.Ij
        mu0 = 4 * np.pi * 1e-7
        
        xp = P[0]
        yp = P[1]
        zp = P[2]
        xc = C[0]
        yc = C[1]
        zc = C[2]
        
        x = xc + Rc * np.cos(beta)
        y = yc + Rc * np.sin(beta)
        z = zc + np.sign(zc) * 2 * p * beta
        
        dx = -Rc * np.sin(beta) * dbeta
        dy = Rc * np.cos(beta) * dbeta
        dz = np.sign(zc) * 2 * p * dbeta
        
        R = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
        if R == 0:
            R = 1e-9
        
        # Magnetic field components
        dBx = (mu0 * I) / (4 * np.pi) * ((z - zp) * dy - (y - yp) * dz) / R**3
        dBy = -(mu0 * I) / (4 * np.pi) * ((z - zp) * dx - (x - xp) * dz) / R**3
        dBz = (mu0 * I) / (4 * np.pi) * ((x - xp) * dy - (y - yp) * dx) / R**3
        
        # Magnetic field gradient (Jacobian)
        mu = 3 * mu0 * I / (4 * np.pi * R**5)
        dBxdx = mu * ((z - zp) * dy - (y - yp) * dz) * (x - xp)
        dBxdy = mu * ((z - zp) * (y - yp) * dy + ((R**2)/3 - (y - yp)**2) * dz)
        dBxdz = -mu * ((z - zp) * (y - yp) * dz + ((R**2)/3 - (z - zp)**2) * dy)
        
        dBydx = mu * ((z - zp) * (x - xp) * dx + ((R**2)/3 - (x - xp)**2) * dz)
        dBydy = mu * ((z - zp) * dx - (x - xp) * dz) * (y - yp)
        dBydz = -mu * ((x - xp) * (z - zp) * dz + ((R**2)/3 - (z - zp)**2) * dx)
        
        dBzdx = -mu * ((x - xp) * (y - yp) * dx + ((R**2)/3 - (x - xp)**2) * dy)
        dBzdy = mu * ((x - xp) * (y - yp) * dy + ((R**2)/3 - (y - yp)**2) * dx)
        dBzdz = mu * ((x - xp) * dy - (y - yp) * dx) * (z - zp)

        J = np.array([[dBxdx, dBxdy, dBxdz], 
                      [dBydx, dBydy, dBydz], 
                      [dBzdx, dBzdy, dBzdz]])  # magnetic field gradient
        
        return dBx, dBy, dBz, J

    def IntegrationOfBiotSavar(self):
        ntpl = self.ntpl
        nd = self.nd
        C = self.CoilPositions
        I = self.I
        nl = self.nl

        betah = np.linspace(0, 2 * ntpl * np.pi, nd)
        self.dbeta = 2 * ntpl * np.pi / nd
        
        Bx = 0
        By = 0
        Bz = 0
        J_total = np.zeros((3, 3))

        for i in range(nd):  # nd is the size of discretization
            self.beta = betah[i]
            dbx = 0
            dby = 0
            dbz = 0
            J0 = np.zeros((3, 3))
            
            for j in range(6):  # the number of coils is equal to 6
                Cj = C[j, :].copy()
                
                # Adjust coordinates based on coil orientation
                if j == 0 or j == 3:
                    Cj_temp = [C[j, 2], C[j, 1], C[j, 0]]
                elif j == 1 or j == 4:
                    Cj_temp = [C[j, 0], C[j, 2], C[j, 1]]
                elif j == 2 or j == 5:
                    Cj_temp = [C[j, 0], C[j, 1], C[j, 2]]
                
                self.Ij = I[j]
                self.Cj = Cj_temp
                [a, b, c, d] = self.BiotSavar()
                
                # Adjust outputs based on coil orientation
                if j == 0 or j == 3:
                    dbx0 = c
                    dby0 = a
                    dbz0 = b
                    J00 = np.array([[d[2, 0], d[2, 1], d[2, 2]],
                                   [d[1, 0], d[1, 1], d[1, 2]],
                                   [d[0, 0], d[0, 1], d[0, 2]]])
                elif j == 1 or j == 4:
                    dbx0 = a
                    dby0 = c
                    dbz0 = b
                    J00 = np.array([[d[0, 0], d[0, 2], d[0, 1]],
                                   [d[2, 0], d[2, 2], d[2, 1]],
                                   [d[1, 0], d[1, 2], d[1, 1]]])
                elif j == 2 or j == 5:
                    dbx0 = a
                    dby0 = b
                    dbz0 = c
                    J00 = d

                dbx += dbx0
                dby += dby0
                dbz += dbz0
                J0 += J00
            
            Bx += dbx
            By += dby
            Bz += dbz
            J_total += J0
        
        # Multiply by number of winding layers
        Bx = nl * Bx
        By = nl * By
        Bz = nl * Bz
        B = np.array([Bx, By, Bz])
        
        J_total = nl * J_total  # magnetic field gradient
        
        return B, J_total