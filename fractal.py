import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.misc import derivative
from matplotlib.colors import ListedColormap

class NewtonRaphson:
    '''
    Class to produce and work with fractal created by Newton - Raphson method.
    
    Global variables:
        N - iteration number to check the convergence towards root of polynomial
        lim - appropriate limit to decide if point converges
        roots, coeffs - arrays for polynomial defining properties
        resolution - res x res is a number of points in resulting array of fractal
        Z - array representing a complex plane of fractal
    '''
    N = 1000
    lim = 1e-9
    roots = []
    coeffs = []
    resolution = 200
    Z = []
    #colors = ['b', 'r', 'g', 'y']
    
    def __init__(self, array, fromRoots=True):
        '''
        Initializes a polynomial to which Newton-Raphson method is applied. fromRoots boolean specifies if the input is an array of roots or polynomial coefficients.
        '''
        if fromRoots:
            self.p = np.polynomial.Polynomial.fromroots(array)
            self.roots = array
            #self.coeffs = np.poly1d
        else:
            self.p = np.poly1d(array)
            self.coeffs = array
            self.roots = np.roots(self.coeffs)
            
    def newton_method(self, z):
        '''
        Implements Newton-Raphson method for polynomial and selected point z (either real or complex) over N iterations. Returns the point (particularly, one of the roots) that z converges to.
        '''
        for i in range(self.N):
            dz = self.p(z) / scipy.misc.derivative(self.p, z, dx = 1e-8)

            if abs(dz) > self.lim:
                z -= dz
            elif dz <= self.lim:
                return z
            
    def grid_scaled(self):
        '''
        Should only be called in case user input does not specify the complex plane coordinates grid for calculation of fractal array
        '''
        
        re_min, re_max = np.min(self.roots.real), np.max(self.roots.real)
        im_min, im_max = np.min(self.roots.imag), np.max(self.roots.imag)

        scale = np.mean([(re_max - re_min)/2, (im_max - im_min)/2])

        X, Y = np.meshgrid(np.linspace(re_min - scale, re_max + scale, self.resolution), np.linspace(im_min - scale, im_max + scale, self.resolution))
        
        self.Z = X+1j*Y
        
        return X+1j*Y

    def assign_root(self, lim = 1e-8):
        '''
        Executes search for the root to which specific point converges to. Returns z dim*dim shape matrix with particular root value from self.roots array.
        '''
        if self.Z == []:
            self.grid_scaled()
        
        if np.iscomplex(self.Z[0][0]):
            m = np.zeros((self.Z.shape[0], self.Z.shape[1]), dtype=complex)
        else:
            m = np.zeros((self.Z.shape[0], self.Z.shape[1]))

        for i, row in enumerate(self.Z):
            for j, z in enumerate(row):
                a = self.roots[np.where(np.isclose(self.roots, self.newton_method(z), atol=1e-8))[0][0]]
                m[i,j] = a
                
        self.matrix = m
        return m
    
    def plotFractal(self):
        '''
        Plots the fractal view from self.matrix. If such value is not created, self.assign_root() is called to create the value.
        '''
        if not hasattr(self, 'matrix'):
            self.assign_root()
        
        f, ax = plt.subplots(figsize=(12,12), dpi=80)
        if np.iscomplex(self.matrix[0][0]):
            ax.contourf(self.Z.real, self.Z.imag, self.matrix)
        else:
            ax.plot(self.Z[0,:], self.p(self.Z[0,:]), 'r-')
            ax.scatter(self.Z[0,:], np.zeros(self.Z.shape[1]), c=self.matrix[0,:], marker='o')
            
class Mandelbrot:
    
    resolution = 1000
    
    def __init__(self, gridArr=[-2, 1, -1, 1], n=2):
        '''
        grid is an array of 4 elements, representing real and imag values of grid boundaries: x_min, x_max, y_min, y_max
        '''
        self.gridArr = gridArr
        self.n = n
    
    def eq_master(self, c, z=0+0j):
        '''
        Recursive equation for Mandelbrot (Multibrot) set
        '''
        return z**self.n + c
    
    def divergence(self, c, z=0+0j, N=20, lim=2):
        '''
        Checks whether the point diverges. The returned value i shows steps necessary to fall out of iteration radius.
        http://mrob.com/pub/muency/escaperadius.html
        https://math.stackexchange.com/questions/890190/mandelbrot-sets-and-radius-of-convergence
        '''
        for i in range(N):
            z = self.eq_master(c, z)
            if (np.abs(z.real + z.imag) >= lim):
                    return i
        return 0
    
    def make_grid(self):
        '''
        Creates a grid of complex values
        '''
        x_min, x_max, y_min, y_max = self.gridArr[0], self.gridArr[1], self.gridArr[2], self.gridArr[3]
        
        X, Y = np.meshgrid(np.linspace(x_min, x_max, self.resolution), np.linspace(y_min, y_max, self.resolution))
        
        c = X + 1j * Y
        
        return c
    
    def determine_point(self, N=20, lim=2):
        
        mandelbrot_set = np.zeros((self.resolution, self.resolution))
        
        c = self.make_grid()
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                mandelbrot_set[i, j] = self.divergence(c[i, j], z=0+0*1j, N=N, lim=lim)
        
        self.mandelbrot_set = mandelbrot_set
        
        return mandelbrot_set
    
    def plotFractal(self, N=20, lim=2):
        '''
        Plots the fractal view in self.matrix. If such value is not created, self.assign_root() is called to create the value.
        '''
        if not hasattr(self, 'mandelbrot_set'):
            self.determine_point(N, lim)
        
        f, ax = plt.subplots(figsize=(12,12), dpi=80)
        
        ax.contourf(self.mandelbrot_set, cmap='gnuplot2')
        