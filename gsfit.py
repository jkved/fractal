import numpy as np
import matplotlib.pyplot as plt
import numba
import emcee

'''
The following functions:
    Calculate NR fractal by NR method (speed up with numba)
        Requires "roots" like 2d array
    Calculate fractal dimension
        Requires array of weighted values (0-1) of elements. Boundary selects boundary for selected points for fractal dimension calculations
    Loads colored image and turns it into grayscale
    Calculates Kolomogorov-Smirnov statistic (KS) for two arrays (images) - image and fractal
'''

@numba.jit 
def quotient(z, roots, dx=1e-8):

    def func(z, roots):
        prod = 1
        for i in range(roots.shape[0]):
            root_val = roots[i][0]+1j*roots[i][1]
            prod = prod * (z - root_val)
        return prod
    
    f = func(z, roots)
    fx = ( func(z+dx, roots) - func(z-dx, roots) ) / (2*dx)
    return f / fx


@numba.jit
def newton_method(z, roots, lim):
    t = 0
    dz = quotient(z, roots)
    while abs(dz) > lim and t < 200:
        t += 1
        dz = quotient(z, roots)
        z = z - dz
    return z, t
        
def grid_scaled(array, resX, resY):

    # re_min, re_max = np.min(array.real), np.max(array.real)
    # im_min, im_max = np.min(array.imag), np.max(array.imag)
    
    # scale = np.mean([(re_max - re_min)/4, (im_max - im_min)/4])
    
    re_min, re_max = -1, 1
    im_min, im_max = -1, 1
    
    scale = 0

    X, Y = np.meshgrid(np.linspace(re_min - scale, re_max + scale, resX), np.linspace(im_min - scale, im_max + scale, resY))
    
    Z = X+1j*Y
    
    return Z

def assign_root(Z, lim, p, resX, resY, array, roots):

    M = np.zeros((resX, resY), dtype=np.complex64)
    N = np.zeros((resX, resY))
    for i, row in enumerate(Z):
        for j, z in enumerate(row):
            z_temp, N[i, j] = newton_method(z, roots, lim)
            M[i,j] = 0 #array[np.where(np.isclose(array, z_temp, atol=1e-8))[0][0]]
            
    return M, N

def plotFractal(Z, m, cmap='viridis'):

    f, ax = plt.subplots(figsize=(8,6), dpi=60)
    
    ax.contourf(Z.real, Z.imag, m, cmap=cmap)
    # plt.colorbar()
 
def calculate_BoxDim(image, min_box_size, max_box_size, n_samples, n_offsets, plotBool, boundary=0.5):

    if max_box_size == None:
        max_box_size = np.log2(np.min(image.shape).astype(int)/2).astype(int)

    scales = np.logspace(max_box_size, min_box_size, n_samples, base=2)
    scales = np.unique(scales)

    locs = np.where(image >= boundary)
    voxels = np.array([(x, y) for x, y in zip(*locs)])

    Nmin = []

    for scale in scales:
        touched = []

        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)

        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in image.shape]
            bin_edges = [np.hstack([0-offset, x+offset]) for x in bin_edges]

            H1, e = np.histogramdd(voxels, bins = bin_edges)

            touched.append(np.sum(H1>0))

        Nmin.append(touched)

    Nmin = np.array(Nmin)
    Nmin = Nmin.min(axis=1)

    scales = np.array([np.min(scales[Nmin == x]) for x in np.unique(Nmin)])

    Nmin = np.unique(Nmin)
    Nmin = Nmin[Nmin > 0]
    scales = scales[:len(Nmin)]

    coeffs = np.polyfit(np.log(1/scales), np.log(Nmin), 1)

    if plotBool:
        fig, ax = plt.subplots(figsize = (8,6))
        ax.scatter(np.log(1/scales), np.log(np.unique(Nmin)), c = "teal", label = "Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1/scales))
        ax.plot(np.log(1/scales), fitted_y_vals, "k--", label = f"Fit: {np.round(coeffs[0],3)}X+{coeffs[1]}")
        ax.legend();
        plt.show


    return coeffs[0]

def image_processing(image_name):
    image_color = plt.imread(image_name)

    if (image_color.shape[2] < 3):
        assert 1==0, "No RGB"

    rgb = [0.2989, 0.5870, 0.1140]
    image_gray = 0
    for i in range(len(rgb)):
        image_gray += rgb[i]*image_color[:,:,i]

    return image_gray

def KS(image, fractal, plot=False):
    
    def extender(temp, tempf):
        if temp[0] > 0:
            temp = np.insert(temp, 0, np.linspace(0, temp[0], 100))
            tempf = np.insert(tempf, 0, np.zeros(100))
        if temp[-1] < 1:
            temp = np.append(temp, np.linspace(temp[-1], 1, 100))
            tempf = np.append(tempf, np.ones(100))
        return temp, tempf
    
    def get_closest(array, values):
        # make sure array is a numpy array
        array = np.array(array)

        # get insert positions
        idxs = np.searchsorted(array, values, side="left")

        # find indexes where previous index is closer
        prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
        idxs[prev_idx_is_less] -= 1

        return array[idxs], idxs

    x1 = np.sort(image.reshape(image.shape[0]*image.shape[1]))
    f1 = np.array(range(x1.shape[0]))/np.float64(x1.shape)
    x1, f1 = extender(x1, f1) 
    
    x2 = np.sort(fractal.reshape(fractal.shape[0]*fractal.shape[1]))
    f2 = np.array(range(x2.shape[0]))/np.float64(x2.shape)
    x2, f2 = extender(x2, f2)

    _, i = get_closest(x1, x2)
    loc = np.argmax(np.abs(f2-f1[i]))
    KS = np.max(np.abs(f2-f1[i]))
    
    if plot:
        plt.plot(x1, f1)
        plt.plot(x2, f2)
        plt.plot([x2[loc], x1[i[loc]]],[f2[loc], f1[i[loc]]], 'r-', linewidth=1, label='Largest diff')
        plt.legend()
        
    return KS

