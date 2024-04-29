import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

## Input
filename = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_L100_L100_1.txt"

output = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/error_phase_plot.pdf"

# Load data
data = np.loadtxt(filename)

x_dat = data[:,0]
y_dat = data[:,1]
z_dat = data[:,4]

# Convert from pandas dataframes to numpy arrays
X, Y, Z, = np.array([]), np.array([]), np.array([])
for i in range(len(x_dat)):
        X = np.append(X, x_dat[i])
        if y_dat[i] > 1:
            y_dat[i] = 1
        Y = np.append(Y, y_dat[i])
        Z = np.append(Z, z_dat[i])

# create x-y points to be used in heatmap
xi = np.linspace(X.min(), X.max(), 1000)
yi = np.linspace(Y.min(), Y.max(), 1000)

# Interpolate for plotting
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')

# I control the range of my colorbar by removing data 
# outside of my range of interest
ymin = 0.0
ymax = 1.0
yi[(yi<ymin) | (yi>ymax)] = None
zmin = np.abs(z_dat).min()
zmax = np.abs(z_dat).max()
zi[(zi<zmin) | (zi>zmax)] = None

# Create the contour plot
plt.axis([0.1, 0.5, 0.6, 1.0])
plt.xlabel(r"$\rho$", fontsize=24)
plt.ylabel(r"$v_+$", fontsize=24)
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                  vmax=zmax, vmin=zmin)
plt.colorbar() 
plt.tight_layout()

plt.savefig(output, format = "pdf", dpi = 300)
plt.show()