import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

## Input
mac_filename1 = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_L100_L100_correct1.txt"
mac_filename2 = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_L100_L100_correct2.txt"
mac_filename3 = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_L100_L100_correct3.txt"
mac_filename_low_den_1 = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_low_densities_L100_L100_1.txt"
mac_filename_low_den_2 = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/results_low_densities_L100_L100_2.txt"
mac_output = "/Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/cython_version/phase_plot_joined.pdf"

windows_filename = "./results_L100_L100.txt"
windows_output = "./phase_plot.pdf"

# Load data
#data = np.loadtxt(mac_filename)
data1 = np.loadtxt(mac_filename1)
data2 = np.loadtxt(mac_filename2)
data3 = np.loadtxt(mac_filename3)
data_low_den_1 = np.loadtxt(mac_filename_low_den_1)
data_low_den_2 = np.loadtxt(mac_filename_low_den_2)
data = np.concatenate((data1, data2, data3, data_low_den_1, data_low_den_2), axis=0)

x_dat = data[:,0]
y_dat = data[:,1]
z_dat = data[:,3]

# Convert from pandas dataframes to numpy arrays
X, Y, Z, = np.array([]), np.array([]), np.array([])
for i in range(len(x_dat)):
        X = np.append(X, x_dat[i])
        if y_dat[i] > 1:
            y_dat[i] = 1
        Y = np.append(Y, y_dat[i])
        Z = np.append(Z, z_dat[i])

# create x-y points to be used in heatmap
xi = np.linspace(X.min(), X.max(), 400)
yi = np.linspace(Y.min(), Y.max(), 400)

# Interpolate for plotting
zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')

# I control the range of my colorbar by removing data 
# outside of my range of interest
ymin = 0.0
ymax = 1.0
yi[(yi<ymin) | (yi>ymax)] = None
zmin = np.abs(z_dat).min()
zmax = np.abs(z_dat).max()
zi[(zi<zmin) | (zi>zmax)] = None

# Create the contour plot
plt.axis([0.05, 0.5, 0.6, 1.0])
plt.xlabel(r"$\rho$", fontsize=24)
plt.ylabel(r"$v_+$", fontsize=24)
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                  vmax=zmax, vmin=zmin)

from matplotlib import ticker

# (generate plot here)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=15)
cb.locator = tick_locator
cb.update_ticks()
#plt.colorbar() 

plt.tight_layout()

#plt.savefig(mac_output, format = "pdf", dpi = 300)
plt.savefig(windows_output, format = "pdf", dpi = 300)
plt.show()