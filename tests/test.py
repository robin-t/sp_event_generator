import numpy as np
from density.density import DensitySet

# Load any M density, e.g. rsi_velocity
d = DensitySet.load("path/to/rsi_velocity_density_folder")

# Check mean of M across all x-bins
# M_bins are the bin edges, M_prob[i,j] = P(M in bin j | x in bin i)
mean_M_per_xbin = []
for i in range(d.M_prob.shape[0]):
    M_mids = (d.M_bins[:-1] + d.M_bins[1:]) / 2
    mean_M_per_xbin.append(np.sum(d.M_prob[i] * M_mids))

print("Mean M per x-bin:", np.array(mean_M_per_xbin))
print("Overall mean M:", np.mean(mean_M_per_xbin))
