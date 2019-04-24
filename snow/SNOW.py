r'''
SNOW: Sub-Network of an Oversegmented Watershed
Copyright (C) 2017 Jeff Gostick

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries


def trim_nearby_peaks(peaks, dt):
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    peaks, N = spim.label(peaks, structure=cube(3))
    crds = spim.measurements.center_of_mass(peaks, labels=peaks,
                                            index=sp.arange(1, N+1))
    crds = sp.vstack(crds).astype(int)  # Convert to numpy array of ints
    # Get distance between each peak as a distance map
    tree = sptl.cKDTree(data=crds)
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    dist_to_solid = dt[list(crds.T)]  # Get distance to solid for each peak
    hits = sp.where(dist_to_neighbor < dist_to_solid)[0]
    # Drop peak that is closer to the solid than its neighbor
    drop_peaks = []
    for peak in hits:
        if dist_to_solid[peak] < dist_to_solid[nearest_neighbor[peak]]:
            drop_peaks.append(peak)
        else:
            drop_peaks.append(nearest_neighbor[peak])
    drop_peaks = sp.unique(drop_peaks)
    # Remove peaks from image
    slices = spim.find_objects(input=peaks)
    for s in drop_peaks:
        peaks[slices[s]] = 0
    return (peaks > 0)


def extend_slice(s, shape, pad=1):
    a = []
    for i, dim in zip(s, shape):
        start = 0
        stop = dim
        if i.start - pad >= 0:
            start = i.start - pad
        if i.stop + pad < dim:
            stop = i.stop + pad
        a.append(slice(start, stop, None))
    return a


def trim_saddle_points(peaks, dt, max_iters=10):
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    labels, N = spim.label(peaks)
    slices = spim.find_objects(labels)
    for i in range(N):
        s = extend_slice(s=slices[i], shape=peaks.shape, pad=10)
        peaks_i = labels[s] == i + 1
        dt_i = dt[s]
        im_i = dt_i > 0
        iters = 0
        peaks_dil = sp.copy(peaks_i)

        while iters < max_iters:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_dil,
                                             structure=cube(3))
            peaks_max = peaks_dil*sp.amax(dt_i*peaks_dil)
            peaks_extended = (peaks_max == dt_i)*im_i
            if sp.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif sp.sum(peaks_extended*peaks_i) == 0:
                peaks_i = False
                break  # Found a saddle point
            peaks[s] = peaks_i
    return peaks


def create_image(shape, porosity):
    im = sp.ones(shape=shape, dtype=bool)
    while im.sum()/im.size > porosity:
        temp = sp.rand(*shape) < 0.9999
        temp = spim.distance_transform_edt(input=temp) > radii
        im *= temp
    return im

if __name__ == '__main__':
    # The code in this file works on 2D & 3D images, but the plotting functions
    # will fail since they are for 2D images only
    sp.random.seed(1)
    sigma = 0.35
    r_max = 5
    shape = [400, 400]
    porosity = 0.8
    radii = 10
    # Create image of overlapping disks
    im = create_image(shape=shape, porosity=porosity)
    # Apply steps of the SNOW algorithm to partition the pore space
    dt = spim.distance_transform_edt(input=im)
    dt = spim.gaussian_filter(input=dt, sigma=sigma)
    peaks = peak_local_max(image=dt, min_distance=r_max-1, exclude_border=0,
                           indices=False)
    Peaks = trim_saddle_points(peaks=peaks, dt=dt)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    regions = watershed(image=-dt, markers=spim.label(peaks)[0])
    edges = find_boundaries(regions)
    plt.subplot(1, 2, 1)
    plt.imshow(peaks/im)
    plt.subplot(1, 2, 2)
    plt.imshow(regions/im*(~edges), cmap=plt.cm.spectral)
