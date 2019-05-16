import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
import numpy.ma as ma
from skimage.filters import threshold_otsu
import porespy as ps
from porespy.filters import find_peaks, trim_saddle_points, trim_nearby_peaks
import scipy.ndimage as spim
from skimage.morphology import watershed
from porespy.tools import randomize_colors
import scipy as sp
import openpnm as op


def crop_cone(mean_z_intensity,bot,top):
    cpt = importr('changepoint')

    cpm_result = cpt.multiple_mean_norm(FloatVector(mean_z_intensity),"BinSeg","MBIC",0,2,True,True,1)

    change_pts = cpt.cpts(cpm_result)

    bot = int(change_pts[0])+ bot
    top = int(change_pts[1])- top
    cut_warp = mean_z_intensity[bot:top]
    
    return (cut_warp,change_pts)

def filter_otsu(im,offset):
    wrap = dsa.WrapDataObject(im)
    dim = im.GetDimensions()
    np_ar = wrap.PointData['Tiff Scalars'].reshape(dim,order='F')
    
    masked_ar = ma.masked_where(np_ar == 0, np_ar)
    thresh = threshold_otsu(masked_ar.compressed())
    
    thres_fil = vtk.vtkImageThreshold()
    thres_fil.ThresholdByLower(thresh+thresh*offset)
    thres_fil.ReplaceInOn()
    thres_fil.SetInValue(0)
    thres_fil.SetOutValue(1)
    thres_fil.SetInputData(im)
    thres_fil.Update()
    
    n_wrap = dsa.WrapDataObject(thres_fil.GetOutput())
    n_np_ar = n_wrap.PointData['Tiff Scalars'].reshape(dim,order='F')
    
    n_masked = ma.masked_where(np_ar==0,n_np_ar)
    
    return n_masked

def particle_sizer(im,sigma):
    dt = spim.distance_transform_edt(input=im)
    dt = spim.gaussian_filter(input=dt, sigma=sigma)
    peaks = find_peaks(dt=dt)

    peaks = trim_saddle_points(peaks=peaks, dt=dt, max_iters=500)

    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)

    regions = watershed(image=-dt, markers=peaks, mask=dt > 0)
    regions = randomize_colors(regions)
    
    net = ps.networks.regions_to_network(im=regions*im, dt=dt, voxel_size=1)

    pn = op.network.GenericNetwork()
    pn.update(net)
    
    return pn
