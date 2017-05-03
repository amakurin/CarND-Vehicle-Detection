import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cProfile
import labutils as lu
import time
from sklearn.preprocessing import StandardScaler

#lu.prepare_dataset('.\\dataset\\vehicles\\','cars.txt')
#lu.prepare_dataset('.\\dataset\\non-vehicles\\','notcars.txt')

tst_base_path = '.\\test_images\\'

tst_imgs = glob.glob(tst_base_path+'*.jpg')

tst_imgs = [lu.imread(imf) for imf in tst_imgs]

#320x1280
#small = 64x96
#big = 96x128

#img = tst_imgs[0]



nsamples = 1000

cars = lu.load_dataset('cars.txt')
noncars = lu.load_dataset('notcars.txt')


params_ranges = {'use_spat'   :[True, False],
                 'spat_cspace':['RGB'], 
                 'spat_size'  :[16, 32],
                 
                 'use_hist'   :[True, False], 
                 'hist_cspace':['RGB'],
                 'hist_bins'  :[16, 32],
                 
                 'use_hog'    :[True, False],
                 'hog_orient' :[7,8,9],
                 'hog_pix_per_cell':[6,8], 
                 'hog_cell_per_block':[2,3],
                 'hog_cspace' :['RGB', 'HSV', 'LUV', 'HLS', 'YCrCb'], 
                 'hog_chan_range':['0:0', '1:1', '2:2', '0:2']
                 }

lu.brute_force_params(cars, noncars, params_ranges, nsamples, 'bfparams.p')
