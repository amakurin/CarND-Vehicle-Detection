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

nsamples = 1000

cars = lu.load_dataset('cars.txt')
noncars = lu.load_dataset('notcars.txt')


params_ranges = {'use_spat'   :[True, False],
                 'spat_cspace':['RGB'], 
                 'spat_size'  :[24, 32],
                 
                 'use_hist'   :[True, False], 
                 'hist_cspace':['RGB', 'HSV', 'LUV', 'HLS', 'YCrCb'],
                 'hist_bins'  :[16, 32],
                 'hist_chan_range':['0:0', '1:1', '2:2'],

                 'use_hog'    :[True, False],
                 'hog_orient' :[8,9],
                 'hog_pix_per_cell':[6], 
                 'hog_cell_per_block':[2,3],
                 'hog_cspace' :['RGB', 'HSV', 'LUV', 'HLS', 'YCrCb'], 
                 'hog_chan_range':['0:0', '1:1', '2:2']
                 }

params = {'hist_cspace': 'RGB', 'hog_pix_per_cell': 6, 'hist_bins': 16, 
'hog_orient': 7, 'hog_cspace': 'RGB', 'use_hist': True, 
'spat_cspace': 'RGB', 'hog_cell_per_block': 2, 'hog_chan_range': '0:0', 
'use_spat': True, 'spat_size': 32, 'use_hog': True} 

#lu.brute_force_params(cars, noncars, params_ranges, nsamples, 'bfparams.p')

#lu.build_classifier(cars, noncars, params, result_file='clfv1.p')

tst_base_path = '.\\test_images\\'

tst_imgs = glob.glob(tst_base_path+'*.jpg')

tst_imgs = [lu.imread(imf) for imf in tst_imgs]

#320x1280
#small = 64x96
#big = 96x128


img = tst_imgs[2]

builtclf = lu.load('clfv1.p')

def search_cars(img, builtclf, win_specs):
    wins = []
    for spec in win_specs:
        spec_wins = lu.slide_window(img, y_start_stop=spec['y_start_stop'],
                                    xy_window=spec['xy_window'])        
        wins.extend(spec_wins)
    
    imgs = [cv2.resize(img[win[0][1]:win[1][1], win[0][0]:win[1][0]], (64,64)) for win in wins]
    
    prediction = lu.predict(imgs, builtclf['clf'], builtclf['scaler'], builtclf['params'])

    wins = np.array(wins)
    return wins[prediction > 0]

#small_wins = lu.slide_window(img, y_start_stop=[400,500], xy_window=(96,64))

#imgs = [cv2.resize(img[win[0][1]:win[1][1], win[0][0]:win[1][0]], (64,64)) for win in small_wins]
#imgs = np.array(imgs)
#prediction = np.array(lu.predict(imgs, builtclf['clf'], builtclf['scaler'], builtclf['params']))
def proc(frame):
    found_wins = search_cars(frame, builtclf, 
        [{'y_start_stop':[400,500], 'xy_window':(96,64)},
        {'y_start_stop':[400,680], 'xy_window':(128,96)}])
    drawn_found = lu.draw_boxes(frame, found_wins)            
    #lu.plot_img_grid([drawn_found])
    return drawn_found

lu.process_video('project_video.mp4', proc, 'prout1.mp4')

#print ('small cnt:',len(small_wins))
#drawn_small = lu.draw_boxes(img, small_wins)
#big_wins = lu.slide_window(img, y_start_stop=[400,680], xy_window=(128,96))
#print ('big cnt:',len(big_wins))
#print ('tot cnt:',len(big_wins)+len(small_wins))
#drawn_big = lu.draw_boxes(img, big_wins)

#
#lu.plot_img_grid([drawn_small,drawn_big])