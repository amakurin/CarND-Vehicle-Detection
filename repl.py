import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cProfile
import labutils as lu
import time
import collections as colls

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

paramsv2 = {'hist_cspace': 'RGB', 'use_hist': True, 
'hog_pix_per_cell': 6, 'hog_cspace': 'HLS', 
'hist_bins': 32, 'hog_orient': 9, 
'spat_cspace': 'RGB', 'hog_chan_range': '1:2', 
'hist_chan_range': '0:2', 'use_spat': True, 
'use_hog': True, 'hog_cell_per_block': 2, 'spat_size': 32}

paramsv3 = {'hist_cspace': 'RGB', 'use_hist': True, 
'hog_pix_per_cell': 6, 'hog_cspace': 'HLS', 
'hist_bins': 32, 'hog_orient': 9, 
'spat_cspace': 'RGB', 'hog_chan_range': '1:2', 
'hist_chan_range': '0:2', 'use_spat': True, 
'use_hog': True, 'hog_cell_per_block': 2, 'spat_size': 32,}

paramsv4 = {'hist_cspace': 'RGB', 'use_hist': True, 
'hog_pix_per_cell': 6, 'hog_cspace': 'YCrCb', 
'hist_bins': 32, 'hog_orient': 9, 
'spat_cspace': 'RGB', 'hog_chan_range': '0:2', 
'hist_chan_range': '0:2', 'use_spat': True, 
'use_hog': True, 'hog_cell_per_block': 2, 'spat_size': 32}
#lu.brute_force_params(cars, noncars, params_ranges, nsamples, 'bfparams.p')

#lu.build_classifier(cars, noncars, params, result_file='clfv1.p')


class Pipeline():
    def init(self):
        self.heatmaps = colls.deque(maxlen=self.n_heat_aggregate)

    def __init__(self, builtclf, win_specs, 
                n_heat_aggregate=3, 
                heat_lo_thresh=2,
                heat_lo_thresh_per_frame=2,
                heat_max_lo_thresh=8,
                dec_fn = False,
                dec_thre = 0,
                precalc_hog = False):
        self.builtclf = builtclf
        self.win_specs = win_specs
        self.n_heat_aggregate = n_heat_aggregate
        self.heat_lo_thresh = heat_lo_thresh
        self.heat_lo_thresh_per_frame = heat_lo_thresh_per_frame
        self.heat_max_lo_thresh = heat_max_lo_thresh
        self.dec_fn = dec_fn
        self.dec_thre = dec_thre
        self.precalc_hog = precalc_hog  
        self.init()

    def sum_heat_map(self, heatmap = None):
        heatmaps  = list(self.heatmaps)
        if heatmap is not None:
            heatmaps.append(heatmap)
        return np.sum(heatmaps, axis=0)

    def side_stack_imgs(self, imgs, side_y_ratio=0.25):
        '''
        Composes imgs to main main area (first img) and side bar (all others)
        '''
        result = imgs[0]
        total = len(imgs)
        if total>1:
            height,width = imgs[0].shape[:2]
            max_side_num = round((1-side_y_ratio)/side_y_ratio)
            if total < max_side_num+1:
                zeros = np.zeros_like(imgs[0])
                imgs.extend([zeros]*(max_side_num+1-total))
            elif total > max_side_num+1:
                imgs = imgs[:max_side_num+1]
            for i in range(1, max_side_num+1):
                imgs[i] = cv2.resize(imgs[i], (0,0), fx=side_y_ratio, fy=side_y_ratio)
                imgs[i][0,:,:] = 100
                imgs[i][-1,:,:] = 100
            side = np.vstack(imgs[1:])
            main_ratio = 1 - side_y_ratio
            main = cv2.resize(imgs[0], (0,0), fx=main_ratio, fy=main_ratio)
            result = np.hstack((main,side))
        return result

    def process_frame(self, frame):
        zero_heat = np.zeros(frame.shape[:2])
        found_wins = lu.search_cars(frame, self.builtclf, self.win_specs, 
                        precalc_hog=self.precalc_hog,
                        dec_fn=self.dec_fn,
                        dec_thre=self.dec_thre)
        drawn_found = lu.draw_boxes(frame, found_wins)            
        heatmap = lu.add_heat(zero_heat, found_wins)
        totalheat = self.sum_heat_map(heatmap)

        agg_thre_ratio = (len(self.heatmaps) + 1)/(self.n_heat_aggregate + 1)
        
        labels = lu.label_heatmap(totalheat, self.heat_lo_thresh * agg_thre_ratio)
        bboxes = lu.labels_bboxes(labels)
        filtered = lu.filter_outlier_boxes(bboxes, totalheat, self.heat_max_lo_thresh * agg_thre_ratio, (48,48))
        
        toaggregate = lu.chan_threshold(heatmap, self.heat_lo_thresh_per_frame)
        #toaggregate = lu.add_heat(toaggregate, filtered)
        self.heatmaps.append(toaggregate)

        drawn_bboxes = lu.draw_boxes(frame, filtered)            
        #if len(found_wins)>0: 
        #    lu.plot_img_grid([drawn_found,heatmap, drawn_bboxes,totalheat], 2,2)
        return self.side_stack_imgs([drawn_bboxes, drawn_found, 
                        lu.scale(np.dstack((heatmap,heatmap,heatmap))), 
                        lu.scale(np.dstack((totalheat,totalheat,totalheat)))])

    def process_video(self, src_path, tgt_path):
        self.init()
        lu.process_video(src_path, self.process_frame, tgt_path)
        pass

builtclf = lu.load('clfv4hnm1.p')

pipeline = Pipeline(builtclf, 
    [{'y_start_stop':[400,470], 'xy_window':[80,48], 'xy_overlap':[0.7,0.7]},
     {'y_start_stop':[400,482], 'xy_window':[80,64], 'xy_overlap':[0.8,0.7]},
     {'y_start_stop':[410,526], 'xy_window':[132,96], 'xy_overlap':[0.8,0.5]},
     {'y_start_stop':[420,680], 'xy_window':(128,128), 'xy_overlap':[0.5,0.5]}
     ],
    #[{'y_start_stop':[400,496], 'xy_window':[80,64], 'xy_overlap':[0.75,0.75]},
    # {'y_start_stop':[386,530], 'xy_window':[128,96], 'xy_overlap':[0.75,0.75]},
    # {'y_start_stop':[400,680], 'xy_window':[244,160], 'xy_overlap':[0.75,0.75]},
    # ],
     n_heat_aggregate=3,
     heat_lo_thresh=1,
     heat_lo_thresh_per_frame=1,
     heat_max_lo_thresh=2,
     precalc_hog=True,
     dec_fn = False,
     dec_thre = 0)


#tst_base_path = './test_images/'
tst_base_path = './project_video/'

tst_imgs = glob.glob(tst_base_path+'*.jpg')

tst_imgs = [lu.imread(imf) for imf in tst_imgs]


#img = lu.imread('.\\dataset\\non-vehicles\\Extras\\extra831.png')
#lu.plot_img_grid([img])
#res = lu.predict([img], builtclf['clf'], builtclf['scaler'], builtclf['params'],decision_result=True)
#print (res)
#print (pipeline.win_specs_max_bounds((720,1280)))
pipeline.process_video('project_video.mp4', 'prout4hnm1.mp4')
#pipeline.process_video('test_video.mp4', 'testout4hnm1.mp4')

#img = lu.imread('.\\dataset\\vehicles\\GTI_MiddleClose\\image0057.png')
#
#f, imgh =lu.hog_chan_feats(img[:,:,1], 9, 6, 2, vis=True)
#lu.plot_img_grid([img,imgh])

#for img in tst_imgs[100:]:
#    pipeline.init()
#    pipeline.process_frame(img)

#pipeline.process_frame(lu.imread('.\\test_video\\frame-16.jpg'))
#pipeline.process_frame(lu.imread('.\\project_video\\frame-1034.jpg'))
#
#cProfile.run('pipeline.process_video("test_video.mp4", "testout2.mp4")')
#cProfile.run('pipeline.process_frame(tst_imgs[6])')

#img = tst_imgs[6]
#hls = lu.cvt_color(img, 'HLS')
#
#t=time.time()
#hf = lu.hog_chan_feats(hls[400:680,:,1], 9, 6, 2, vis=False, feature_vec=False)
#t2=time.time()
#print ('pre:', t2-t)
#print ('hf.shape:',hf.shape)
#
#wins = lu.slide_window(img.shape, 
#            x_start_stop=[None, None], y_start_stop=[400, 680], 
#                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#
#print ('wins cnt:', len(wins))
#
#t=time.time()
#for win in wins:
#    srch_img = hls[win[0][1]:win[1][1], win[0][0]:win[1][0],1]    
#    hf = lu.hog_chan_feats(srch_img, 9, 6, 2, vis=False, feature_vec=False)
#t2=time.time()
#print ('direct:', t2-t)
