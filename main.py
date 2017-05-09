import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cProfile
import labutils as lu
import time
import collections as colls

class Pipeline():
    def init(self):
        '''
        Inits heatmaps as empty deque
        '''
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
        '''
        Calculates aggregated heatmap as sum of stored and (if) provided heatmap
        '''
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
        '''
        Frame processing pipeline
        '''
        # Search for wins with nonzeroprediction
        found_wins = lu.search_cars(frame, self.builtclf, self.win_specs, 
                        precalc_hog=self.precalc_hog,
                        dec_fn=self.dec_fn,
                        dec_thre=self.dec_thre)
        # Draw found wins for side view
        drawn_found = lu.draw_boxes(frame, found_wins)            
        # Calculate frame heatmap
        zero_heat = np.zeros(frame.shape[:2])
        heatmap = lu.add_heat(zero_heat, found_wins)
        # Calculate aggregated heat
        totalheat = self.sum_heat_map(heatmap)
        # Calculate current threshold for aggregated heat
        agg_thre_ratio = (len(self.heatmaps) + 1)/(self.n_heat_aggregate + 1)
        # Threshold aggregated heatmap and label regions
        labels = lu.label_heatmap(totalheat, self.heat_lo_thresh * agg_thre_ratio)
        # Resolve labels to bounding boxes
        bboxes = lu.labels_bboxes(labels)
        # Filter boxes by Max heat in a box and boxsize
        filtered = lu.filter_outlier_boxes(bboxes, totalheat, self.heat_max_lo_thresh * agg_thre_ratio, (48,48))
        # Threshold frame heat for aggregation
        toaggregate = lu.chan_threshold(heatmap, self.heat_lo_thresh_per_frame)
        # Append thresholded frame heatmap to internal collection
        self.heatmaps.append(toaggregate)
        # Draw filteres bounding boxes on main output
        drawn_bboxes = lu.draw_boxes(frame, filtered)            
        # Compose result view and return
        return self.side_stack_imgs([drawn_bboxes, drawn_found, 
                        lu.scale(np.dstack((heatmap,heatmap,heatmap))), 
                        lu.scale(np.dstack((totalheat,totalheat,totalheat)))])

    def process_video(self, src_path, tgt_path):
        '''
        Processes video file
        '''
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
     n_heat_aggregate=3,
     heat_lo_thresh=1,
     heat_lo_thresh_per_frame=1,
     heat_max_lo_thresh=2,
     precalc_hog=True,
     dec_fn = False,
     dec_thre = 0)

pipeline.process_video('project_video.mp4', 'prout4hnm10.mp4')
