import pickle
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import glob
import time
import os 
import os.path
import itertools
from moviepy.editor import VideoFileClip, ImageSequenceClip
from skimage.feature import hog as skhog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

def cvt_color(img, cspace='RGB'):
    '''
    Converts img to different color space
    img - RGB image
    cspace - one of 'RGB','LUV','YUV','HSV', 'HLS','YCrCb','gray'
    '''
    cspacemap = {'RGB':None, 'HSV':cv2.COLOR_RGB2HSV,
                 'LUV':cv2.COLOR_RGB2LUV, 'HLS':cv2.COLOR_RGB2HLS,
                 'YUV':cv2.COLOR_RGB2YUV,'YCrCb':cv2.COLOR_RGB2YCrCb,
                 'gray': cv2.COLOR_RGB2GRAY}
    converter = cspacemap[cspace]
    if converter is None:
        result = np.copy(img) 
    else:
        result = cv2.cvtColor(img, converter)
    return result

def scale(img, max=255):
    '''
    Scales img to max
    '''
    return np.uint8(img*max/np.max(img))

def spatial_feats(img, cspace='RGB', size=32):
    '''
    Computes spatial features of RGB img 
    cspace - color space for pre convertion
    size - size of template
    '''
    feature_image = cvt_color(img, cspace)
    features = cv2.resize(feature_image, (size, size)).ravel() 
    return features    

def hist_feats(img, cspace = 'RGB', chan_range=None, bins=32, hrange=(0, 256)):
    '''
    Computes histogram features of an RGB img
    cspace - color space for preconvertion
    chan_range - string range of converted image's channels to use for feature extraction 'start_chan:end_chan' 
    bins - number of histogram bins to compute on each channel
    hrange - intensity range to use for histogram calculation
    '''
    feature_image = cvt_color(img, cspace)
    hists = []
    if len(feature_image.shape)==2:
        hists= [np.histogram(feature_image, bins=bins, range=hrange)]
    else:
        if chan_range is None:
            chan_range = range(feature_image.shape[2])
        else:
            bounds = chan_range.split(':')
            chan_range = range(int(bounds[0]), int(bounds[1])+1)    
        for chan in chan_range:
            hist = np.histogram(img[:,:,chan], bins=bins, range=hrange)
            hists.append(hist)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate([hist[0] for hist in hists])
    return hist_features, hists

def hog_chan_feats(chan_img, orient, pix_per_cell, cell_per_block, 
              vis=False, feature_vec=True):
    '''
    Calculates HOG features of single channeled chan_img
    orient - number of orintation histogram bins
    pix_per_cell - number of pixels per cell
    cell_per_block - number of cells per block
    vis - if True, returns HOG image
    feature_vec - if True, flattens calculated HOG to single dimensional vector
    '''
    if vis == True:
        features, hog_image = skhog(chan_img, orientations=orient, 
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), 
                                transform_sqrt=False, 
                                visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = skhog(chan_img, orientations=orient, 
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), 
                    transform_sqrt=False, 
                    visualise=False, feature_vector=feature_vec)
        return features    

def hog_feats(img, orient, pix_per_cell, cell_per_block,
              cspace='RGB', chan_range=None):
    '''
    Calculates vector of HOG features for RGB img
    cspace - color space for preconvertion
    chan_range - string range of converted image's channels to use for feature extraction 'start_chan:end_chan' 
    orient, pix_per_cell, cell_per_block - same as for hog_chan_feats
    '''
    feature_image = cvt_color(img, cspace)
    hog_features = []
    if len(feature_image.shape)==2:
        hog_features = hog_chan_feats(feature_image, orient, 
                                pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True)
    else:
        if chan_range is None:
            chan_range = range(feature_image.shape[2])
        else:
            bounds = chan_range.split(':')
            chan_range = range(int(bounds[0]), int(bounds[1])+1)    
        for chan in chan_range:
            chan_feats = hog_chan_feats(feature_image[:,:,chan], orient, 
                                pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True)
            hog_features.append(chan_feats)
    hog_features = np.ravel(hog_features)
    return hog_features

def hog_precalculate(img, params):
    '''
    Precalculats HOG for RGB img with respect to
    params - map of all params for hog_feats except img, param keys starts with 'hog_' prefix: eg hog_orient
    '''
    orient = params['hog_orient']
    pix_per_cell = params['hog_pix_per_cell'] 
    cell_per_block= params['hog_cell_per_block']
    cspace = params['hog_cspace']
    chan_range = params['hog_chan_range']
    x_start, x_stop = params['x_bounds']
    y_start, y_stop = params['y_bounds']

    feature_image = cvt_color(img[y_start:y_stop, x_start:x_stop, :], cspace)
    hog = []
    if len(feature_image.shape)==2:
        hog = [hog_chan_feats(feature_image, orient, 
                                pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=False)]
    else:
        if chan_range is None:
            chan_range = range(feature_image.shape[2])
        else:
            bounds = chan_range.split(':')
            chan_range = range(int(bounds[0]), int(bounds[1])+1)    
        for chan in chan_range:
            chan_hog = hog_chan_feats(feature_image[:,:,chan], orient, 
                                pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=False)
            hog.append(chan_hog)
    return hog

def hog_feats_precalc(params):
    '''
    Calculates HOG feature vector of params['win'], with respect to params starting with hog_ prefix, using params['hog_precalc']
    '''
    # Extract all params needed
    win = params['win']
    pix_per_cell = params['hog_pix_per_cell']
    precalc = params['hog_precalc']
    # Extract bounds used to precalculation 
    x_start, x_stop = params['x_bounds']
    y_start, y_stop = params['y_bounds']
    base_win = np.array(params['base_win'])//pix_per_cell-1
    # Calculate win coords with relative to bounds
    wxs = win[0][0]-x_start
    wys = win[0][1]-y_start
    # Calculte win coords in hog_cell units
    wxs_cells = wxs // pix_per_cell 
    wxe_cells = wxs_cells+base_win[0] #wxe // pix_per_cell 
    wys_cells = wys // pix_per_cell 
    wye_cells = wys_cells+base_win[1] #wye // pix_per_cell 
    # Extract HOG values of win for each channel image
    hog_features = []
    for chan in precalc:
        chan_feats = chan[wys_cells:wye_cells, wxs_cells:wxe_cells,:,:,:]
        hog_features.append(chan_feats)
    # Flatten features and return
    return np.ravel(hog_features)

def get_img_feats(img, params={}):
    '''
    Calcultes feature vector for an RGB img with respect to params
    params - is a map with keys 
    use_spat, use_hist, use_hog - if True, include features of corresponding type to feature vector
    spat_* - same params as for spatial_feats
    hist_* - same params as for hist_feats
    hog_* - same params as for hog_feats
    hog_precalc - precalculated HOG, if precalculation used, have to be provided with x_bounds and y_bounds keys - region on which HOG was precalculated
    '''
    use_spatial        = params.get('use_spat', True)
    spat_cspace        = params.get('spat_cspace', 'RGB')
    spat_size          = params.get('spat_size', 32)
    spat_fts           = []
    if use_spatial:
        spat_fts = spatial_feats(img, cspace=spat_cspace, 
                            size=spat_size)

    use_hist           = params.get('use_hist', True)
    hist_cspace        = params.get('hist_cspace', 'RGB')
    hist_bins          = params.get('hist_bins', 32)
    hist_chan_range    = params.get('hist_chan_range')
    hist_fts           = []
    if use_hist:
        hist_fts, hists = hist_feats(img, 
                            cspace=hist_cspace, chan_range=hist_chan_range, bins=hist_bins)

    use_hog            = params.get('use_hog', True)
    hog_orient         = params.get('hog_orient', 9)
    hog_pix_per_cell   = params.get('hog_pix_per_cell', 8) 
    hog_cell_per_block = params.get('hog_cell_per_block', 2)
    hog_cspace         = params.get('hog_cspace', 'gray')
    hog_chan_range     = params.get('hog_chan_range')
    hog_fts            = []
    if use_hog:
        if params.get('hog_precalc') is not None:
            hog_fts = hog_feats_precalc(params)
        else:
            hog_fts = hog_feats(img, hog_orient, 
                              hog_pix_per_cell, hog_cell_per_block,
                              cspace=hog_cspace, chan_range=hog_chan_range)
    return np.concatenate((spat_fts, hist_fts, hog_fts))

def get_feats(imgs, params, wins = None):
    '''
    Returns feature vectors for RGB imgs, calculted with respect to params
    params - feature extraction params See get_img_feats
    wins - windows corresponding to provided imgs, have to be specified when HOG precalc is used
    '''
    feats = []
    i = 0
    for img in imgs:
        if wins is not None:
            params['win'] = wins[i]
        img_feats = get_img_feats(img, params)
        feats.append(img_feats)
        i +=1
    return feats

def prepare_train_test_sets(cars, noncars, params={},
                            test_size=0.2, random_state=0, take_samples=None):
    '''
    Prepares training and test sets of samples
    cars - array of RGB cars images 64x64
    noncars - array of RGB non cars images 64x64
    params - feature extraction params See get_img_feats
    test_size - ratio of test set size relative to whole dataset
    random_state - shuffle random state initial value
    take_samples - number of samples to take for each class
    '''
    if take_samples is None:
        take_samples = min(len(cars), len(noncars))

    cars = cars[:take_samples]
    noncars = noncars[:take_samples]

    car_imgs = [imread(imf) for imf in cars]
    noncar_imgs = [imread(imf) for imf in noncars]
    t=time.time()
    car_feats = get_feats(car_imgs, params)
    noncar_feats = get_feats(noncar_imgs, params)

    X = np.vstack((car_feats, noncar_feats)).astype(np.float64)
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)
    t2 = time.time()
    y = np.hstack((np.ones(len(car_feats)), np.zeros(len(noncar_feats))))

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                        test_size=test_size, random_state=random_state)
    return [X_train, y_train], [X_test, y_test], scaler, round((t2-t)/len(X), 7)

def train_svm(X, y, C=None, dual=None):
    '''
    Returnes trained SVM classifier with linear kernel, C and dual paramaters used
    X - vector of samples (future vectors)
    y - vector of labels (1 - car, 0 - noncar)
    '''
    if C is None:
        C = 1.0
    if dual is None:
        dual = True
    clf = LinearSVC(C=C, dual=dual)
    t=time.time()
    clf.fit(X, y)
    t2 = time.time()
    return clf, round(t2-t, 2)

def test_clf(clf, X, y, n_predicts=None):
    '''
    Evaluates trained clf on test samples X. Returnes prediction, real labels, accuracy and time took to predict per sample 
    y - real labels
    n_predicts - number ofsambles to take from X for prediction
    '''
    accuracy = round(clf.score(X, y), 4)
    if n_predicts is not None:
        X = X[:n_predicts]
        y = y[:n_predicts]
    t=time.time()
    y_predicted = clf.predict(X)
    t2 = time.time()
    return y_predicted, y, accuracy, round((t2-t)/len(X), 7)

def gen_param_set(param_ranges):
    '''
    Generates grid_like params set based on params_ranges
    params_ranges - map with same keys as for get_img_feats and values - vectors of possible param values
    '''
    def filter_without(filt, params_set, keys):
        remove_ks = [k for k in keys if k.startswith(filt)]
        use_param = 'use_'+filt
        with_use = [ps for ps in params_set if ps.get(use_param, True)]
        without_use = [ps for ps in params_set if not ps.get(use_param, True)]

        filtered = []
        for ps in without_use:
            for k in remove_ks:
                ps.pop(k, None)
            filtered.append(ps)
        filtered = [dict(s) for s in set(frozenset(d.items()) for d in filtered)]
        with_use.extend(filtered)
        return with_use

    keys = [] 
    vals = []
    for key, value in sorted(param_ranges.items()):
        keys.append(key)
        vals.append(value)

    params_set = []

    for param_vals in itertools.product(*vals):
        params = dict(zip(keys, param_vals))
        params_set.append(params)

    params_set = filter_without('spat', params_set, keys)
    params_set = filter_without('hist', params_set, keys)
    params_set = filter_without('hog', params_set, keys)
    
    params_set = [ps for ps in params_set if (ps.get('use_spat', True) or ps.get('use_hist', True) or ps.get('use_hog', True))]

    return params_set

def build_classifier(cars, noncars, params, nsamples=None, result_file=None):
    '''
    Builds train and test set from cars and noncars images.
    Extracts features using params, see get_img_feats
    Takes nsamples of each class to train\test set
    Writes trained classifier, scaler and params to result_file if specified
    '''
    train, test, scaler, exttime = prepare_train_test_sets(cars, noncars, 
                                        params=params, take_samples=nsamples)
    print (exttime,'s per img to extract')

    clf, traintime = train_svm(train[0], train[1], params.get('C'), params.get('dual'))

    print (traintime,'s to train')
    
    y_predicted, y, accuracy, predtime = test_clf(clf, test[0], test[1])

    print (predtime,'s per img to predict')
    print ('accuracy', accuracy)
    print ('total time for',len(y),'imgs =',round((exttime+predtime)*len(y),4),'s')
    if (result_file is not None):
        result = {'clf': clf, 'scaler': scaler, 'params': params,
                  'accuracy': accuracy, 'exttime': exttime, 
                  'predtime': predtime, 'traintime': traintime}
        save(result, result_file)
    return clf, scaler, params, accuracy, exttime, predtime, traintime  

def predict(imgs, clf, scaler, params, wins=None, dec_fn=False):
    '''
    Computes predictions for imgs, using classifier clf, scaler and params for feature extraction
    wins - corresponding sliding windows, have to be provided when HOG precalculation is used
    dc_fn - if True, decision function values will be returned along with predictions
    '''
    X = get_feats(imgs, params, wins=wins)
    scaled_X = scaler.transform(X)
    y_predicted = clf.predict(scaled_X)
    if dec_fn:
        return y_predicted, clf.decision_function(X)
    else:    
        return y_predicted

def brute_force_params(cars, noncars, params_ranges, nsamples, result_file, save_each=20):
    '''
    Builds, trains and tests classifier using carsand noncars images for each possible combination of params in params_ranges
    Takes nsamples for each class
    Writes statistics to result_file each save_each combination of parameters
    '''
    params_sets = gen_param_set(params_ranges)
    total = len(params_sets)
    cur = 1
    result = []
    spend_time = 0
    for ps in params_sets:
        t = time.time()
        print ('----------- {} of {}--------------'.format(cur,total))
        print ('params:', ps,'\n')
        clf, scaler, params, accuracy, exttime, predtime, traintime = build_classifier(cars, noncars, ps, nsamples)
        t2 = time.time()
        last_time = t2-t
        spend_time += last_time
        print ('----------- {}m of {}m--------------'.format(round(spend_time/60,1),
                                                             round(total*last_time/60,1)))
        result.append([ps, [exttime, predtime, accuracy, traintime]])
        if (result_file is not None) and (cur >= save_each) and (cur % save_each == 0):
            save(result, result_file)
        cur += 1

    if (result_file is not None):
        save(result, result_file)
    return result

def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5), pix_per_cell = None):
    '''
    Computes sliding windows of size xy_window with xy_overlap for img with img_shape
    using x boundings x_start_stop, y boundings y_start_stop
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None: 
        y_start_stop[1] = img_shape[0]
    # Adjust wins to cells if pix_per_cell is provided
    if pix_per_cell is not None:
        #print('original: ',xy_window,xy_overlap,x_start_stop,y_start_stop)
        xy_overlap[0]= (xy_overlap[0]*xy_window[0]//pix_per_cell)*pix_per_cell/xy_window[0]
        xy_overlap[1]= (xy_overlap[1]*xy_window[1]//pix_per_cell)*pix_per_cell/xy_window[1]
        #print('adjusted: ',xy_window,xy_overlap,x_start_stop,y_start_stop)
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    if pix_per_cell is not None:
        nx_pix_per_step = (nx_pix_per_step // pix_per_cell) * pix_per_cell
        ny_pix_per_step = (ny_pix_per_step // pix_per_cell) * pix_per_cell

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        starty = ys*ny_pix_per_step + y_start_stop[0]
        endy = starty + xy_window[1]
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def prepare_win_specs(win_specs, img_shape, base_win, base_overlap):
    '''
    Initializes windows specifications feeling missing parameters
    Computes maximum ROI bounds
    win_specs - vector of maps with keys - parameters to slide_window
    '''
    bounds = []
    win_specs = win_specs.copy()
    for ws in win_specs:
        if 'x_start_stop' not in ws:
            ws['x_start_stop'] = [0, img_shape[1]]
        if 'y_start_stop' not in ws:
            ws['y_start_stop'] = [0, img_shape[0]]
        if 'xy_window' not in ws:
            ws['xy_window'] = base_win
        if 'xy_overlap' not in ws:
            ws['xy_overlap'] = base_overlap
        x_s, x_e = ws['x_start_stop']
        y_s, y_e = ws['y_start_stop']
        bounds.append([x_s, x_e, y_s, y_e])
    bounds = np.array(bounds)
    return win_specs, [np.min(bounds[:,0]),np.max(bounds[:,1])], [np.min(bounds[:,2]),np.max(bounds[:,3])]

def search_cars(img, builtclf, win_specs, precalc_hog=False, dec_fn=False, dec_thre=0):
    '''
    Finds sliding windows with nonzero predictions 
    img - RGB img to analize
    buildclf - map with keys clf - trained classifier, scaler - fitted scaler, params - feature extraction params 
    win_specs - vector of maps with keys - parameters to slide_window
    precalc_hog - if True, HOG precalculation will b used
    dec_fn - if True, decision function result will be considered and thresholded by dec_thre as low threshold
    '''
    base_win = [64,64]
    base_overlap = [0.5,0.5]
    params = builtclf['params']
    params['base_win'] = base_win
    pix_per_cell = None
    if precalc_hog:
        pix_per_cell = params['hog_pix_per_cell']
    win_specs, x_bounds, y_bounds = prepare_win_specs(win_specs, img.shape, base_win, base_overlap)
    wins = []
    for spec in win_specs:
        x_start_stop = np.array(spec['x_start_stop'])
        y_start_stop = np.array(spec['y_start_stop'])
        cur_win = spec['xy_window']
        x_scale = base_win[0]/cur_win[0]
        y_scale = base_win[1]/cur_win[1]

        scaled = cv2.resize(img, (0,0), fx=x_scale, fy=y_scale)
        # Adjust bounds to cells if pix_per_cell is provided
        if pix_per_cell is not None:
            params['x_bounds'] = np.uint32(np.array(x_bounds)*x_scale // pix_per_cell * pix_per_cell)
            params['y_bounds'] = np.uint32(np.array(y_bounds)*y_scale // pix_per_cell * pix_per_cell)
            cur_x_bounds = np.uint32(x_start_stop*x_scale // pix_per_cell * pix_per_cell)
            cur_y_bounds = np.uint32(y_start_stop*y_scale // pix_per_cell * pix_per_cell)
        else:    
            params['x_bounds'] = np.uint32(np.array(x_bounds)*x_scale)
            params['y_bounds'] = np.uint32(np.array(y_bounds)*y_scale)
            cur_x_bounds = np.uint32(x_start_stop*x_scale )
            cur_y_bounds = np.uint32(y_start_stop*y_scale )
        if precalc_hog:
            params['hog_precalc'] = hog_precalculate(scaled, params)

        spec_wins = slide_window(scaled.shape, 
                        cur_x_bounds,
                        cur_y_bounds,
                        base_win, spec['xy_overlap'], pix_per_cell=pix_per_cell)  

        imgs = [scaled[win[0][1]:win[1][1], win[0][0]:win[1][0]] for win in spec_wins]
        prediction = predict(imgs,builtclf['clf'], builtclf['scaler'], params, wins=spec_wins, dec_fn=dec_fn)
        if dec_fn:
            found_wins = np.array(spec_wins)[(prediction[0] > 0)&(prediction[1] > dec_thre)].tolist()    
        else:
            found_wins = np.array(spec_wins)[(prediction > 0)].tolist()
        for win in found_wins:
            win[0][0] = int(win[0][0] / x_scale)
            win[0][1] = int(win[0][1] / y_scale)
            win[1][0] = int(win[1][0] / x_scale)
            win[1][1] = int(win[1][1] / y_scale) 
        #if True:#(len(found_wins)<5):
        #    i = builtclf.get('i',0)
        #    wimgs = [img for img,p in zip(imgs,prediction) if p>0]#
        #    for wimg in wimgs:
        #        mpimg.imsave('{}{}-{}{}'.format('./hnmt/', 'win', i, '.jpg'), wimg)
        #        i+=1
        #    builtclf['i']=i
        wins.extend(found_wins)
    return np.array(wins)

def search_cars_v1(img, builtclf, win_specs, precalc_hog=False):
    '''
    Naive version of search_cars
    '''
    wins = []
    for spec in win_specs:
        spec_wins = slide_window(img.shape, 
                        x_start_stop=spec.get('x_start_stop', [None,None]),
                        y_start_stop=spec.get('y_start_stop', [None,None]),
                        xy_window=spec.get('xy_window', (64,64)),
                        xy_overlap=spec.get('xy_overlap', (0.5,0.5)))  
        wins.extend(spec_wins)
    imgs = [cv2.resize(img[win[0][1]:win[1][1], win[0][0]:win[1][0]], (64,64)) for win in wins]
    
    prediction = predict(imgs,builtclf['clf'], builtclf['scaler'], builtclf['params'], wins=wins)

    wins = np.array(wins)
    return wins[prediction > 0]

def add_heat(heatmap, bbox_list):
    '''
    Increases increases `heat` of pixels of heatmap each time they falls into bounding box from bbox_list
    '''
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def chan_threshold(chan, lo_thresh):
    '''
    Returns copy of chan with all pixels with values lower than lo_thresh set to zero
    '''
    chan = np.copy(chan)
    chan[chan <= lo_thresh] = 0
    return chan

def label_heatmap(heatmap, lo_thresh):
    '''
    Labels all separate nonzero regions of heatmap after thresholding
    heatmap - single channelled image
    lo_thresh - same as for chan_threshold 
    '''
    heatmap = chan_threshold(heatmap, lo_thresh)
    labels = label(heatmap)
    return labels

def labels_bboxes(labels):
    '''
    Returnes bounding boxes by labeled heatmap
    labels - pair [labeled image, labels]
    '''
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
    return bboxes

def filter_outlier_boxes(bboxes, total_heat, heat_max_lo_thresh, lo_xy_thresh=[48,48]):
    '''
    Filters bboxes out of bounding boxes of size less then lo_xy_thresh, or maximum heat withing box less then heat_max_lo_thresh
    '''
    result = []
    for box in bboxes:
        if (box[1][0]-box[0][0] > lo_xy_thresh[0]) \
            and (box[1][1]-box[0][1] > lo_xy_thresh[1])\
            and (np.max(total_heat[box[0][1]:box[1][1],box[0][0]:box[1][0]]) >= heat_max_lo_thresh):
            result.append(box)
    return result

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    Draws each of bounding box from bboxes on copy of img using color and thick 
    '''
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
    return imcopy

def prepare_dataset(base_dir, tgt_file):
    '''
    Composes text tgt_file with paths to all imgs inside child folders of base_dir
    '''
    dirs = os.listdir(base_dir)
    count = 0
    with open(tgt_file, 'w') as f:
        for dir in dirs:
            for fn in glob.glob(os.path.join(base_dir,dir,'*')):
                count +=1
                f.write(fn+'\n')
    return count
    
def load_dataset(src_file):
    '''
    Loads images from paths specified in text src_file
    '''
    result = None
    with open(src_file) as f:
        result = [x.strip('\n') for x in f.readlines()]
    return result

def imread(img_file):
    '''
    Uniformly reads png and jpg img_file as np RGB array 
    '''
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def process_video(src_path, process_fn, tgt_path):
    '''
    Processes frames of video file on src_path, by process_fn and saves resulting video to file on tgt_path
    '''
    clip = VideoFileClip(src_path)
    white_clip = clip.fl_image(process_fn) 
    white_clip.write_videofile(tgt_path, audio=False)
    pass

def save_video_frames(src_path, tgt_path, start=None, stop = None):
    '''
    Saves frames of video file on src_path to directory on tgt_path
    '''
    clip = VideoFileClip(src_path)
    frames = clip.iter_frames()
    if start is None:
        start = 0
    if stop is None:
        stop = int(clip.fps * clip.duration)+100
    src_filename = os.path.basename(src_path)
    fn = 'frame'
    ext = '.jpg'
    i = 0
    for frame in frames:
        if (i>=start)and (i<=stop):
            print ('{}{}-{}{}'.format(tgt_path, fn, i, ext)) 
            mpimg.imsave('{}{}-{}{}'.format(tgt_path, fn, i, ext), frame)
        i = i + 1
    pass
    
def plot_img_grid(images, rows=None, cols=1, 
                    titles=None, 
                    figid=None, figsize=(9, 4), 
                    hspace=0.0, wspace=0.0,
                    cmaps=None):
    '''
    Plots grid of specified images it in a window 
    '''
    if rows is None:
        rows = len(images)
    cellscnt = rows*cols
    fig = plt.figure(figid, figsize)
    gs = gridspec.GridSpec(rows, cols, hspace=hspace, wspace=wspace)
    axs = [plt.subplot(gs[i]) for i in range(cellscnt)]
    imgslen = len(images)
    tlen = 0
    if titles is not None: 
        tlen = len(titles)
    cmlen = 0
    if cmaps is not None: 
        cmlen = len(cmaps)
    for i in range(cellscnt):
        if i < imgslen:
            img = images[i]
            if i < tlen: 
                axs[i].set_title(titles[i])
            if (i < cmlen) and cmaps[i] is not None:
                axs[i].imshow(img, cmap=cmaps[i])
            else:
                axs[i].imshow(img)
    gs.tight_layout(fig, pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()

def calc_bin_centers(hist):
    '''
    Calculates bin centers of hist
    '''
    bin_edges = hist[1]
    return (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

def plot_hists(hists):
    '''
    Plots array of histograms
    hists - array of len 3 of numpy hists  
    '''
    bincen = calc_bin_centers(hists[0])
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bincen, hists[0][0])
    plt.xlim(0, 256)
    plt.title('0 Histogram')
    plt.subplot(132)
    plt.bar(bincen, hists[1][0])
    plt.xlim(0, 256)
    plt.title('1 Histogram')
    plt.subplot(133)
    plt.bar(bincen, hists[2][0])
    plt.xlim(0, 256)
    plt.title('2 Histogram')
    fig.tight_layout()
    plt.show()

def load(path):
    '''
    Loads serialized obj from file on path
    '''
    obj = pickle.load(open(path, 'rb'))
    return obj

def save(obj, path):
    '''
    Serializes obj and saves to file specified by path
    '''
    pickle.dump(obj, open(path, 'wb'))