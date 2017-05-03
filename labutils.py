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

def cvt_color(img, cspace='RGB'):
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

def spatial_feats(img, cspace='RGB', size=32):
    feature_image = cvt_color(img, cspace)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, (size, size)).ravel() 
    # Return the feature vector
    return features    

def hist_feats(img, cspace = 'RGB', bins=32, hrange=(0, 256)):
    img = cvt_color(img, cspace)
    # Compute the histogram of the RGB channels separately
    chans = [img[:,:,i] for i in range(img.shape[2])]
    hists = [np.histogram(ch, bins=bins, range=hrange) for ch in chans]
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate([hist[0] for hist in hists])
    return hist_features, hists

def hog_chan_feats(chan_img, orient, pix_per_cell, cell_per_block, 
              vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = skhog(chan_img, orientations=orient, 
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), 
                                transform_sqrt=False, 
                                visualise=True, feature_vector=False)
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

def get_img_feats(img, params={}):
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
    hist_fts           = []
    if use_hist:
        hist_fts, hists = hist_feats(img, cspace=hist_cspace, 
                            bins=hist_bins)

    use_hog            = params.get('use_hog', True)
    hog_orient         = params.get('hog_orient', 9)
    hog_pix_per_cell   = params.get('hog_pix_per_cell', 8) 
    hog_cell_per_block = params.get('hog_cell_per_block', 2)
    hog_cspace         = params.get('hog_cspace', 'gray')
    hog_chan_range     = params.get('hog_chan_range')
    hog_fts            = []
    if use_hog:
        hog_fts = hog_feats(img, hog_orient, 
                          hog_pix_per_cell, hog_cell_per_block,
                          cspace=hog_cspace, chan_range=hog_chan_range)
    return np.concatenate((spat_fts, hist_fts, hog_fts))

def get_feats(imgs, params):
    feats = []
    for img in imgs:
        img_feats = get_img_feats(img, params)
        feats.append(img_feats)
    return feats

def prepare_train_test_sets(cars, noncars, params={},
                            test_size=0.2, random_state=0, take_samples=None):
    if take_samples is not None:
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

def train_svm(X, y):
    clf = LinearSVC()
    t=time.time()
    clf.fit(X, y)
    t2 = time.time()
    return clf, round(t2-t, 2)

def test_clf(clf, X, y, n_predicts=None):
    accuracy = round(clf.score(X, y), 4)
    if n_predicts is not None:
        X = X[:n_predicts]
        y = y[:n_predicts]
    t=time.time()
    y_predicted = clf.predict(X)
    t2 = time.time()
    return y_predicted, y, accuracy, round(t2-t, 7)

def gen_param_set(param_ranges):
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

def brute_force_params(cars, noncars, params_ranges, nsamples, result_file, save_each=20):
    params_sets = gen_param_set(params_ranges)
    total = len(params_sets)
    cur = 1
    result = []
    spend_time = 0
    for ps in params_sets:
        t = time.time()
        print ('----------- {} of {}--------------'.format(cur,total))
        print ('params:', ps,'\n')
        train, test, scaler, exttime = prepare_train_test_sets(cars, noncars, 
                                            params=ps, take_samples=nsamples)
        print (exttime,'s per img to extract')

        clf, trtime = train_svm(train[0], train[1])

        print (trtime,'s to train')
        
        y_predicted, y, accuracy, predtime = test_clf(clf, test[0], test[1])

        print (predtime,'s to predict', len(y),'samples')
        print ('accuracy', accuracy)
        print ('total time for',len(y),'imgs =',exttime*len(y)+predtime,'s')
        t2 = time.time()
        last_time = t2-t
        spend_time += last_time
        print ('----------- {}m of {}m--------------'.format(round(spend_time/60,1),
                                                             round(total*last_time/60,1)))
        result.append([ps, [exttime, predtime/len(y), accuracy, trtime]])
        if result_file and (cur >= save_each) and (cur % save_each == 0):
            save(result, result_file)
        cur += 1

    if result_file:
        save(result, result_file)
    return result

def prepare_dataset(base_dir, tgt_file):
    dirs = os.listdir(base_dir)
    count = 0
    with open(tgt_file, 'w') as f:
        for dir in dirs:
            for fn in glob.glob(os.path.join(base_dir,dir,'*')):
                count +=1
                f.write(fn+'\n')
    return count
    
def load_dataset(src_file):
    result = None
    with open(src_file) as f:
        result = [x.strip('\n') for x in f.readlines()]
    return result

def imread(img_file):
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
    bin_edges = hist[1]
    return (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

def plot_hists(hists):
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