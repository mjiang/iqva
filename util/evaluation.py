import scipy
import numpy as np
import cv2

eps = 1e-15 #regularization value

def g_filter(shape =(200,200), sigma=60):
    """
    Using Gaussian filter to generate center bias
    """
    x, y = [edge /2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in np.arange(-x, x)] for j in np.arange(-y, y)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter


def add_center_bias(salMap):
    cb = g_filter()
    cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[0]),interpolation = cv2.INTER_LINEAR)
    salMap = salMap*cb
    return salMap

def cal_cc_score(salMap, fixMap):
    """
    Compute CC score between two attention maps
    """
    if np.sum(salMap)>0:
        salMap /= np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)
    score = np.corrcoef(salMap.reshape(-1), fixMap.reshape(-1))[0][1]

    return score

def cal_sim_score(salMap, fixMap):
    """
    Compute SIM score between two attention maps
    """
    salMap = salMap/np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)

    sim_score = np.sum(np.minimum(salMap,fixMap))

    return sim_score


def cal_kld_score(salMap,fixMap): #recommand salMap to be free-viewing attention
    """
    Compute KL-Divergence score between two attention maps
    """
    salMap = salMap/np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)
    kl_score = fixMap*np.log(eps+fixMap/(salMap+eps))
    kl_score = np.sum(kl_score)

    return kl_score


def cal_auc_score(salMap, fixMap):
    """
    compute the AUC score for saliency prediction
    """
    salMap /= salMap.max()
    fixmap = (fixMap==1).astype(int)
    salShape = salMap.shape
    fixShape = fixmap.shape

    predicted = salMap.reshape(salShape[0]*salShape[1], -1,
                               order='F').flatten()
    actual = fixmap.reshape(fixShape[0]*fixShape[1], -1,
                            order='F').flatten()
    labelset = np.arange(2)

    auc = area_under_curve(predicted, actual, labelset)
    return auc

def area_under_curve(predicted, actual, labelset):
    tp, fp = roc_curve(predicted, actual, np.max(labelset))
    auc = auc_from_roc(tp, fp)
    return auc

def auc_from_roc(tp, fp):
    h = np.diff(fp)
    auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
    return auc

def roc_curve(predicted, actual, cls):
    si = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[si]==cls))
    fp = np.cumsum(np.single(actual[si]!=cls))
    tp = tp/np.sum(actual==cls)
    fp = fp/np.sum(actual!=cls)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp

def cal_nss_score(salmap,fixmap,center_bias=False):
    #compute the normalized scanpath saliency
    salmap = (salmap-np.mean(salmap))/np.std(salmap)
    
    return np.sum(salmap * fixmap)/np.sum(fixmap)

def cal_sauc_score(salMap,fixmap,shufMap,stepSize=.01):
    salMap -= salMap.min()
    if salMap.max()>0:
        salMap /= salMap.max()
    rows,cols = [], []
    # get the fixation localtion
    for i in range(len(fixmap)):
        for j in range(len(fixmap[i])):
            if fixmap[i][j] > 0:
                rows.append(i)
                cols.append(j)

    Sth = np.asarray([ salMap[y][x] for y,x in zip(rows, cols) ])
    Nfixations = len(rows)

    others = np.copy(shufMap)
    for y,x in zip(rows, cols):
        others[y][x] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = salMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix) + 1e-6

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

def distortion_corr(input):
    h,w = input.shape
    weight = np.sin((np.arange(h)/h)*np.pi).reshape(-1,1)
    weight = np.repeat(weight,w,axis=1)
    return input*weight


def get_fix_pos(pred,gt):
    pred_x, pred_y = np.unravel_index(np.argmax(pred,axis=None),pred.shape)
    pred_x_norm, pred_y_norm = pred_x/pred.shape[0], pred_y/pred.shape[1]
    gt_x, gt_y = np.unravel_index(np.argmax(gt,axis=None),gt.shape)
    gt_x_norm, gt_y_norm = gt_x/gt.shape[0], gt_y/gt.shape[1]
    return (pred_y_norm,pred_x_norm), (gt_y_norm,gt_x_norm) # width x height


def cal_fix_mse(pred,gt,fixation):
    score = fixation*((pred-gt)**2)
    return score.sum()/np.count_nonzero(fixation)
