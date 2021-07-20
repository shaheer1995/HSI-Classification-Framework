import math
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


def load_dataset(Dataset, split):
    data_path = '../dataset/'

    if Dataset == 'WHU_HC':
        d = sio.loadmat(data_path + 'WHU_Hi_HanChuan.mat')
        gtd = sio.loadmat(data_path + 'WHU_Hi_HanChuan_gt.mat')
        data_hsi = d['WHU_Hi_HanChuan']
        gt_hsi = gtd['WHU_Hi_HanChuan_gt']
        K = 274
        TOTAL_SIZE = 144788
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    
    if Dataset == 'WHU_LK':
        d = sio.loadmat(data_path + 'WHU_Hi_LongKou.mat')
        gtd = sio.loadmat(data_path + 'WHU_Hi_LongKou_gt.mat')
        data_hsi = d['WHU_Hi_LongKou']
        gt_hsi = gtd['WHU_Hi_LongKou_gt']
        K = 270
        TOTAL_SIZE = 204542
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'WHU_HH':
        d = sio.loadmat(data_path + 'WHU_Hi_HongHu.mat')
        gtd = sio.loadmat(data_path + 'WHU_Hi_HongHu_gt.mat')
        data_hsi = d['WHU_Hi_HongHu']
        gt_hsi = gtd['WHU_Hi_HongHu_gt']
        K = 270
        TOTAL_SIZE = 386693
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat(data_path + 'PaviaU.mat')
        gt_uPavia = sio.loadmat(data_path + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        K = 103
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'IN':
        mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
        mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        K = 200
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    
  
    shapeor = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    data_hsi = PCA(n_components=K).fit_transform(data_hsi)
    shapeor = np.array(shapeor)
    shapeor[-1] = K
    data_hsi = data_hsi.reshape(shapeor)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT