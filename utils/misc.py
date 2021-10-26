import random
import os
import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(heatmaps_reshaped, 2)
	maxvals           = np.amax(heatmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask

	return preds, maxvals

if __name__ == '__main__':
    import cv2
    from glob import glob

    root_path = '/scratch/PI/cqf/datasets/h36m'
    img_path = root_path + '/img'
    pos2d_path = root_path + '/pos2d'

    img_fns = glob(img_path+'/*.jpg')
    '''
    cm = AverageMeter()
    for img_fn in img_fns:
        img = cv2.imread(img_fn)
        mean = np.mean(img, axis=(0,1))
        cm.update(mean)
        if cm.count % 10000 == 0:
            print(f'Iter {cm.count}: {cm.avg}')
    print(cm.avg)
    '''
    #from tqdm import tqdm
    import time
    mean = np.array([66.49247512, 70.43421173, 112.82712325])
    std = AverageMeter()
    start_time = time.time()
    for img_fn in img_fns:
        img = cv2.imread(img_fn)
        img = np.mean(np.square(img.reshape(-1, 3) - mean), axis=0)
        std.update(img)
        if std.count % 10000 == 0:
            print(f'Iter {std.count}: {std.avg}')
    print(np.sqrt(std.avg))
    print((time.time() - start_time)/ 3600.0)