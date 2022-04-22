import random
import os
import numpy as np
import torch

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

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

def get_max_preds(batch_hmaps):
	batch_size = batch_hmaps.shape[0]
	num_joints = batch_hmaps.shape[1]
	width      = batch_hmaps.shape[3]

	hmaps_reshaped = batch_hmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(hmaps_reshaped, 2)
	maxvals           = np.amax(hmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask

	return preds, maxvals

def dist_acc(dists, thresholds = 0.5):
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], thresholds).sum() * 1.0 / num_dist_cal
	else:
		return -1

def calc_dists(preds, target, normalize):
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0]))

	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds   =  preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n]    = -1

	return dists

def accuracy(output, target, thr_PCK, thr_PCKh, threshold=0.5):
	num_joint = output.shape[1]
	norm = 1.0

	pred, _   = get_max_preds(output)
	target, _ = get_max_preds(target)

	h         = output.shape[2]
	w         = output.shape[3]
	norm      = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10

	dists = calc_dists(pred, target, norm)

	acc     = np.zeros(num_joint)
	avg_acc = 0
	cnt     = 0
	visible = np.zeros(num_joint)

	for i in range(num_joint):
		acc[i] = dist_acc(dists[i])
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros(num_joint)
	avg_PCKh = 0

	headLength = np.linalg.norm(target[:,9,:] - target[:,10,:])

	for i in range(num_joint):
		PCKh[i] = dist_acc(dists[i], thr_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh


	# PCK
	PCK = np.zeros(num_joint)
	avg_PCK = 0

	torso = np.linalg.norm(target[:,0,:] - target[:,9,:])

	for i in range(num_joint):
		PCK[i] = dist_acc(dists[i], thr_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK

	return acc, PCK, PCKh, cnt, pred, visible

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
    '''
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
    '''
	
    import numpy as np
    hmaps = np.random.randn(2, 17, 32, 32)
    output = np.random.randn(2, 17, 32, 32)
    acc, PCK, PCKh, cnt, pred, visible = accuracy(output, hmaps, 0.2, 0.5)
    print(acc)
    print(PCK)