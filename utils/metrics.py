import torch
import numpy as np

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

def accuracy_2d_pose(output, target, thr_PCK, thr_PCKh, threshold=0.5):
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

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    if len(predicted.shape) == 4:
        B, T, N, D = predicted.shape
        predicted_new = predicted.reshape(B * T, N, D)
        target_new = target.reshape(B * T, N, D)
    else:
        predicted_new = predicted
        target_new = target

    muX = np.mean(target_new, axis=1, keepdims=True)
    muY = np.mean(predicted_new, axis=1, keepdims=True)

    X0 = target_new - muX
    Y0 = predicted_new - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted_new, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target_new, axis=len(target_new.shape) - 1))


class Accuracy:
    def __call__(self, preds, labels):
        return (preds.argmax(dim=-1) == labels).float().mean()

class MPCK_MPCKh:
    def __init__(self, thr_pck=0.2, thr_pckh=0.5):
        self.thr_pck = thr_pck
        self.thr_pckh = thr_pckh
    def __call__(self, preds, targets):
        preds_array = preds.cpu().detach().numpy()
        targets_array = targets.cpu().detach().numpy()
        _, PCK, PCKh, _, _, _ = accuracy_2d_pose(preds_array, targets_array, 0.2, 0.5)
        return torch.tensor([PCK.mean(), PCKh.mean()])

class MPJPE_PMPJPE:
    def __call__(self, preds, targets):
        mPJPE = mpjpe(preds.detach(), targets.detach())
        pmPJPE = p_mpjpe(preds.cpu().detach().numpy(), targets.cpu().detach().numpy())
        pmPJPE = torch.tensor(pmPJPE)
        return torch.tensor([mPJPE, pmPJPE])