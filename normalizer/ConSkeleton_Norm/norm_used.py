import numpy as np
from numpy.linalg import norm

class MMFitMetaData:
    num_classes = 11
    classes = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5,
               'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8,
               'lateral_shoulder_raises': 9, 'non_activity': 10}
    num_joints = 17
    joint_labels = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
                         'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 
                         'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    skeleton_edges = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), 
                      (9, 8), (10, 9), (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]

    skeleton_avg_length = [0.13300511, 0.45407227, 0.45066762, 0.13300464, 0.45407124, 0.45066731,
                           0.24290385, 0.25399462, 0.11620144, 0.11500009, 0.15595447, 0.28262905,
                           0.24931334, 0.15595369, 0.28263132, 0.24931305]

def NormEachPoint(skeleton):
    num_kp, num_dim = skeleton.shape
    norm_skeleton = np.zeros((num_kp, num_dim))
    for kp in range(num_kp):
        if kp == 0:
            continue
        norm_skeleton[kp] = skeleton[kp]/norm(skeleton[kp])

    return norm_skeleton

def Norm17dVector(skeleton):
    num_kp, num_dim = skeleton.shape
    norm_skeleton = np.copy(skeleton)
    norm_skeleton = norm_skeleton.reshape(num_kp*num_dim,)
    norm_skeleton = norm_skeleton/norm(norm_skeleton)
    norm_skeleton = norm_skeleton.reshape(17,-1)
    return norm_skeleton


def NormWFK(skeleton):
    dataclass = MMFitMetaData()
    conditional_skeleton = np.ones((17,3))
    conditional_skeleton[0,:] = [0,0,0]

    #batch 1
    #1,4,7
    vector10 = dataclass.skeleton_avg_length[0]*(skeleton[1]/norm(skeleton[1]))
    vector40 = dataclass.skeleton_avg_length[3]*(skeleton[4]/norm(skeleton[4]))
    vector70 = dataclass.skeleton_avg_length[6]*(skeleton[7]/norm(skeleton[7]))
    conditional_skeleton[1] = vector10 + [0,0,0]
    conditional_skeleton[4] = vector40 + [0,0,0]
    conditional_skeleton[7] = vector70 + [0,0,0]
    #batch 2
    #2,5,8
    vector21 = skeleton[2] - skeleton[1]
    vector54 = skeleton[5] - skeleton[4]
    vector87 = skeleton[8] - skeleton[7]

    vector21 = dataclass.skeleton_avg_length[1]*(vector21/norm(vector21))
    vector54 = dataclass.skeleton_avg_length[4]*vector54/norm(vector54)
    vector87 = dataclass.skeleton_avg_length[7]*vector87/norm(vector87)

    conditional_skeleton[2] = vector21 + conditional_skeleton[1]
    conditional_skeleton[5] = vector54 + conditional_skeleton[4]
    conditional_skeleton[8] = vector87 + conditional_skeleton[7]
    conditional_skeleton
    #batch 3
    #3,6,9,11,14
    vector32 = skeleton[3] - skeleton[2]
    vector65 = skeleton[6] - skeleton[5]
    vector98 = skeleton[9] - skeleton[8]
    vector118 = skeleton[11] - skeleton[8]
    vector148 = skeleton[14] - skeleton[8]


    vector32 = dataclass.skeleton_avg_length[2]*vector32/norm(vector32)
    vector65 = dataclass.skeleton_avg_length[5]*vector65/norm(vector65)
    vector98 = dataclass.skeleton_avg_length[8]*vector98/norm(vector98)
    vector118 = dataclass.skeleton_avg_length[10]*vector118/norm(vector118)
    vector148 = dataclass.skeleton_avg_length[13]*vector148/norm(vector148)

    conditional_skeleton[3] = vector32 + conditional_skeleton[2]
    conditional_skeleton[6] = vector65 + conditional_skeleton[5]
    conditional_skeleton[9] = vector98 + conditional_skeleton[8]
    conditional_skeleton[11] = vector118 + conditional_skeleton[8]
    conditional_skeleton[14] = vector148 + conditional_skeleton[8]
    conditional_skeleton
    #batch 4
    #10,12,15
    vector109 = skeleton[10] - skeleton[9]
    vector1211 = skeleton[12] - skeleton[11]
    vector1514 = skeleton[15] - skeleton[14]

    vector109 = dataclass.skeleton_avg_length[9]*vector109/norm(vector109)
    vector1211 = dataclass.skeleton_avg_length[11]*vector1211/norm(vector1211)
    vector1514 = dataclass.skeleton_avg_length[14]*vector1514/norm(vector1514)

    conditional_skeleton[10] = vector109 + conditional_skeleton[9]
    conditional_skeleton[12] = vector1211 + conditional_skeleton[11]
    conditional_skeleton[15] = vector1514 + conditional_skeleton[14]
    conditional_skeleton
    #batch5
    #13,16
    vector1312 = skeleton[13] - skeleton[12]
    vector1615 = skeleton[16] - skeleton[15]

    vector1312 = dataclass.skeleton_avg_length[12]*vector1312/norm(vector1312)
    vector1615 = dataclass.skeleton_avg_length[15]*vector1615/norm(vector1615)

    conditional_skeleton[13] = vector1312 + conditional_skeleton[12]
    conditional_skeleton[16] = vector1615 + conditional_skeleton[15]

    return conditional_skeleton