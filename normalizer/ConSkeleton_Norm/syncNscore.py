from norm_used import MMFitMetaData
import numpy as np
from numpy.linalg import norm
from numpy import dot
from dtaidistance import dtw
from dtaidistance import dtw_ndim

#n_dim.warping_paths: This method returns the dependent DTW (DTW_D) [1] distance between two n-dimensional sequences. 
# If you want to compute the independent DTW (DTW_I) distance, use the 1-dimensional version: 
# that’s why have to do on each kp

def matchingFrame(input_skeleton,standard_skeleton):
    num_of_frames = input_skeleton.shape[0]
    num_of_keypoints = input_skeleton.shape[1]
    match_pairs = []
    match_pair_dict = {input_frame : [] for input_frame in range(num_of_frames)}
    for kp in num_of_keypoints:
        input_kp_frames = input_skeleton[:,kp,:]
        standard_kp_frames = standard_skeleton[:,kp,:]
        dis, all_possible_pairs = dtw_ndim.warping_paths(input_kp_frames,standard_kp_frames)
        match_pairs_kp = dtw.best_path(all_possible_pairs) ##output: array of pairs, mapping each frame_input to frame_output
        for each_pair in match_pairs_kp:
            match_pair_dict[each_pair[0]].append(each_pair[1])

    for frame in num_of_frames:
        matched_frame_of_standard = max(match_pair_dict[frame], key = match_pair_dict[frame].count)
        match_pairs.append((frame,matched_frame_of_standard))

    return match_pairs

def calculateSim(input_skeleton,standard_skeleton,match_pairs):
    NumOfFrames = input_skeleton.shape[0]
    dataParamsClass = MMFitMetaData()
    scoreOfFrames = []
    for each_pair in match_pairs:
        frame_of_input = each_pair[0]
        frame_of_standard = each_pair[1]
        skeletonEdges = dataParamsClass.skeleton_edges
        score_of_kps_EachFrame = []
        for edge in skeletonEdges:
        ## calculate vector between input_points(frame,edge[0]) and model_points(match_frame,edge[0])
            vector_input = input_skeleton[frame_of_input][edge[0]] - input_skeleton[frame_of_input][edge[1]]
            vector_model = standard_skeleton[frame_of_standard][edge[0]] - standard_skeleton[frame_of_standard][edge[1]]

            # now calculate the cosinmilarity
            CosSim = dot(vector_input, vector_model)/(norm(vector_input)*norm(vector_model))

            #skeletonsOfFrames_dict[frame].append(vector_input)
            score_of_kps_EachFrame.append(CosSim)

        scoreOfFrames.append(sum(score_of_kps_EachFrame)/16)

    similarity = sum(scoreOfFrames)/NumOfFrames

    return similarity

def percentageScore(similarity):
    percentage = similarity* 100
    return int(percentage)

        
