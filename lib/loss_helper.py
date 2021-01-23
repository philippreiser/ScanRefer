# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

# for cfg.prepare_epochs
from util.config import cfg

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = data_dict['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_reference_loss(data_dict, config):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # NOTE: data_dict["cluster_ref"] are the cluster confidences from the match_module.py
    #       B := batch_size

    # unpack
    cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # GT segmentation 
    # label creation without using the class labels and real ground thruths
    # because loc. loss should be independant of obj. class. loss and 
    # segmentation loss. (+ the same class can appear more than once)
    # hence we want to compare each cluster with the real cluster to find 
    # the best cluster. 
    gt_instances = data_dict['instance_labels'] # (B*N)
    target_inst_id = data_dict['object_id'] # (B)
    # as no extra batch_dim exists this gives the index of a next sample
    start_of_samples = data_dict['offset'] # (B)

    # PointGroup: 
    # NOTE: in PG clustering alg. only points of the same class can be in one cluster 
    
    preds_segmentation = data_dict['semantic_preds'] # (B*N), long
    # dim 1 for cluster_id, dim 2 for corresponding point idxs in N
    # sumNPoint: Total number of points belonging to some cluster 
    #            Some points don't get assigned a cluster
    preds_instances = data_dict['proposals_idx'] # (B*sumNPoint, 2)
    # to get the num_proposals we look at the length of the first sample in cluster_preds
    batch_size, num_proposals = len(start_of_samples), len(cluster_preds[:start_of_samples[1]])
    labels = torch.zeros(batch_size, num_proposals)

    # TODO: vectorize - instead of double iterative approach
    # for each sample in batch
    for i in range(batch_size):
        start = start_of_samples[i]
        end = start_of_samples[i+1]
        # gt_instances contains for each of the points their corresponding cluster_id
        # NOTE: we assume the point_ids are assigned based on their order in gt_instances 
        #       we also assume that these ids match with the point_ids from PG
        correct_indices = (
            torch.arange(
                len(gt_instances[start:end]))[
                gt_instances[start:end]==target_inst_id[i]
                ]
            ).cuda()
        # nSamples is the number of points that are asigned to some clusters in one scene
        # NOTE: only works with an extra batch_size dimension
        #nSamples = preds_instances[i].shape[0] 
        numbSamplePerCluster = torch.zeros(num_proposals)

        for j, point_in_cluster in enumerate(preds_instances[start:end]):
            cluster_id, member_point = point_in_cluster
            numbSamplePerCluster[cluster_id] += 1
            if int(member_point) in correct_indices: 
                # in preds_instances for every point there is one entry (one assigned cluster_id)
                # I want all of them to count for one sample (the underlying scan)
                # the cluster_id with the most counts will be the true label.
                # counts are defined as points being in the correct_indices (-> IoU)
                # use index instead of i if no batch_size, but all batches directly concatenated 
                # = what we first assumed would be how PG trains on multiple batches.
                #index = int(np.floor(j/nSamples))
                labels[i, cluster_id] += 1

        # union of points in real instance (gt) and respective pred instance
        # - labels to not have the intersection count double
        numbSamplePerCluster += len(correct_indices)-labels[i]
        # normalize intersection with union => IoU score now
        labels[i] = labels[i]/numbSamplePerCluster
        max_elem = labels[i].max()
        # convert to one-hot-matrix with 0 on max per row
        if max_elem != 0:
            labels[i] = torch.floor(labels[i]/max_elem)
        else:
            labels[i] = torch.zeros(len(labels[i])) # reset all counts
            # TODO: Sensible? Decoupling not completely given anymore! 
            #       All ones? (All zeros leads to low loss, because most preds are 0 --> we want loss increase)
            labels[i, target_inst_id] = 1 # If no IoU with GT 

    cluster_labels = torch.FloatTensor(labels).cuda()

    # TODO: check if cluster_id starts with 0

    # reference loss
    criterion = SoftmaxRankingLoss()
    loss = criterion(cluster_preds, cluster_labels.float().clone())
    return loss, cluster_preds, cluster_labels

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss

def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # PointGroup: 
    # no vote_loss, objectness_loss, box_loss, sem_cls_loss needed anymore

    # Vote loss
    #vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    #objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    #num_proposal = objectness_label.shape[1]
    #total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    #data_dict['objectness_label'] = objectness_label
    #data_dict['objectness_mask'] = objectness_mask
    #data_dict['object_assignment'] = object_assignment
    #data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    #data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

    # Box loss and sem cls loss
    #center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    #box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

    #if detection:
    #    data_dict['vote_loss'] = vote_loss
    #    data_dict['objectness_loss'] = objectness_loss
    #    data_dict['center_loss'] = center_loss
    #    data_dict['heading_cls_loss'] = heading_cls_loss
    #    data_dict['heading_reg_loss'] = heading_reg_loss
    #    data_dict['size_cls_loss'] = size_cls_loss
    #    data_dict['size_reg_loss'] = size_reg_loss
    #    data_dict['sem_cls_loss'] = sem_cls_loss
    #    data_dict['box_loss'] = box_loss
    #else:
    #    data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['center_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
    #    data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if reference and data_dict['epoch'] > cfg.prepare_epochs:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # # Reference loss
        # ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        # data_dict["cluster_labels"] = cluster_labels

        #data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()
        #data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()

        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
        data_dict["cluster_labels"] = torch.zeros(1)[0].cuda()

    if reference and use_lang_classifier and data_dict['epoch'] > cfg.prepare_epochs:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    #loss = data_dict['vote_loss'] + 0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] \
    #    + 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]

    # PointGroup:
    # only the last two losses are needed
    # add the loss calculated during the RefNet.forward by calling pointgroup.model_fn
    # TODO: weight set to 0.8 for now (so that weights sum up to 1)
    # default: 0.1, 0.1, 0.8
    loss = 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"] + 0.8 * data_dict['pg_loss']
    loss *= 10 # amplify

    data_dict['loss'] = loss

    return loss, data_dict
