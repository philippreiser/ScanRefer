import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
#from models.backbone_module import Pointnet2Backbone
#from models.voting_module import VotingModule
#from models.proposal_module import ProposalModule

# new segemnation pipeline - PointGroup
from models.pointgroup import PointGroup
from util.config import cfg

from models.lang_module import LangModule
from models.match_module import MatchModule

# for cfg.prepare_epochs
from util.config import cfg

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=256, batch_size=1, fix_match_module_input=False,
    model_fn=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference
        self.fix_match_module_input = fix_match_module_input
        self.batch_size = batch_size
        self.model_fn = model_fn


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        #self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        #self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        #self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        ### replace segmentation pipeline with PointGroup ###
        # --config config/pointgroup_run1_scannet.yaml needs to be done as well when calling the train function 
        # TODO: passed in cfg have to be either forwarded from train.py or statically imported into this doc
        self.pointgroup = PointGroup(cfg)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size,
                                        batch_size=batch_size)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        #data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        #xyz = data_dict["fp2_xyz"]
        #features = data_dict["fp2_features"]
        #data_dict["seed_inds"] = data_dict["fp2_inds"]
        #data_dict["seed_xyz"] = xyz
        #data_dict["seed_features"] = features
        
        #xyz, features = self.vgen(xyz, features)
        #features_norm = torch.norm(features, p=2, dim=1)
        #features = features.div(features_norm.unsqueeze(1))
        #data_dict["vote_xyz"] = xyz
        #data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        #data_dict = self.proposal(xyz, features, data_dict)

        # --------- PointGroup ---------
        # needs of next modules self.lang and self.match
        # self.lang: "lang_feat" are already in data_dict
        # self.match: "aggregated_vote_features", 
        #             "objectness_scores" (not necessary with PG)
        #             (both from self.proposal)
        #             "lang_emb" (comes from self.lang)

        # in PointGroup train.py/train_epoch: (data_dict has to be batch)
        # data_dict needs to conatin: 
        #       - coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        #       - voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        #       - p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        #       - v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda
        #       - coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        #       - feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        #       - labels = batch['labels'].cuda()                        # (N), long, cuda
        #       - instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        #       - instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        #       - instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        #       - batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        #       - spatial_shape = batch['spatial_shape']
        #       - EXTRA: epoch = batch['epoch']

        #model_fn = model_fn_decorator()
        loss, preds, visual_dict, _ = self.model_fn(data_dict, self.pointgroup, data_dict['epoch'], self.batch_size) # data_dict['epoch'] = 129
        print("semantic_loss: ", visual_dict["semantic_loss"].item())
        print("offset_norm_loss: ", visual_dict["offset_norm_loss"].item())
        print("offset_dir_loss: ", visual_dict["offset_dir_loss"].item())
        print("score_loss: ", visual_dict["score_loss"].item())
        print("--------")
        print("Loss: ", loss.item())
        print("------------------------------------------------")
        # forward loss 
        data_dict['pg_loss'] = loss
        data_dict['pg_end'] = time.time()
        if not self.no_reference and (data_dict['epoch'] > cfg.prepare_epochs):
            # bridge important data for next computations
            data_dict['object_id'] = data_dict['object_id'].cuda()
            data_dict['instance_labels'] = data_dict['instance_labels'].cuda()
            data_dict['semantic_preds'] = preds['semantic_preds'].cuda()
            data_dict['proposals_idx'] = preds['proposals'][0].cuda()
            data_dict['proposals_offset'] = preds['proposals'][1].cuda()
            data_dict['score_feats'] = preds['score_feats'].cuda()
            data_dict['aggregated_vote_features'] = preds['score_feats'].cuda()
            if self.fix_match_module_input:
                if data_dict['_global_iter']==0:
                    data_dict['0semantic_preds'] = preds['semantic_preds']
                    data_dict['0proposals_idx'] = preds['proposals'][0]
                    data_dict['0proposals_offset'] = preds['proposals'][1]
                    data_dict['0score_feats'] = preds['score_feats']
                    data_dict['0aggregated_vote_features'] = preds['score_feats']
                else:
                    data_dict['semantic_preds'] = data_dict['0semantic_preds'].cuda()
                    data_dict['proposals_idx'] = data_dict['0proposals_idx'].cuda()
                    data_dict['proposals_offset'] = data_dict['0proposals_offset'].cuda()
                    data_dict['score_feats'] = data_dict['0score_feats'].cuda()
                    data_dict['aggregated_vote_features'] = data_dict['0aggregated_vote_features'].cuda()

            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)
            data_dict['match_end'] = time.time()

        return data_dict
