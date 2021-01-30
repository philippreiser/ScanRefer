import torch
import torch.nn as nn
from util.config import cfg

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, batch_size=1):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + cfg.m, hidden_size, 1), #nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        # self.match = nn.Conv1d(hidden_size, 1, 1)
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # NOTE: batch corresponds to number of scenes per batch!!!

        # unpack outputs from detection branch
        # NOTE: num_proposals is variable!
        features = data_dict['aggregated_vote_features'] # num_proposal, 128
        proposals_idx, proposals_offset = data_dict['proposals_idx'], data_dict['proposals_offset']
        b_offsets = data_dict['offsets']
        
        # PointGroup: 
        # for now no masking substitute
        #objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1
        #TODO: Add comments for feature fill-up with batch_size>1
        proposal_batch_ids = torch.zeros(features.shape[0], dtype=torch.int32).cuda()
        batch_idx = 0
        batch_id = 0

        for i, proposal_offset in enumerate(proposals_offset[:-1]):
            for batch_idx in range(self.batch_size):
                batch_begin_idx, batch_end_idx = b_offsets[batch_idx:batch_idx+2]
                if (batch_begin_idx <= proposals_idx[proposal_offset, 1]) and (
                     proposals_idx[proposal_offset, 1]< batch_end_idx):
                    batch_id = batch_idx
                    break
            proposal_batch_ids[i] = batch_id
        # save for loss_helper
        data_dict['proposal_batch_ids'] = proposal_batch_ids
        batch_features = []
        max_n_proposals = 0
        for batch_idx in range(self.batch_size):
            batch_n_proposals = (proposal_batch_ids==batch_idx).sum()
            if batch_n_proposals > max_n_proposals:
                max_n_proposals = batch_n_proposals
        #max_n_proposals = 1000
        for batch_idx in range(self.batch_size):
            batch_id_mask = torch.nonzero(proposal_batch_ids==batch_idx).cuda()
            batch_id_features = features[batch_id_mask].squeeze(1)
            adapted_features = torch.zeros([max_n_proposals-batch_id_features.shape[0], batch_id_features.shape[1]]).cuda()
            batch_id_features = torch.cat([batch_id_features, adapted_features], dim=0)
            batch_features.append(batch_id_features)
            #if max_n_proposals < batch_id_features.shape[0]:
            #    max_n_proposals = batch_id_features.shape[0]
        
        features = torch.cat(batch_features).reshape(self.batch_size, -1, cfg.m)
        

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, max_n_proposals, 1) # batch_size, num_proposals, lang_size

        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        # fuse
        # num_proposals can vary depending on cluster alg

            
        features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals
        
        # fuse features
        features = self.fuse(features) # batch_size, hidden_size, num_proposals
        
        # PointGroup: 
        # for now no masking substitute
        # mask out invalid proposals
        #objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        #features = features * objectness_masks

        # match
        confidences = self.match(features).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
