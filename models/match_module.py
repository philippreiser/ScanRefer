import torch
import torch.nn as nn

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size

        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
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

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        
        # PointGroup: 
        # for now no masking substitute
        #objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size

        # fuse
        # num_proposals can vary depending on cluster alg
        features = features[None, :, :]
        # TODO: (example for batch_size=2)
        # features          (1,  14, 128)
        # adapted_features  (2, 242, 128)
        # lang_feat         (2, 300, 128)
        # --> figure out in features where batch_id 1 ends and where batch_id 2 starts
        # --> separate cluster features of each batch
        adapted_features = torch.zeros([lang_feat.shape[0], self.num_proposals-features.shape[1], features.shape[2]]).cuda()
        features = torch.cat([features, adapted_features], dim=1)
        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
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
