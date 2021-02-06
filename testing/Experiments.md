Experiments

Scene_overfitting 
Batch_size = 7
Weighting: 0.6*data_dict["ref_loss"] + 0.4*data_dict["lang_loss"]
----------------------
Scenes:

1. Start_scene_id = 0
2. Start_scene_id = 50
3. Start_scene_id = 100

Experiments:
a) Use pretrained PG (on all scenes)
b) Use GT instance labels + GT proposals
c) (optionally: use pretrained PG (on one scene))