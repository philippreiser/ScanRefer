### ScannetReferencePointGroupDataset keys + shapes
'locs' ([50000, 4])
Long Tensor where locs[i] = [idx, x, y, z]
'locs_float' ([50000, 4]) 

'voxel_locs' ([48853, 4])
'p2v_map': ([50000])

'v2p_map': ([48853, 5])

'feats': ([50000, 3])
RGB features for each point
'labels': ([50000]), 
Class label for each point
'instance_labels': ([50000])
Instance label for each point
'spatial_shape': ([3])

'instance_info':  ([50000, 9]) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
Mean, Min, Max of the Instance of the point
'instance_pointnum': ([68])
Number of instances in scene
'offsets': ([1])
Number of points in batch (makes only sense if multiple scenes are in one batch)

'lang_feat': ([1, 126, 300])
'lang_len': ([1])
'object_id': int
'load_time': ([1])


### Abbreviations
N: total number of points of one or multiple scenes
nClass: class labels {-100, 0, ..., 19}
nInst: number of instances in scene