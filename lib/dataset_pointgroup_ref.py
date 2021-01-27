import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
import glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from lib.pointgroup_ops.functions import pointgroup_ops

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ScannetReferencePointGroupDataset(Dataset):

    def __init__(self, scanrefer, scanrefer_all_scene,
        split="train",
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        data_augmentation=False,
        scale=50,
        full_scale=[128, 512],
        max_npoint=250000,
        batch_size=1,
        mode=4,
        train_workers=4):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height # TODO
        self.use_normal = use_normal # TODO
        self.use_multiview = use_multiview # TODO
        self.augment = augment # TODO
        self.data_augmentation = data_augmentation
        self.scale = scale
        self.full_scale = full_scale
        self.max_npoint = max_npoint
        self.batch_size = batch_size
        self.mode = mode
        self.train_workers = train_workers

        # load data
        self._load_data()
        # remap data
        self._prepare_data()
        self.multiview_data = {}

    def __len__(self):
        return len(self.scanrefer)

    def trainLoader(self):
        train_set = list(range(len(self.scanrefer)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)

    def trainMerge_old(self, idx):
        start = time.time()
        idx = idx[0] #TODO: replace with for loop for multiple batches
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"])
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        # if self.use_height:
        #     floor_height = np.percentile(point_cloud[:,2],0.99)
        #     height = point_cloud[:,2] - floor_height
        #     point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        xyz_origin = point_cloud
        label, instance_label = semantic_labels, instance_labels
        rgb = pcl_color
        ### jitter / flip x / rotation
        # TODO: Data augmentation
        xyz_middle = self.dataAugment(xyz_origin, True, True, True)

        ### scale
        xyz = xyz_middle * self.scale

        ### elastic
        xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
        xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        ### offset
        xyz -= xyz.min(0)

        ### crop
        xyz, valid_idxs = self.crop(xyz)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        label = label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        ### get instance information
        inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
        inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_infos = torch.from_numpy(inst_info).to(torch.float32) # float (N, 9) (meanxyz, minxyz, maxxyz)
        inst_pointnum = inst_infos["instance_pointnum"] # (nInst), list
        instance_pointnum = torch.tensor(inst_pointnum, dtype=torch.int)  # int (total_nInst)
        batch_offsets = [0 + xyz.shape[0]]
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        total_inst_num = inst_num

        locs = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()], 1)
        locs_float = torch.from_numpy(xyz_middle).to(torch.float32)
        feats = torch.from_numpy(rgb) + torch.randn(3) * 0.1
        labels = torch.from_numpy(label.astype(np.int64)).long()   # long (N)
        instance_labels = torch.from_numpy(instance_labels.astype(np.int64)).long()   # long (N)
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)
        lang_feat = torch.from_numpy(lang_feat.astype(np.float32))[None, :, :]
        lang_len = torch.from_numpy(np.array(lang_len).astype(np.int64))[None]
        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
        object_cat = torch.from_numpy(np.array(object_cat).astype(np.int64))[None]
        load_time = torch.from_numpy(np.array(time.time() - start))[None]
        return {'locs': locs, 'locs_float': locs_float, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'feats': feats, 'labels': labels, 'instance_labels': instance_labels, 'spatial_shape': spatial_shape,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'offsets': batch_offsets, 
                "lang_feat":lang_feat, "lang_len": lang_len,
                'object_id': object_id, "load_time": load_time, "object_cat": object_cat
                }


    def trainMerge(self, id):
        start = time.time()

        ## PointGroup Input ##
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0

        ## ScanRefer Input ##
        lang_feats = []
        lang_lens = []
        object_cats = []
        object_ids = []

        for i, idx in enumerate(id):
            scene_id = self.scanrefer[idx]["scene_id"]
            object_id = int(self.scanrefer[idx]["object_id"])
            object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
            ann_id = self.scanrefer[idx]["ann_id"]
            
            # get language features
            lang_feat = self.lang[scene_id][str(object_id)][ann_id]
            lang_len = len(self.scanrefer[idx]["token"])
            lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN else CONF.TRAIN.MAX_DES_LEN

            # get pc
            mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
            instance_label = self.scene_data[scene_id]["instance_labels"]
            semantic_label = self.scene_data[scene_id]["semantic_labels"]
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

            if not self.use_color:
                point_cloud = mesh_vertices[:,0:3] # do not use color for now
                pcl_color = mesh_vertices[:,3:6]
            else:
                point_cloud = mesh_vertices[:,0:6] 
                point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
                pcl_color = point_cloud[:,3:6]
            
            if self.use_normal:
                normals = mesh_vertices[:,6:9]
                point_cloud = np.concatenate([point_cloud, normals],1)

            if self.use_multiview:
                # load multiview database
                pid = mp.current_process().pid
                if pid not in self.multiview_data:
                    self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

                multiview = self.multiview_data[pid][scene_id]
                point_cloud = np.concatenate([point_cloud, multiview],1)

            # if self.use_height:
            #     floor_height = np.percentile(point_cloud[:,2],0.99)
            #     height = point_cloud[:,2] - floor_height
            #     point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
            point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
            instance_label = instance_label[choices]
            semantic_label = semantic_label[choices]
            pcl_color = pcl_color[choices]

            xyz_origin = point_cloud
            label = semantic_label
            rgb = pcl_color
            if self.data_augmentation:
                # TODO: Data augmentation
                ### jitter / flip x / rotation
                xyz_middle = self.dataAugment(xyz_origin, True, True, True)

                ### scale
                xyz = xyz_middle * self.scale

                ### elastic
                xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
                xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

                ### offset
                xyz -= xyz.min(0)

                ### crop
                xyz, valid_idxs = self.crop(xyz)

                xyz_middle = xyz_middle[valid_idxs]
                xyz = xyz[valid_idxs]
                rgb = rgb[valid_idxs]
                label = label[valid_idxs]
                instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
            
            ### get instance information
            if not self.data_augmentation:
                xyz_middle  = self.dataAugment(xyz_origin, False, False, False)
                ### scale
                xyz = xyz_middle * self.scale
                ### offset
                xyz -= xyz.min(0)
                ### crop
                xyz, valid_idxs = self.crop(xyz)

                xyz_middle = xyz_middle[valid_idxs]
                xyz = xyz[valid_idxs]
                rgb = rgb[valid_idxs]
                label = label[valid_idxs]
                instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"] # (nInst), list
            
            instance_label[np.where(instance_label != -100)] += total_inst_num
            object_id += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch (PG)
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label.astype(np.int64)))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

            ### merge the scene to the batch (SR)
            # TODO: Check shape of lang_feats, lang_len, object_cats
            lang_feats.append(torch.from_numpy(lang_feat.astype(np.float32)))
            lang_lens.append(torch.from_numpy(np.array(lang_len).astype(np.int64)))
            object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
            object_cats.append(torch.from_numpy(np.array(object_cat).astype(np.int64)))
            object_ids.append(object_id)

        ### merge all the scenes in the batchd (PG)
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)
        
        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)
        
        ### SC
        lang_feats = torch.cat(lang_feats, 0).reshape((self.batch_size, 126, 300)) # float (B, 126, 300)
        lang_lens = torch.tensor(lang_lens, dtype=torch.int64) # float (B, 1)
        object_cats = torch.tensor(object_cats) # float (B, )
        load_time = torch.from_numpy(np.array(time.time() - start))[None]
        
        return {'locs': locs, 'locs_float': locs_float, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'feats': feats, 'labels': labels, 'instance_labels': instance_labels, 'spatial_shape': spatial_shape,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'offsets': batch_offsets, 
                "lang_feat":lang_feats, "lang_len": lang_lens,
                'object_id': object_ids, "load_time": load_time, "object_cat": object_cats
                }


    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label


    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        print("loading data...")
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()


    def _prepare_data(self):
        """ 
        Prepare data for PointGroup module
        """
        # Map relevant classes to {0,1,...,19}, and ignored classes to -100
        remapper = np.ones(150) * (-100)
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            remapper[x] = i
        for scene_id in self.scene_list:
            mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
            coords = np.ascontiguousarray(mesh_vertices[:, :3] - mesh_vertices[:, :3].mean(0))
            colors = np.ascontiguousarray(mesh_vertices[:, 3:6]) / 127.5 - 1
            self.scene_data[scene_id]["mesh_vertices"][:, :3] = coords
            self.scene_data[scene_id]["mesh_vertices"][:, 3:6] = colors
            # instance_labels = self.scene_data[scene_id]["instance_labels"]
            semantic_labels = self.scene_data[scene_id]["semantic_labels"]
            self.scene_data[scene_id]["semantic_labels"] = remapper[np.array(semantic_labels)]
            # instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
