import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from models.pointgroup import PointGroup
from models.pointgroup import model_fn_decorator
from lib.dataset_pointgroup_ref import ScannetReferencePointGroupDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from lib.pointgroup_ops.functions import pointgroup_ops
from util.log import logger
import util.utils as utils
from lib.loss_helper import get_loss

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment, dataset_class=ScannetReferenceDataset):
    dataset = dataset_class(
        scanrefer=scanrefer[split], 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        batch_size=args.batch_size,
        data_augmentation = args.data_augmentation,
        shuffle_dataloader = args.shuffle_dataloader
    )
    dataset.trainLoader()
    dataloader = dataset.train_data_loader
    return dataset, dataloader

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes=-1, overfit=False, start_scene_id=0, num_samples=-1):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[start_scene_id:start_scene_id+num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        # slice val_scene_list
        val_scene_list = val_scene_list[start_scene_id:start_scene_id+num_scenes]

        # filter data in chosen scenes
        new_scanrefer_val = []
        for data in scanrefer_val:
            if data["scene_id"] in val_scene_list:
                new_scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list
    if overfit:
        new_scanrefer_train = new_scanrefer_train[0:1]
        new_scanrefer_val = new_scanrefer_val[0:1]
    if num_samples>-1:
        new_scanrefer_train = new_scanrefer_train[:num_samples]
        new_scanrefer_val = new_scanrefer_val[:num_samples]
    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model_fn = model_fn_decorator()
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        batch_size=args.batch_size,
        fix_match_module_input=args.fix_match_module_input,
        model_fn=model_fn
    )    
    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained PointGroup...")
        pretrained_model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=True
        )
        if args.use_pretrained[-4:]!=".pth":
            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        else:
            pretrained_path = os.path.join(CONF.PATH.BASE, args.use_pretrained)
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.pointgroup = pretrained_model.pointgroup
        print("loaded pretrained PG model: ", pretrained_path)
        if args.no_pg:
            # freeze PG
            for param in model.pointgroup.parameters():
                param.requires_grad = False
            print("freezed pg params")
    
    # to CUDA
    model = model.cuda()

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.tag: stamp += "_"+args.tag.upper()
    root = os.path.join(CONF.PATH.OUTPUT, stamp)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # scheduler parameters for training solely the detection pipeline TODO:?
    LR_DECAY_STEP = None
    LR_DECAY_RATE = None
    BN_DECAY_STEP = None
    BN_DECAY_RATE = None

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        pg = not args.no_pg,
        fix_pg_input = args.fix_pg_input,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        prepare_epochs=args.prepare_epochs,
        loss_weights=args.loss_weights
    )
    num_params = get_num_params(model)
    return solver, num_params, root


def func(args):
    # dataset
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.overfit, args.start_scene_id, args.num_samples)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True, ScannetReferencePointGroupDataset)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False, ScannetReferencePointGroupDataset)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    solver, num_params, root = get_solver(args, dataloader)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=6) # SET: 8
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50) #SET: 50
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=20)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)#5000
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples [default: -1]")
    parser.add_argument("--start_scene_id", type=int, default=0, help="Start with scene id [default: 0]")
    parser.add_argument("--overfit", action="store_true", help="Train only on one element of the dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--no_pg", action="store_true", help="Do NOT train the pg module.")
    parser.add_argument("--use_proposal_gt", action="store_true", help="Use GT as input to match module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--use_sparseconv", action="store_true", help="Use SparseConv Backbone.")
    parser.add_argument("--prepare_epochs", type=int, help="number of prepare_epochs for PG", default=-1)
    parser.add_argument('--loss_weights', action='append', type=float, help='Loss weights [0.1, 0.1, 0.8]')
    ## Debug Args
    parser.add_argument("--shuffle_dataloader", action="store_true", help="Shuffle Dataloader.")
    parser.add_argument("--data_augmentation", action="store_true", help="Data augmentation in dataset.")
    parser.add_argument("--fix_pg_input", action="store_true", help="Fix input to PG (i.e. dataloader returns always the same element")
    parser.add_argument("--fix_match_module_input", action="store_true", help="Fix input to match module (i.e. no pg forward")
    args = parser.parse_args()
    print("Loss weights: ", args.loss_weights)
    args.tag = "sr_numscenes_{}_scene_{}_numsamples{}_batch_size_{}_shuffle_{}_dataaug_{}_pgfixed_{}_weights_{}".format(args.num_scenes, args.start_scene_id, args.num_samples,
     args.batch_size, args.shuffle_dataloader, args.data_augmentation, args.no_pg, args.loss_weights)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    func(args)