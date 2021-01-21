'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler

class SolverDebug():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10, 
    detection=True, reference=True, use_lang_classifier=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose

        for epoch_id in range(epoch):
            try:
                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self, data_dict):
        # optimize
        self.optimizer.zero_grad()
        # Calculate Backward using PG loss
        data_dict['pg_loss'].backward()# self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(
            data_dict=data_dict, 
            config=self.config, 
            detection=self.detection,
            reference=self.reference, 
            use_lang_classifier=self.use_lang_classifier
        )


    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)
        

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

 
        for data_dict in dataloader:
            # Empty cuda cache for pointgroup
            torch.cuda.empty_cache()
            data_dict['lang_feat'] = data_dict['lang_feat'].cuda()
            data_dict["epoch"] = epoch_id
            
            with torch.autograd.set_detect_anomaly(True):
                # forward
                self.optimizer.zero_grad()
                data_dict = self._forward(data_dict)
                # self._compute_loss(data_dict)
                print("phase: ", phase)
                print("Epoch: ", epoch_id," Loss: ", data_dict['pg_loss'])
                # backward
                if phase == "train":
                    self._backward(data_dict)
