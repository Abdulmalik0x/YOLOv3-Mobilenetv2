from __future__ import division

# from models import *
from utils_fold.utils import *
from utils_fold.datasets import *
from utils_fold.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import argparse
import dataset
import model_sum as model
import utils
import torch



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDatasetListFile', type=str, default="./lic_plate_test.part", help='test dataset list file')
    parser.add_argument('--trainDatasetDirectory', type=str, default="./dataset/img", help='test dataset directory')
    parser.add_argument('--trainDatasetLabelDirectory', type=str, default="./dataset/label", help='test dataset directory')
    parser.add_argument('--names', type=str, default="./../expirement_dataset/lic_plate.names", help='test dataset directory')
    parser.add_argument('--imgSquareSize', type=int, default=416, help='Padded squared image size length')
    parser.add_argument('--batchSize', type=int, default=2, help='Batch size')
    parser.add_argument('--noClasses', type=int, default=1, help='number of classes')
    parser.add_argument('--pretrainedParamFile', type=str, default="yoloParam14000.pt", help='Pretrained parameter file')
    parser.add_argument("--iou_thres", type=float, default=0.4, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    args = parser.parse_args()
    return args


def evaluate(model, opt, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    # dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    # )
    testDataSet = dataset.ListDataset(opt)
    dataloader = DataLoader(
        testDataSet,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=testDataSet.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        for batch_i_in, (imgs_in_batch, imgs_in_target) in enumerate(zip(imgs, targets)):
            # Extract labels
            targets = imgs_in_target
            imgs = imgs_in_batch
            # print('labels : ', targets)
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            # print('targets : ', targets)
            imgs = Variable(imgs.type(Tensor), requires_grad=False).unsqueeze(0)
            # print('imgs shape : ', imgs.shape)
            with torch.no_grad():
                # print('Output')
                outputs, _ = model(imgs)
                # print('outputs : ', outputs[0].shape)
                # print('out shape : ', outputs, type(outputs))
                outputs = torch.cat(outputs, dim=1).cpu()
                outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
                # print('outputs : ', outputs)
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print('precision, recall, AP, f1, ap_class : ', precision, recall, AP, f1, ap_class)
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    opt = parse_args()
    print(opt)

    trainDataSet = dataset.ListDataset(opt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = load_classes(opt.names)

    # net = model.objDetNet(opt)
    # net.to(device)
    # net.loadPretrainedParams()

    # Initiate model
    model = model.objDetNet(opt).to(device)
    model.loadPretrainedParams()
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_pretrained_params(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        opt,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batchSize,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
