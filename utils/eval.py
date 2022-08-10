""" compute IoU to evaluate the accuracy of a classification
-----------------------------------------------------------------------
Function structure
    - eval_net_loader
        |- main program
        |- compute_IoU
"""


import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torchio as tio


def compute_IoU(cm):

    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    # extract a diagonal and construct a diagonal
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = true_positives / denominator

    return iou, np.nanmean(iou)


def eval_net_loader(net, val_loader, class_number, device='gpu'):

    # eval mode
    net.eval()
    labels = np.arange(class_number)
    cm = np.zeros((class_number, class_number))

    for i, sample_batch in enumerate(val_loader):

        # get images and masks from Dataloader
        imgs = sample_batch['image'][tio.DATA]
        true_masks = sample_batch['mask'][tio.DATA]

        # change dimensions to meet input requirements of Unet
        imgs = torch.squeeze(imgs, dim=4)
        true_masks = torch.squeeze(true_masks, dim=1)
        true_masks = torch.squeeze(true_masks, dim=3)

        # change device to GPU
        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        # do prediction
        outputs = net(imgs)
        probs = outputs
        preds = torch.argmax(probs, dim=1)

        for j in range(len(true_masks)):

            # convert tensor to numpy
            true = true_masks[j].cpu().detach().numpy().flatten()
            pred = preds[j].cpu().detach().numpy().flatten()

            # compute confusion matrix to evaluate the accuracy of a classification
            cm += confusion_matrix(true, pred, labels=labels)
    # calculate IoU from confusion_matrix
    class_iou, mean_iou = compute_IoU(cm)

    return class_iou, mean_iou



