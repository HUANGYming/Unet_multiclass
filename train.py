""" multi classification semantic segmentation by Unet
-----------------------------------------------------------------------
Typical example:

    Identify blood vessels and their directions (horizontal or vertical)

    More detail, see README.md
-----------------------------------------------------------------------
Usage:
    train_epoch(): the training procedure for each epoch
    validate_epoch(): save the assessment(IOU) of training
    train_net(): the main training program of training
    get_args(): get the training parameters entered by users
"""

import json
import sys
import os
from optparse import OptionParser
from PIL import Image
import torch
import torchio as tio
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from Unet.unet_model import UNet
from utils.eval import *
from utils.load_data import *
from utils.visualization import *
from utils.labels_process import *

"""epoch training
-----------------------------------------------------------------------------
    Args:
        epoch: the times of epoch
        net: neural network (Unet)
        train_loader: training image and mask 
        criterion: loss function
        optimizer: update hyperparameter
        batch_size: batch size (set by pytorch.DataLoader)
        scheduler: update LR
        lr: learning rate
-----------------------------------------------------------------------------
    Returns:
        no return, but print out training details
    """


def train_epoch(epoch, net, train_loader, criterion, optimizer, batch_size, scheduler, lr, train_settings):

    # sets the module in training mode
    net.train()
    # initialize the accumulated loss
    epoch_loss = 0

    for i, sample_batch in enumerate(train_loader):

        # get data from Subject type
        imgs = sample_batch['image'][tio.DATA]
        true_masks = sample_batch['mask'][tio.DATA]

        # change dimensions from (batchSize, channelNumber, 480, 640, 1) to (batchSize, channelNumber, 480, 640)
        imgs = torch.squeeze(imgs, dim=4)
        # change dimensions from (batchSize, 1, 480, 640, 1) to (batchSize, 480, 640)
        true_masks = torch.squeeze(true_masks, dim=1)
        true_masks = torch.squeeze(true_masks, dim=3)

        true_masks = true_masks.long()

        # change the variables form CPU to GPU
        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        """
        training 
        input images with torch.Size(batchSize, channelNumber, 480, 640)
        output images with torch.Size(batchSize, classNumber, 480, 640) 
        """
        outputs = net(imgs)

        """
        softmax + argmax 
        for a pixel, select target class with the maximum probability
        """
        probs = outputs
        masks_pred = torch.argmax(probs, dim=1)

        """
        CrossEntropyLoss = logsoftmax() + NLLLoss()
        outputs (batchSize, classNumber, 480, 640) 
        true_masks (batchSize, 480, 640)
        """
        loss = criterion(outputs, true_masks)

        # calculate the accumulated loss
        epoch_loss += loss.item()

        # print epoch + iteration + loss + lr
        print(
            f'epoch = {epoch+1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}, lr = {lr}')

        # save to tensorboard
        if i % 100 == 0:
            writer.add_scalar('train_loss_iter',
                              loss.item(),
                              i + len(train_loader) * epoch)
            writer.add_figure('predictions vs. actuals',
                                plot_net_predictions(
                                  imgs, true_masks, masks_pred, batch_size, train_settings),
                              global_step=i + len(train_loader) * epoch)

        # zero out the gradients
        optimizer.zero_grad()

        # back propragation
        loss.backward()

        # updating the Weights and biases
        optimizer.step()

    # when finish one epoch, show some information out
    print(
        f'Epoch finished ! Loss: {epoch_loss/i:.2f}, lr:{scheduler.get_last_lr()}')


"""epoch validate
-------------------------------------------------------------------
    Args:
        epoch: the times of epoch
        train_loader: training image with mask
        val_loader: validation image with mask
        device: CPU/GPU
-------------------------------------------------------------------
    Returns:
        mean IOU
    """


def validate_epoch(epoch, train_loader, val_loader, device, class_number):

    # calculate IOU
    class_iou, mean_iou = eval_net_loader(net, val_loader, class_number, device)
    print('Class IoU:', ' '.join(
        f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}')

    # save to tensorboard
    writer.add_scalar('background_iou', class_iou[0], len(
        train_loader) * (epoch+1))
    writer.add_scalar('violet_iou', class_iou[1], len(
        train_loader) * (epoch+1))
    writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))

    return mean_iou


"""train net
---------------------------------------------------------------------------------------
    Args:
        train_loader: training image with mask (from pytorch.DataLoader)
        val_loader: validation image with mask (from pytorch.DataLoader)
        net: Unet (initialized)
        devise: CPU/GPU
        epochs: the quantity of epoch (from args)
        batch_size: the quantity of batch (from args and set in pytorch.DataLoader)
        lr: learning rate
        save_cp: save the best training net
---------------------------------------------------------------------------------------
    Returns:
        no return
    """


def train_net(train_loader, val_loader, net, class_number, device, train_settings, epochs, batch_size, lr, save_cp):

    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')

    '''
    Stochastic gradient descent (SGD): randomly select a sample as the whole loss
    optional: batch gradient, mini-batch gradient
    '''
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    '''
    Decays the learning rate of each parameter group by gamma every step_size epochs
    lr(n) = gamma*lr(n-1), after one step_size
    '''
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.3*epochs), gamma=0.1)

    '''
    CrossEntropyLoss = logsoftmax() + NLLLoss()
    '''
    loss_weight = train_settings["loss_weight"]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight).cuda())

    # used to save best model
    best_precision = 0

    # start training
    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # training every epoch
        train_epoch(epoch, net, train_loader, criterion, optimizer,
                    batch_size, scheduler, scheduler.get_last_lr(), train_settings)

        # keep the current LR so as to pass to next epoch
        scheduler.step()

        # calulate the IOU of validation data
        precision = validate_epoch(epoch, train_loader, val_loader, device, class_number)

        # save the best model according to mean-IOU
        if save_cp and (precision > best_precision):
            state_dict = net.state_dict()
            if device == "cuda":
                state_dict = net.state_dict()
            torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
            print('Checkpoint {} saved !'.format(epoch + 1))
            best_precision = precision

    writer.close()


"""get user's input of training parameters
----------------------------------------------------------------------------
    Args:
        -e, --epochs: the quantity of epoch 
        -b, --batch-size: the quantity of batch 
        -l, --learning-rate: learning rate
        -c, --load: load existing model
        -f, --folder: data folder (image and mask)
---------------------------------------------------------------------------
    Returns:
        options
    """


def get_args():

    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', 
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', 
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load existing model')
    parser.add_option('-f', '--folder', dest='folder',
                    help='data folder (image and mask)')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    PATH_PARAMETERS = './params.json'
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    train_settings = params['train']

    # set basic model parameter 
    n_channels = train_settings["n_channels"]
    class_number = train_settings["n_classes"]
    val_ratio = train_settings["validation_ratio"]

    # set device of GPU or CPU
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    # initialize file path and confirm training parameter
    if args.folder is None:
        args.folder = train_settings["data_folder"]
    if args.epochs is None:
        args.epochs = train_settings["epoch"]
    if args.batchsize is None:
        args.batchsize = train_settings["batchsize"]
    if args.lr is None:
        args.lr = train_settings["learning_rate"]

    dir_data = f'{args.folder}'
    dir_checkpoint = f'./checkpoints/{args.folder}_b{args.batchsize}/'
    dir_summary = f'./runs/{args.folder}_b{args.batchsize}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 6}
    make_checkpoint_dir(dir_checkpoint)

    # initialize tensorboard
    writer = SummaryWriter(dir_summary)

    # Transforms
    if train_settings["augmentation"] == "True":
        transforms = tio.transforms.Compose([
            # tio.RandomAffine(scales=(0.9, 1.1),degrees=20,isotropic=True,image_interpolation='nearest',),
            # tio.RandomElasticDeformation(num_control_points=8,max_displacement=5)
            # tio.RandomBiasField(coefficients=(0,0.2),order=2),
            # tio.RandomNoise(mean=0,std=(0, 0.001)),
            # tio.RandomFlip(axes=('LR')),
            # tio.RandomGamma(log_gamma=(-0.3, 0.3))
        ])

    # Read and split datasets
    train_loader, val_loader = make_dataloaders(
        val_ratio, params, transforms, train_settings)

    # initialize Unet
    net = UNet(n_channels=n_channels, n_classes=class_number)
    net.to(device)

    # check whether load existing model or not
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # check whether train model in parallel on multiple-GPUs or not
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    try:
        # train
        train_net(train_loader, val_loader, net, class_number, device, train_settings,
                  epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, save_cp = True)

    except KeyboardInterrupt:
        # when training interrupted, save model
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        torch.cuda.empty_cache()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
