'''Test out the best num_workers
---------------------------------------------------------------------------------
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, *, prefetch_factor=2,
            persistent_workers=False)
    num_workers (int, optional) : how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
----------------------------------------------------------------------------------
Usage:
    See README.md

'''
from time import time
import multiprocessing as mp
import load_data
from optparse import OptionParser
import torchio as tio

def get_args():
          
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-f', '--folder', dest='folder', 
                      default='', help='data folder (image and mask)')        
    (options, args) = parser.parse_args()
    return options

# test the reading speed of pictures and labels
if __name__ == '__main__':

    val_ratio=0.1

    args = get_args()

    transforms = tio.transforms.Compose([])

    dir_data = f'../data/{args.folder}'

    for num_workers in range(2, mp.cpu_count(), 2):  
        params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': num_workers}
        train_loader, val_loader  = load_data.make_dataloaders(dir_data, val_ratio, params, transforms)

        start = time()
        for epoch in range(0, args.epochs):
            for i, sample_batch in enumerate(train_loader):
                imgs = sample_batch['image'][tio.DATA]
                true_masks = sample_batch['mask'][tio.DATA]
                pass

        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))