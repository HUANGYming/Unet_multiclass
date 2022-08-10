import matplotlib.pyplot as plt
from utils.labels_process import labels_process
import shutil
from pathlib import Path

def plot_net_predictions(imgs, true_masks, masks_pred, batch_size, train_settings):
    
    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))
    
    for i in range(batch_size):
        
        self_labels_process = labels_process(train_settings)

        img  = imgs[i].cpu().squeeze().detach().numpy()
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = true_masks[i].cpu().detach().numpy()
    
        ax[0,i].imshow(img)
        # print(np.unique(img))
        ax[1,i].imshow(self_labels_process.mask2rgb(mask_pred))
        ax[1,i].set_title('Predicted')
        ax[2,i].imshow(self_labels_process.mask2rgb(mask_true))
        ax[2,i].set_title('Ground truth')
    return fig

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def make_checkpoint_dir(dir_checkpoint):
        
    path = Path(dir_checkpoint)
    # remove folder if it exists
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)




