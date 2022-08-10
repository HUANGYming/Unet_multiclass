""" prediction the video or real-time camera

Usage:
    Method1: params.json
    Method2: Terminal

"""

import argparse
import logging
import pathlib
import json

import numpy as np
import torch
from PIL import Image

from Unet.unet_model import UNet
from utils.video_preprocess import BasicDataset
import cv2
import time
from utils.labels_process import labels_process

# set the log information output
logging.getLogger().setLevel(logging.INFO)

"""prediction sigel image

    predict the result mask

    Args:
        net: network
        full_img: image to predict
        device: CUDA/CPU
        scale_factor: image scale
        out_threshold: confidence threshold
    Returns:
        result mask
    """


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = output

        masks_pred = torch.argmax(probs, dim=1)

    return masks_pred


"""set parameters from terminal

    Args:
        -
    Returns:
        paramters input in terminals
    """


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m',
                        required=False,
                        type=pathlib.Path,
                        help="path of the model")
    parser.add_argument('--camera', '-c',
                        required=False,
                        type=pathlib.Path,
                        help='path of the camera/video')
    parser.add_argument('--output', '-o',
                        nargs='+',
                        type=pathlib.Path,
                        help='path of output')
    parser.add_argument('--save', '-a',
                        action='store_true',
                        default=False,
                        help="Save the output masks")
    parser.add_argument('--scale', '-s',
                        type=float,
                        default=1.0,
                        help="Scale factor for the input images")
    return parser.parse_args()


"""convert result tensor to image

    Args:
        mask: pridiction result tensor
    Returns:
        pridiction result image
    """


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


# main function
if __name__ == "__main__":
    # get terminal input
    args = get_args()

    # read parameters from JSON
    PATH_PARAMETERS = './params.json'
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    pre_settings = params['prediction']

    # set Unet
    n_channels = pre_settings["n_channels"]
    n_classes = pre_settings["n_classes"]

    # load nerwork
    net = UNet(n_channels, n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)       # load network to CPU/GPU

    label_process = labels_process(pre_settings)

    if args.camera is None:
        args.camera = pre_settings["video"]

    # load trained model
    if args.model is None:
        args.model = pre_settings["model_path"]

    logging.info("Loading model {}".format(args.model))
    try:
        net.load_state_dict(torch.load(
            args.model, map_location=device))  # model loading
    except FileNotFoundError:
        logging.error("No such Model file!")
        exit()

    logging.info("Model loaded !")

    video_id = str(args.camera)

    cap = cv2.VideoCapture(video_id)

    # the initialization of result saving
    if args.save:
        out_files = pre_settings["./output/"]
        filename = out_files+"{}.avi".format(time.time())
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(
            *'XVID'), 20.0, (580, 750), True)

    model_width = pre_settings["image_width"]
    model_height = pre_settings["image_height"]
    
    # predict picture per frame
    while cap.isOpened():
        start_time = time.time()
        # get fream from video
        ret, frame = cap.read()

        original_image = frame.copy()

        # put text into image
        cv2.putText(original_image, "Original image", (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 255, 100))
        cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original image', 800, 1100)
        cv2.imshow("Original image", original_image)

        # convert format from opencv (BGR) to PIL (RGB)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # resize or not
        if pre_settings["resize"] == "True":
            width, height = img.size
            img = img.resize((model_width, model_height))
            frame = Image.fromarray(frame)
            frame = frame.resize((model_width, model_height))
            frame = np.array(frame)

        # prediction
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        # convert format from tensor to PIL
        mask = mask.cpu().detach().numpy()
        mask = np.squeeze(mask)
        result = label_process.mask2rgb(mask)

        end_time = time.time()

        # image overlay for display
        img_result = cv2.addWeighted(
            frame, 0.4, result, 0.6, 0)    # mix to original image

        # put FPS in the video
        cv2.putText(img_result, "Mask image", (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 255, 100))
        cv2.putText(img_result, "FPS:{:.2f}".format(
            1/(end_time-start_time)), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 255, 100))

        # change the size of window and display the image
        cv2.namedWindow('Mask image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mask image', 800, 1100)
        cv2.imshow("Mask image", img_result)

        # Receive users' input and execute corresponding commands
        key_time = 25
        key = cv2.waitKey(key_time)

        # When users type 'q', the system will exit
        if key == ord('q'):
            logging.info("Interrupted detection.....QUIT....")
            break
        # When users type 'Esc', the system will exit
        elif key == 27:
            logging.info("Interrupted detection.....QUIT....")
            break
        # When users type 's', the result will be saved.
        elif key == ord('s'):
            filename = out_files+"{}.png".format(time.time())
            cv2.imwrite(filename, img_result)
        # result saving
        if args.save:
            writer.write(img_result)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
