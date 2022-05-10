import numpy as np
import cv2
import os
from typing import Tuple
import io
import time
import torch
import torchvision
from PIL import Image, ImageOps
from mobilenet_v2_tsm import MobileNetV2

import IPython

# socket
from socket import *


SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

def get_executor():
    torch_module = MobileNetV2(n_class=27)
    torch_module.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar", map_location='cuda'))

    return torch_module


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


n_still_frame = 0

def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2
    
    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

def show_array(a, fmt='jpeg'):
    """
    Display array in Jupyter cell output using ipython widget.
    params:
        a (np.array): the input array
        fmt='jpeg' (string): the extension type for saving. Performance varies
                             when saving with different extension types.
    """
    f = io.BytesIO() # get byte stream
    Image.fromarray(a).convert('RGB').save(f, fmt) # save array to byte stream
    display(IPython.display.Image(data=f.getvalue())) # display saved array
    

def inference():
    print("Open camera...")
    cap = cv2.VideoCapture(0)
    
    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    
    host = "192.168.1.193" # set to IP address of target computer
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)

    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor = get_executor()
    
    buffer = (
        torch.zeros([1, 3, 56, 56], device='cuda'),
        torch.zeros([1, 4, 28, 28], device='cuda'),
        torch.zeros([1, 4, 28, 28], device='cuda'),
        torch.zeros([1, 8, 14, 14], device='cuda'),
        torch.zeros([1, 8, 14, 14], device='cuda'),
        torch.zeros([1, 8, 14, 14], device='cuda'),
        torch.zeros([1, 12, 14, 14], device='cuda'),
        torch.zeros([1, 12, 14, 14], device='cuda'),
        torch.zeros([1, 20, 7, 7], device='cuda'),
        torch.zeros([1, 20, 7, 7], device='cuda')
    )
    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = 0
    try:
        print("Ready!")
        while True:
            i_frame += 1
            _, img = cap.read()  # (480, 640, 3) 0 ~ 255
            if i_frame % 1 == 0:  # option: skip every other frame to obtain a suitable frame rate
                t1 = time.time()
                img_tran = transform([Image.fromarray(img).convert('RGB')])
                input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
                img_nd = input_var.cuda()
                inputs = (img_nd,) + buffer 
                with torch.no_grad():
                    outputs = executor(*inputs)
                feat, buffer = outputs[0], outputs[1:]

                if SOFTMAX_THRES > 0:
                    feat_np = feat.cpu().detach().numpy().reshape(-1)
                    feat_np -= feat_np.max()
                    softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                    print(max(softmax))
                    if max(softmax) > SOFTMAX_THRES:
                        idx_ = np.argmax(feat.cpu().detach().numpy(), axis=1)[0]
                    else:
                        idx_ = idx
                else:
                    idx_ = np.argmax(feat.cpu().detach().numpy(), axis=1)[0]

                if HISTORY_LOGIT:
                    history_logit.append(feat.cpu().detach().numpy())
                    history_logit = history_logit[-12:]
                    avg_logit = sum(history_logit)
                    idx_ = np.argmax(avg_logit, axis=1)[0]
                idx, history = process_output(idx_, history)

                t2 = time.time()
                print(f"{i_frame} {catigories[idx]}")


                current_time = t2 - t1
                history_timing.append(current_time)
                
            UDPSock.sendto(catigories[idx].encode(), addr)
#             show_array(img)
            IPython.display.clear_output(wait=True)

    except KeyboardInterrupt:
        print("Video feed stopped.")
        cap.release()
        UDPSock.close()
        os._exit(0)
