import numpy as np
import cv2
import os
from typing import Tuple
import io
import tvm
import tvm.relay
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import tvm.contrib.graph_runtime as graph_runtime
from model import MobileNetV2
from onnxsim import simplify
import IPython
from utils.transform import get_transform
from socket import *


SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

def torch2tvm_module(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    torch_module.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(torch_inputs):
            name = "i" + str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape
        buffer = io.BytesIO()
        torch.onnx.export(torch_module, torch_inputs, buffer, input_names=input_names, output_names=["o" + str(i) for i in range(len(torch_inputs))], opset_version=10)
        outs = torch_module(*torch_inputs)
        buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)

        onnx_model, success = simplify(onnx_model)  # this simplifier removes conversion bugs.
        assert success
        
        relay_module, params = tvm.relay.frontend.from_onnx(onnx_model, shape=input_shapes)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target, params=params)
    return graph, tvm_module, params


def torch2executor(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    prefix = f"mobilenet_tsm_tvm_{target}"
    lib_fname = f'{prefix}.tar'
    graph_fname = f'{prefix}.json'
    params_fname = f'{prefix}.params'
    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(params_fname):
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(lib_fname)
        params = tvm.relay.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    else:
        graph, tvm_module, params = torch2tvm_module(torch_module, torch_inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(tvm.relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    return executor, ctx


def get_executor(use_gpu=True):
    torch_module = MobileNetV2(n_class=27)    
    torch_module.load_state_dict(torch.load("./models/mobilenetv2_jester_online.pth.tar"))
    torch_inputs = (torch.rand(1, 3, 224, 224),
                    torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7]))
    if use_gpu:
        target = 'cuda'
    else:
        target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'
    return torch2executor(torch_module, torch_inputs, target)




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
    
    host = "192.168.1.193" # set to IP address of target computer
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)

    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor, ctx = get_executor()
    buffer = (
        tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
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
            if i_frame % 1 == 0:  # skip every other frame to obtain a suitable frame rate
                t1 = time.time()
                img_tran = transform([Image.fromarray(img).convert('RGB')])
                input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
                img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
                inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
                outputs = executor(inputs)
                feat, buffer = outputs[0], outputs[1:]
                assert isinstance(feat, tvm.nd.NDArray)

                if SOFTMAX_THRES > 0:
                    feat_np = feat.asnumpy().reshape(-1)
                    feat_np -= feat_np.max()
                    softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                    print(max(softmax))
                    if max(softmax) > SOFTMAX_THRES:
                        idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                    else:
                        idx_ = idx
                else:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

                if HISTORY_LOGIT:
                    history_logit.append(feat.asnumpy())
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
