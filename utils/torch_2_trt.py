import tensorrt as trt
from torch2trt import torch2trt

torch_module = torch_module.eval().cuda()
torch_inputs = (
                torch.zeros([1, 3, 224, 224], device='cuda'),
                torch.zeros([1, 3, 56, 56], device='cuda'),
                torch.zeros([1, 4, 28, 28], device='cuda'),
                torch.zeros([1, 4, 28, 28], device='cuda'),
                torch.zeros([1, 8, 14, 14], device='cuda'),
                torch.zeros([1, 8, 14, 14], device='cuda'),
                torch.zeros([1, 8, 14, 14], device='cuda'),
                torch.zeros([1, 12, 14, 14], device='cuda'),
                torch.zeros([1, 12, 14, 14], device='cuda'),
                torch.zeros([1, 20, 7, 7], device='cuda'),
                torch.zeros([1, 20, 7, 7], device='cuda'))
model_trt = torch2trt(torch_module, [*torch_inputs])

torch.save(model_trt.state_dict(), 'mobilenetV2_trt.pth')

y = torch_module(*torch_inputs)
y_trt = model_trt(*torch_inputs)