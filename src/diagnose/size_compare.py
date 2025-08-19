import torch
import torchvision.models as models
import os

# save PyTorch model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch.save(model.state_dict(), "resnet18.pth")

# compare size
pytorch_size = os.path.getsize("resnet18.pth") / 1024 / 1024
onnx_size = os.path.getsize("resnet18.onnx") / 1024 / 1024

print(f"PyTorch: {pytorch_size:.1f}MB")
print(f"ONNX: {onnx_size:.1f}MB")
print(f"차이: {pytorch_size - onnx_size:.1f}MB")
