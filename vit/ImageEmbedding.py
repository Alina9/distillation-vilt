import torchvision.models as models
import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, dim_embedding, patches_size):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.conv = nn.Conv2d(3, dim_embedding, kernel_size=patches_size, stride=patches_size)

    def freeze(self, layers):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


    def forward(self, image):
        output = self.conv(image)
        n, c, h, w = output.shape
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(n, h * w, c)
        return output
