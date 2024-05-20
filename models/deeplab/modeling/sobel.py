import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        # R
        R_img = img[:, 0].unsqueeze(1)
        R_x = self.filter(R_img)
        R_x = torch.mul(R_x, R_x)
        R_x = torch.sum(R_x, dim=1, keepdim=True)
        R_x = torch.sqrt(R_x)

        # G
        G_img = img[:, 1].unsqueeze(1)
        G_x = self.filter(G_img)
        G_x = torch.mul(G_x, G_x)
        G_x = torch.sum(G_x, dim=1, keepdim=True)
        G_x = torch.sqrt(G_x)

        # B
        B_img = img[:, 2].unsqueeze(1)
        B_x = self.filter(B_img)
        B_x = torch.mul(B_x, B_x)
        B_x = torch.sum(B_x, dim=1, keepdim=True)
        B_x = torch.sqrt(B_x)


        x = torch.cat([R_x, G_x, B_x], 1)

        return x