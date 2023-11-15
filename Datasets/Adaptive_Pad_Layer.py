import torch
import torch.nn as nn

class Adaptive_Pad_Layer(nn.Module):
    def __init__(self, target_height, target_width):
        super(Adaptive_Pad_Layer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x):
        current_height, current_width = x.size(1), x.size(2)
        pad_height = self.target_height - current_height
        pad_width = self.target_width - current_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return padded_x
