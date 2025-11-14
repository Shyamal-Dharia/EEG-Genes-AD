import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_model(nn.Module):
    def __init__(self, num_channels, dropout, num_classes):
        super(conv_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=1, padding=(1, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(73728, 512)# 73728
        self.fc2 = nn.Linear(512, num_classes)
        
        # These attributes will store the feature maps and their gradients
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        # This hook saves the gradients of the feature maps from conv3
        self.gradients = grad

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.dropout(x)
        # Only register hook if gradients are being computed.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        # feature_maps = x
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        feature_maps = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature_maps



class conv_model_ab_psd_hfd(nn.Module):
    def __init__(self, num_channels, dropout, num_classes):
        super(conv_model_ab_psd_hfd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=1, padding=(1, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(7680, 512)# 73728
        self.fc2 = nn.Linear(512, num_classes)
        
        # These attributes will store the feature maps and their gradients
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        # This hook saves the gradients of the feature maps from conv3
        self.gradients = grad

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.dropout(x)
        # Only register hook if gradients are being computed.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        # feature_maps = x
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        feature_maps = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature_maps

class conv_model_ab_csv(nn.Module):
    def __init__(self, num_channels, dropout, num_classes):
        super(conv_model_ab_csv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=1, padding=(1, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(58368, 512)# 73728
        self.fc2 = nn.Linear(512, num_classes)
        
        # These attributes will store the feature maps and their gradients
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        # This hook saves the gradients of the feature maps from conv3
        self.gradients = grad

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.dropout(x)
        # Only register hook if gradients are being computed.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        # feature_maps = x
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        feature_maps = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature_maps

class conv_model_ab_csv_and_psd_or_hfd(nn.Module):
    def __init__(self, num_channels, dropout, num_classes):
        super(conv_model_ab_csv_and_psd_or_hfd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=1, padding=(1, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(66048, 512)# 73728
        self.fc2 = nn.Linear(512, num_classes)
        
        # These attributes will store the feature maps and their gradients
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        # This hook saves the gradients of the feature maps from conv3
        self.gradients = grad

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.dropout(x)
        # Only register hook if gradients are being computed.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        # feature_maps = x
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        feature_maps = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature_maps
    
class conv_model_ab_psd_and_hfd(nn.Module):
    def __init__(self, num_channels, dropout, num_classes):
        super(conv_model_ab_psd_and_hfd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(32, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(16, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 1), stride=1, padding=(1, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(15360, 512)# 73728
        self.fc2 = nn.Linear(512, num_classes)
        
        # These attributes will store the feature maps and their gradients
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        # This hook saves the gradients of the feature maps from conv3
        self.gradients = grad

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.silu(self.conv3(x))
        x = self.dropout(x)
        # Only register hook if gradients are being computed.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        # feature_maps = x
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        feature_maps = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feature_maps
