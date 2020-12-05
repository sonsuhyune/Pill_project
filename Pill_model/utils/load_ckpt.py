import torch
import os
import sys

def load_model(checkpoint_dir, net):

    n_epoch = 0
    checkpoint_path = checkpoint_dir

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return net

