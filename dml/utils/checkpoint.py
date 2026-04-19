import os

import torch


def save_checkpoint(model, save_dir, epoch, filename=None):
    state = {
        "epoch": epoch,
        "arch": model.__class__.__name__,
        "state_dict": model.state_dict(),
    }
    if filename:
        path = os.path.join(save_dir, filename)
    else:
        path = os.path.join(save_dir, "checkpoint_epoch_%d.pth" % epoch)
    torch.save(state, path)


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
