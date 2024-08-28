import os

import torch
from safetensors.torch import load_file

from folder_paths import models_dir

from ..modules.guidance_encoder import GuidanceEncoder


GUIDERS_PATH = os.path.join(models_dir, 'guiders')
os.makedirs(GUIDERS_PATH, exist_ok=True)


class LoadChampGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(GUIDERS_PATH), )}}

    RETURN_TYPES = ("FEA_GUIDER",)
    FUNCTION = "load"

    CATEGORY = "champ"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(GUIDERS_PATH, checkpoint)
        state_dict = torch.load(checkpoint_path)
        guider = GuidanceEncoder(320, 3, [16,32,96,256])
        # guider = GuidanceEncoder(320, 3, [96,256])
        guider.load_state_dict(state_dict, strict=False)
        return (guider,)