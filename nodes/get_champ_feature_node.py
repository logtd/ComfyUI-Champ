import torch

import comfy.model_management


class GetChampFeatureNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "fea_guider": ("FEA_GUIDER",),
                    "image": ("IMAGE",),
                },
                "optional": {
                }
            }

    RETURN_TYPES = ("FEA_EMBED",)

    FUNCTION = "patch"
    CATEGORY = "champ"

    def patch(self, fea_guider, image):    
        torch_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.intermediate_device()
        

        feat = fea_guider.to(torch_device)(image.permute(0,3,1,2).to(torch_device)).to(offload_device)

        return (feat, )