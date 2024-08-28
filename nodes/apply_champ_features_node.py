import torch


class InputBlockPatch(torch.nn.Module):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def to(self, device):
        self.features = self.features.to(device).to(torch.float16)
        return self

    def forward(self, h, transformer_options):
        features = self.features
        if transformer_options['block'][1] == 0 and features is not None:
            len_conds = len(transformer_options['cond_or_uncond'])
            ad_params = transformer_options.get('ad_params', {})
            sub_idxs = ad_params.get('sub_idxs', None)
            if sub_idxs is not None:
                features = features[sub_idxs]
            h = h + features.to(h.dtype).repeat(len_conds, 1, 1, 1)
        return h


class ApplyChampFeaturesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "model": ("MODEL",),
                },
                "optional": {
                    "depth_fea": ("FEA_EMBED",),
                    "pose_fea": ("FEA_EMBED",),
                    "semantic_fea": ("FEA_EMBED",),
                    "normal_fea": ("FEA_EMBED",),
                }
            }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "patch"
    CATEGORY = "champ"

    def patch(self, model, depth_fea=None, pose_fea=None, semantic_fea=None, normal_fea=None):    
        features = [depth_fea, pose_fea, semantic_fea, normal_fea]
        features = list(filter(lambda f: f is not None, features))
        features = torch.stack(features).sum(0) if len(features) > 0 else None

        model = model.clone()
        model.set_model_input_block_patch(InputBlockPatch(features))   
        return (model, )