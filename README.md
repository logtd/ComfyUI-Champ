# ComfyUI-Champ (WIP)
ComfyUI nodes to use [Champ](https://github.com/fudan-generative-vision/champ)

## Installation

To use these nodes you need to also have the [Reference UNet](https://github.com/logtd/ComfyUI-RefUNet) framework nodes

### Models
The required models can be found on the [Champ Huggingface](https://huggingface.co/fudan-generative-ai/champ/tree/main)

Download them to the corresponding directory in `ComfyUI/models/`

| Model name | Directory |
|------------|-----------|
| [guidance_encoder_depth](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/guidance_encoder_depth.pth)    | guiders |
| [guidance_encoder_dwpose](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/guidance_encoder_dwpose.pth)    | guiders |
| [guidance_encoder_normal](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/guidance_encoder_normal.pth)    | guiders |
| [guidance_encoder_semantic_map](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/guidance_encoder_semantic_map.pth)    | guiders |
| [denoising_unet](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/denoising_unet.pth) | unet |
| [reference_unet](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/reference_unet.pth) | unet |
| [motion_module](https://huggingface.co/fudan-generative-ai/champ/blob/main/champ/motion_module.pth) | animatediff_models |
| [pytorch_model.bin](https://huggingface.co/fudan-generative-ai/champ/tree/main/image_encoder) | clip_vision |


## Examples
You can find an example workflow in the `example_workflows` directory

## TODO

* SMPL model generation and other alternate preprocessors
* Parametric Shape Alignment (w/o smoothing)
