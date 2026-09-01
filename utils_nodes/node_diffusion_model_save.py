import os
import torch

import folder_paths
import comfy.sd
import comfy.model_base
import comfy.model_sampling


class DiffusionModelSave:
    SEARCH_ALIASES = [
        "save diffusion model",
        "export diffusion model",
        "diffusion model save",
    ]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "diffusion_models/cleaned_model",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "deainsane/utils"

    def save(self, model, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
            )
        )

        os.makedirs(full_output_folder, exist_ok=True)

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        # Giữ lại một số tensor kỹ thuật cần thiết nếu có.
        # Những key này không phải metadata dạng prompt/workflow.
        extra_keys = {}

        model_sampling = model.get_model_object("model_sampling")

        if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
            if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
                extra_keys["edm_vpred.sigma_max"] = torch.tensor(
                    model_sampling.sigma_max
                ).float()
                extra_keys["edm_vpred.sigma_min"] = torch.tensor(
                    model_sampling.sigma_min
                ).float()

        if model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
            extra_keys["v_pred"] = torch.tensor([])

            if getattr(model_sampling, "zsnr", False):
                extra_keys["ztsnr"] = torch.tensor([])

        # metadata=None: không ghi metadata vào file safetensors.
        comfy.sd.save_checkpoint(
            output_checkpoint,
            model,
            None,  # clip
            None,  # vae
            None,  # clip_vision
            metadata=None,
            extra_keys=extra_keys,
        )

        return {}


NODE_CLASS_MAPPINGS = {
    "DiffusionModelSave": DiffusionModelSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelSave": "Diffusion Model Save",
}