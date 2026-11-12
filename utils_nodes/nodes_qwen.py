import math
import torch

import comfy.utils
import comfy.model_management
import node_helpers


class DeainsaneTextEncodeQwenImage21:
    """Text Encode Qwen Image 2.1 with reference image resizing and output."""

    SEARCH_ALIASES = ["text encode qwen image 2.1", "qwen image encode"]

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            "vae": ("VAE",),
            "resolution": (
                "INT",
                {
                    "default": 1024,
                    "min": 0,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Reference images are resized to about resolution x resolution pixels, at multiples of 'round_to_multiple', preserving aspect ratio. 0 keeps each reference at its own size, rounded to a multiple.",
                },
            ),
            "round_to_multiple": (      
                [8, 16, 32, 64, 128],
                {
                    "default": 64,
                    "tooltip": "Round resized image dimensions to multiples of this value.",
                },
            ),
        }

        # Tương đương với Autogrow từ image_1 đến image_16
        for i in range(1, 17):
            optional_inputs[f"image_{i}"] = (
                "IMAGE",
                {
                    "tooltip": "Reference images, seen by the text encoder and spliced into the sequence as VAE latents."
                },
            )

        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/qwen image"

    def encode(
        self,
        clip,
        prompt,
        negative_prompt,
        vae=None,
        resolution=1024,
        round_to_multiple=64,
        **kwargs,
    ):
        ref_latents = []
        images_vl = []
        resized_images = []

        round_to_multiple = int(round_to_multiple)
        latent_w = latent_h = resolution or 1024

        # Lấy danh sách các image_1..image_16 từ kwargs và sắp xếp theo thứ tự
        image_keys = [k for k in kwargs.keys() if k.startswith("image_")]
        for name in sorted(image_keys, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = kwargs[name]
            if image is None:
                continue

            # Cắt lấy frame đầu tiên và đổi trục để tính toán resize
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = (
                    round(math.sqrt(resolution * resolution * ratio) / round_to_multiple)
                    * round_to_multiple
                )
                height = (
                    round(math.sqrt(resolution * resolution / ratio) / round_to_multiple)
                    * round_to_multiple
                )
            else:
                width = round(samples.shape[3] / round_to_multiple) * round_to_multiple
                height = round(samples.shape[2] / round_to_multiple) * round_to_multiple

            width, height = max(round_to_multiple, width), max(round_to_multiple, height)

            if (width, height) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(
                    samples, width, height, "lanczos", "disabled"
                ).movedim(1, -1)

            if not images_vl:
                latent_w, latent_h = width, height

            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                # Vision tower nhìn ảnh có alpha đè lên nền trắng, trong khi VAE giữ cả 4 kênh
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])

            images_vl.append(rgb)
            resized_images.append(rgb)

            if vae is not None:
                ref_latents.append(vae.encode(s))

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(
                prompt,
                images=images_vl,
                keep_vision=keep_vision,
                prevent_empty_text=True,
            )
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(
                negative_prompt,
                images=images_vl,
                keep_vision=keep_vision,
                prevent_empty_text=True,
            )
        )

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

        latent = torch.zeros(
            [1, 64, latent_h // 16, latent_w // 16],
            device=comfy.model_management.intermediate_device(),
        )

        # Nếu không có ảnh đầu vào nào được nối, trả về 1 ảnh rỗng theo kích thước chuẩn để tránh lỗi
        if not resized_images:
            resized_images = [torch.zeros([1, latent_h, latent_w, 3], dtype=torch.float32)]

        return (positive, negative, {"samples": latent}, resized_images)


NODE_CLASS_MAPPINGS = {
    "DeainsaneTextEncodeQwenImage21": DeainsaneTextEncodeQwenImage21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeainsaneTextEncodeQwenImage21": "Text Encode Qwen Image 2.1 🐺",
}