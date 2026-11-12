import math
import torch

import comfy.utils
import comfy.model_management
import node_helpers
from comfy_api.latest import io


class DeainsaneTextEncodeQwenImage21(io.ComfyNode):
    """Text Encode Qwen Image 2.1 with reference image resizing and output."""

    SEARCH_ALIASES = ["text encode qwen image 2.1", "qwen image encode"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImage21",
            display_name="Text Encode Qwen Image 2.1",
            category="model/conditioning/qwen image",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input(
                    "resolution",
                    default=1024,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Reference images are resized to about resolution x resolution pixels, at multiples of 'round_to_multiple', preserving aspect ratio. 0 keeps each reference at its own size, rounded to a multiple.",
                ),
                io.Combo.Input(
                    "round_to_multiple",
                    options=[8, 16, 32, 64, 128],
                    default=64,
                    tooltip="Round resized image dimensions to multiples of this value.",
                ),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference images, seen by the text encoder and spliced into the sequence as VAE latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Empty latent on the first reference image's size, to match with sampling as any other size shifts the edit.",
                ),
                io.Image.Output(
                    display_name="image",
                    tooltip="First reference image (image_1) resized to target resolution.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        negative_prompt,
        vae=None,
        resolution=1024,
        round_to_multiple=64,
        images: io.Autogrow.Type = None,
    ) -> io.NodeOutput:
        ref_latents = []
        images_vl = []
        first_resized_image = None

        images = images or {}
        round_to_multiple = int(round_to_multiple)
        latent_w = latent_h = resolution or 1024

        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
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

            if first_resized_image is None:
                first_resized_image = rgb

            images_vl.append(rgb)

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

        # Nếu không nối ảnh đầu vào nào, tạo ảnh rỗng theo kích thước latent để tránh lỗi ở node sau
        if first_resized_image is None:
            first_resized_image = torch.zeros(
                [1, latent_h, latent_w, 3], dtype=torch.float32
            )

        return io.NodeOutput(positive, negative, {"samples": latent}, first_resized_image)


NODE_CLASS_MAPPINGS = {
    "DeainsaneTextEncodeQwenImage21": DeainsaneTextEncodeQwenImage21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeainsaneTextEncodeQwenImage21": "🐺 Text Encode Qwen Image 2.1",
}