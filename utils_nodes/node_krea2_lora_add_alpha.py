import os
import re
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
import safetensors.torch
import folder_paths


class Krea2LoraAddAlphaNode:
    """Thêm key `.alpha` còn thiếu vào LoRA.

    Giá trị alpha bằng rank của tensor LoRA A/down tương ứng, đồng
    thời kế thừa dtype và device của tensor đó.
    """

    _LORA_A_PATTERNS = (
        re.compile(r"^(?P<base>.+)\.lora_A(?:\.[^.]+)?\.weight$"),
        re.compile(r"^(?P<base>.+)\.lora_down\.weight$"),
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_file": (folder_paths.get_filename_list("loras"),),
                "overwrite_existing_alpha": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "convert_lora"
    OUTPUT_NODE = True
    CATEGORY = "deainsane/utils"

    def convert_lora(
        self,
        lora_file: str,
        overwrite_existing_alpha: bool = False,
    ):
        lora_path = folder_paths.get_full_path("loras", lora_file)
        if not lora_path:
            print(f"[Add Alpha] Không tìm thấy LoRA: {lora_file}")
            return ()

        print(f"[Add Alpha] Đang xử lý: {lora_path}")

        try:
            lora_data, metadata = self._load_lora(lora_path)
        except Exception as exc:
            print(f"[Add Alpha] Đọc file thất bại: {exc}")
            return ()

        if not lora_data:
            print("[Add Alpha] File không chứa tensor hợp lệ.")
            return ()

        converted_data, added, overwritten, skipped = self._add_alpha_keys(
            lora_data,
            overwrite_existing_alpha,
        )

        output_dir = os.path.join(
            folder_paths.get_folder_paths("loras")[0],
            "converted_loras",
        )
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(lora_file))[0]
        output_path = os.path.join(
            output_dir,
            f"{base_name}_with_alpha.safetensors",
        )
        temp_path = f"{output_path}.tmp"

        try:
            safetensors.torch.save_file(
                converted_data,
                temp_path,
                metadata=metadata or None,
            )
            os.replace(temp_path, output_path)
        except Exception as exc:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            print(f"[Add Alpha] Lưu file thất bại: {exc}")
            return ()

        print(
            "[Add Alpha] Hoàn tất: "
            f"thêm {added}, ghi đè {overwritten}, giữ nguyên {skipped} alpha."
        )
        print(f"[Add Alpha] File đã lưu tại: {output_path}")
        return ()

    def _load_lora(
        self,
        lora_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        extension = os.path.splitext(lora_path)[1].lower()

        if extension == ".safetensors":
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(lora_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for key in handle.keys():
                    tensors[key] = handle.get_tensor(key).contiguous()
            return tensors, metadata

        try:
            loaded = torch.load(
                lora_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            loaded = torch.load(lora_path, map_location="cpu")

        if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
            loaded = loaded["state_dict"]

        if not isinstance(loaded, dict):
            raise TypeError("Checkpoint không chứa state_dict dạng dictionary.")

        tensors = {
            str(key): value.detach().cpu().contiguous()
            for key, value in loaded.items()
            if torch.is_tensor(value)
        }
        return tensors, None

    def _add_alpha_keys(
        self,
        lora_data: Dict[str, torch.Tensor],
        overwrite_existing_alpha: bool,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, int]:
        converted_data = dict(lora_data)

        # base_key -> (rank, dtype, device)
        alpha_candidates: Dict[
            str,
            Tuple[int, torch.dtype, torch.device],
        ] = {}

        for key, weight in lora_data.items():
            base_key = self._match_lora_a_key(key)
            if base_key is None:
                continue

            if weight.ndim < 1 or weight.shape[0] <= 0:
                print(
                    "[Add Alpha] Bỏ qua tensor không xác định được rank: "
                    f"{key}"
                )
                continue

            rank = int(weight.shape[0])
            previous = alpha_candidates.get(base_key)

            if previous is not None and previous[0] != rank:
                print(
                    f"[Add Alpha] Cảnh báo: {base_key} có nhiều rank "
                    f"({previous[0]} và {rank}); giữ rank đầu tiên."
                )
                continue

            alpha_candidates[base_key] = (
                rank,
                weight.dtype,
                weight.device,
            )

        added_count = 0
        overwritten_count = 0
        skipped_count = 0

        for base_key, (rank, dtype, device) in alpha_candidates.items():
            alpha_key = f"{base_key}.alpha"
            alpha_exists = alpha_key in converted_data

            if alpha_exists and not overwrite_existing_alpha:
                skipped_count += 1
                continue

            # Kế thừa dtype/device từ LoRA A/down.
            converted_data[alpha_key] = torch.tensor(
                rank,
                dtype=dtype,
                device=device,
            )

            if alpha_exists:
                overwritten_count += 1
            else:
                added_count += 1

        return (
            converted_data,
            added_count,
            overwritten_count,
            skipped_count,
        )

    def _match_lora_a_key(self, key: str) -> Optional[str]:
        for pattern in self._LORA_A_PATTERNS:
            match = pattern.match(key)
            if match:
                return match.group("base")
        return None


NODE_CLASS_MAPPINGS = {
    "Krea2LoraAddAlphaNode": Krea2LoraAddAlphaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2LoraAddAlphaNode": "LoRA Add Alpha",
}
