import importlib.util
import glob
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

base_dir = os.path.dirname(__file__)

# 1. Khai báo danh sách các thư mục bạn muốn quét
# (Chỉ cần thêm/sửa tên thư mục ở đây, không cần sửa logic bên dưới)
target_dirs = [
    "utils_nodes",
]

# 2. Vòng lặp ngoài: Duyệt qua từng thư mục
for folder in target_dirs:
    folder_path = os.path.join(base_dir, folder)
    
    # Kiểm tra xem thư mục có tồn tại không (tránh lỗi nếu thiếu thư mục)
    if not os.path.isdir(folder_path):
        continue
        
    files = glob.glob(os.path.join(folder_path, "*.py"))
    
    # 3. Vòng lặp trong: Duyệt qua từng file .py trong thư mục đó
    for file in files:
        filename = os.path.basename(file)
        
        if filename == "__init__.py":
            continue
            
        # ⚠️ CỰC KỲ QUAN TRỌNG: Tránh xung đột tên module
        # Nếu image_nodes/blur.py và text_nodes/blur.py cùng tồn tại,
        # ta phải đặt tên module là "image_nodes.blur" và "text_nodes.blur"
        # để Python không ghi đè lẫn nhau trong sys.modules
        module_name = f"{folder}.{os.path.splitext(filename)[0]}"
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            
            # Đăng ký vào sys.modules với tên có chứa folder
            sys.modules[module_name] = module 
            spec.loader.exec_module(module)
            
            # Gộp mapping (Sử dụng getattr cho gọn)
            node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {})
            if node_classes:
                NODE_CLASS_MAPPINGS.update(node_classes)
                
            node_display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
            if node_display_names:
                NODE_DISPLAY_NAME_MAPPINGS.update(node_display_names)
                
        except Exception as e:
            # In ra lỗi kèm tên thư mục và file để dễ debug
            print(f"\n[ERROR] Failed to load node: {folder}/{filename}")
            print(f"Reason: {e}\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]