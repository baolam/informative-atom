from typing import Dict, Tuple, Any

def create_update(origins : Dict[str, Any], pair : Tuple[str, Any]):
    key, value = pair
    if not isinstance(key, str):
        raise ValueError(f"key phải là kiểu dữ liệu str, {key}-{type(key)}")
    origins[key] = value

def delete_key(origins : Dict[str, Any], key : str):
    del origins[key]