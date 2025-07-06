from typing import Dict, Any, Tuple

def add_meta(dictionary : Dict[str, Any], key : str, value : Any) -> Dict[str, Any]:
    if key in dictionary:
        raise TypeError(f"{key} existed")
    dictionary[key] = value
    return dictionary

def update_meta(dictionary : Dict[str, Any], key : str, value : Any) -> Dict[str, Any]:
    if key not in dictionary:
        raise TypeError(f"{key} does not exist!")
    dictionary.update({ key : value })
    return dictionary

def pop_meta(dictionary : Dict[str, Any], key : str) -> Tuple[Dict[str, Any], Any]:
    if key not in dictionary:
        raise TypeError(f"{key} does not exist!")
    popped_value = dictionary.pop(key)
    return dictionary, popped_value