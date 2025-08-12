from inspect import signature, Parameter
from typing import Callable, Dict, Any

def get_parameter_through_function(func : Callable) -> Dict[str, Any]:
    """
    Trả về các tham số được truyền vào một hàm
    """
    params = signature(func)
    results = { }

    for name, parameter in params.parameters.items():
        if parameter.default is not Parameter.empty:
            results[name] = parameter.default
    
    return results