from typing import Tuple, Callable

def assign_loss(loss_component, infor : Tuple[str, Callable]):
    property_name, loss_fn = infor
        
    if property_name not in loss_component:
        raise ValueError(f"{property_name} does not exist!")
    
    loss_component[property_name] = loss_fn
    return loss_component