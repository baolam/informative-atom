from .. import ENCODING_CODE, MANAGEMENT_EXT, AI_CONTENT_EXT
from torch import nn
from json import dump as management_dump
from pickle import dump as content_dump

def save_management(metadata, filename):
    with open(filename, "w", encoding=ENCODING_CODE) as f:
        management_dump(metadata, f)

def save_contentai(entity : nn.Module, filename):
    assert issubclass(type(entity), nn.Module)
    with open(filename, "wb") as f:
        content_dump(entity.state_dict(), f)

def save_management_ext(metadata, _id, folder_name):
    save_management(metadata, f"{folder_name}/management_{_id}{MANAGEMENT_EXT}")

def save_contentai_ext(entity : nn.Module, _id, folder_name):
    save_contentai(entity, f"{folder_name}/content_{_id}{AI_CONTENT_EXT}")