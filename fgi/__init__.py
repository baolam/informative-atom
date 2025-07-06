VERSION = "1.0.0"

import torch
import pickle
import json
import inspect
import lightning

# Định dạng lưu trữ
MANAGEMENT_EXT = ".json"
AI_CONTENT_EXT = ".pt"
NONAI_CONTENT_EXT = ".json"
ENCODING_CODE = "utf-8"

from .layer import *
from .graph import *
from .units import *
from .problem import *
from .learner import *