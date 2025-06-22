import pytest
from torch import cuda

def test_gpu_available():
    assert cuda.is_available()