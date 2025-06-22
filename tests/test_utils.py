import pytest
from architecture.utils.id_management import generate_id
from architecture.utils.list_operator import ReadOnlyList, find_index
# from architecture.utils.dict_operator import 

def test_generate_id():
    assert isinstance(generate_id(), str)

def test_find_index():
    arr = [1, 2, 3, 4, 5]
    assert find_index(2, arr) == 1
    assert find_index(6, arr) == -1

def test_readonly():
    arr = [1, 3, 7, 9, 11, 13]
    ronly = ReadOnlyList(arr)

    assert len(ronly) == 6
    for i, x in enumerate(ronly):
        assert arr[i] == x
    
    with pytest.raises(TypeError):
        ronly[2] = 5
    
    assert 7 in ronly
    assert 8 not in ronly