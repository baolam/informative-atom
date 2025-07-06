from architecture_v1.units.base import HardUnit, SoftUnit, HybridUnit
from architecture_v1.units.base_behavior import CodingBehavior, NonCodingBehavior
from architecture_v1.relations.base import BaseRelation

class FakeBehavior(CodingBehavior):
    def __init__(self, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return x
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
    
    def recognize(self, *args, **kwargs):
        return super().recognize(*args, **kwargs)


class FakeHard(HardUnit):
    def __init__(self, _id=None, behavior = None, metadata=..., *args, **kwargs):
        super().__init__(_id, behavior, metadata, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        return super().add_meta(key, value, *args, **kwargs)
    
    def pop_meta(self, key, *args, **kwargs):
        return super().pop_meta(key, *args, **kwargs)
    
    def update_meta(self, key, value, *args, **kwargs):
        return super().update_meta(key, value, *args, **kwargs)

class FakeNonCodeBehavior(NonCodingBehavior):
    def __init__(self, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return x
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)
    
    def recognize(self, *args, **kwargs):
        return super().recognize(*args, **kwargs)


class FakeSoft(SoftUnit):
    def __init__(self, _id=None, behavior = None, metadata=..., *args, **kwargs):
        super().__init__(_id, behavior, metadata, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        return super().add_meta(key, value, *args, **kwargs)
    
    def update_meta(self, key, value, *args, **kwargs):
        return super().update_meta(key, value, *args, **kwargs)
    
    def pop_meta(self, key, *args, **kwargs):
        return super().pop_meta(key, *args, **kwargs)

class FakeHybrid(HybridUnit):
    def __init__(self, _id=None, coding_behavior = None, non_coding_behavior = None, metadata = {}, *args, **kwargs):
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        return super().add_meta(key, value, *args, **kwargs)
    
    def update_meta(self, key, value, *args, **kwargs):
        return super().update_meta(key, value, *args, **kwargs)
    
    def pop_meta(self, key, *args, **kwargs):
        return super().pop_meta(key, *args, **kwargs)
    
    def intepret(self, *args, **kwargs):
        return super().intepret(*args, **kwargs)
    

class FakeRelation(BaseRelation):
    def __init__(self, _from, _to, _id = None, metadata = ..., *args, **kwargs):
        super().__init__(_from, _to, _id, metadata, *args, **kwargs)
    
    def add_meta(self, key, value):
        return super().add_meta(key, value)
    
    def pop_meta(self, key):
        return super().pop_meta(key)
    
    def update_meta(self, key, value, *args, **kwargs):
        return super().update_meta(key, value, *args, **kwargs)