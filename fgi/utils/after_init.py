from abc import ABCMeta

class AfterInitMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # Cài đặt gọi cấu hình phụ trợ sau khi setup hoàn thành
    def after_init(self, *args, **kwargs):
        pass

class AutoAfterInitMeta(ABCMeta):
    """
    Lớp gọi after_init, cấu hình bổ sung ở đây
    """
    def __call__(self, *args, **kwds):
        instance = super().__call__(*args, **kwds)
        if hasattr(instance, "after_init"):
            instance.after_init(*args, **kwds)
        return instance