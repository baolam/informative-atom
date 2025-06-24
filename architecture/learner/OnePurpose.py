from .Learner import Learner


class OnePurpose(Learner):
    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)