from typing import List


class Context(object):
    
    def __init__(self, name, value):
        self.name = name
        self.value = value


class Feedback(object):
    pass


class ImplicitFeedback(Feedback):

    def __init__(self, item_id: int, contexts: List[Context]):
        self.item_id = item_id
        self.contexts = contexts
