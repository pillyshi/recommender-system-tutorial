class Feedback(object):
    pass


class ImplicitFeedback(Feedback):

    def __init__(self, item_id: int, user_id: int):
        self.item_id = item_id
        self.user_id = user_id
