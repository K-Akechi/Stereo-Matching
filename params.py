class Params():
    def __init__(self):
        self.batch_size = 1
        self.target_h = 256
        self.target_w = 768

        self.original_h = 540
        self.original_w = 960

        self.channels = 3

        self.max_disparity = 144

        self.enqueue_many_size = 200

        self.is_training = True
        # self.is_training = False
