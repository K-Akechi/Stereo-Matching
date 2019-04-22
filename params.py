class Params():
    def __init__(self):
        self.batch_size = 1
        self.target_h = 512
        self.target_w = 960

        self.original_h = 540
        self.original_w = 960

        self.channels = 3

        self.max_disparity = 192

        self.enqueue_many_size = 200
