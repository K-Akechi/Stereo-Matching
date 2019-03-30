class Params:
    def __init__(self):
        self.batch_size = 16

        self.original_h = 540
        self.original_w = 960

        self.target_h = 256
        self.target_w = 512

        self.max_disparity = 192

        self.enqueue_many_size = 200
