
class iCaRL():
    def __init__(self, memory_size):
        super(iCaRL, self).__init__()

        self.memory_size = memory_size

        self.x_memory = []
        self.y_memory = []       