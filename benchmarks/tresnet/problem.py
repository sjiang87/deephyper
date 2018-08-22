from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        space['epochs'] = (30, 250)
        space['batch_size'] = (16, 512)
        space['learning_rate'] = (1e-4, 1e1)

        self.space = space
        self.params = self.space.keys()
        self.starting_point = [5, 120, 0.5, 0.2] #1.0, 0.5, 0.01, 0, 0, False, 0.9, 1e-08, 0.9, 0.999]

if __name__ == '__main__':
    instance = Problem()
    print(instance.space)
    print(instance.params)
