class History:

    def __init__(self, aspects=()):
        self.history = {aspect: [] for aspect in ["generation"] + list(aspects)}

    def record(self, data):
        for key in data:
            self.history[key].append(data[key])

    def __getitem__(self, item):
        return self.history[item]
