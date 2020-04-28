class BasicBrain:
    def __init__(self, input_dim, output_dim, method):
        self.one = 1
        self._method = method
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method: str):
        self._method = method
