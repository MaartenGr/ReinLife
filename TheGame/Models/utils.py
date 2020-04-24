class BasicBrain:
    def __init__(self):
        self.one = 1
        self._method = None

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method: str):
        self._method = method
