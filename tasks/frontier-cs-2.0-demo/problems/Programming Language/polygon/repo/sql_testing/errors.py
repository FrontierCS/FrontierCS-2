class ParserSyntaxError(SyntaxError):
    def __init__(self, messages: str):
        messages = f'{self.__class__.__name__}: cannot parse `{messages}`'
        super(SyntaxError, self).__init__(messages)


class SMTSolverError(RuntimeError):
    def __init__(self, messages: str = None):
        if messages is None:
            messages = self.__class__.__name__
        else:
            messages = f'{self.__class__.__name__}: `{messages}`'
        super(RuntimeError, self).__init__(messages)
