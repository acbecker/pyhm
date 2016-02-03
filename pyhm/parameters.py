
class Parameter(object):
    def __init__(self, value=0.0, prior=None, parent=None):
        self.prior  = prior
        self.parent = parent

    def __eval__(self, value):
        pass
        
        
               
# Need to figure out how to get a lambda out of this; probably model
