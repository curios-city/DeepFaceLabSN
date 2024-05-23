from core.leras import nn
tf = nn.tf

class LayerBase(nn.Saveable):
    #override
    def build_weights(self):
        pass
    
    #override
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

nn.LayerBase = LayerBase