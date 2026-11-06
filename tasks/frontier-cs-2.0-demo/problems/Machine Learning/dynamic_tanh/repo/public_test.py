import torch.nn as nn
from dynamic_tanh import DynamicTanh, convert_ln_to_dyt

assert issubclass(DynamicTanh, nn.Module)
model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
converted = convert_ln_to_dyt(model)
assert converted is not None
assert not any(isinstance(module, nn.LayerNorm) for module in converted.modules())
print("public conversion contract passed")
