import torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a,b)

x=torch.rand(5,3)
print(x)
y = x.view(15)
z = x.view(-1, 3)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())

# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
