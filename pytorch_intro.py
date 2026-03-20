# Pytorch 101

import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unsqueeze(1)
y_train = torch.tensor([7.0, 12.0, 17.0, 22.0, 27.0, 32.0]).unsqueeze(1)

x_test = torch.tensor([10.0]).unsqueeze(0)

model = nn.Linear(in_features=1, out_features=1)
sgd = optim.SGD(model.parameters(), lr=0.01)

epoches = 5000
cri = nn.MSELoss()

before_opt = model(x_test)
print(f"before_opt: {before_opt.item()}")

for epoche in range(epoches):
    pred = model(x_train)
    loss = cri(pred, y_train)
    sgd.zero_grad()
    loss.backward()
    sgd.step()

after_opt = model(x_test)
print(f"after_opt: {after_opt.item()}")
