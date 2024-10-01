import torch as ch
from torch import nn
from tqdm import tqdm

# Simple 3 layer FC network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = ch.relu(self.fc1(x))
        x = ch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(X, y, model):
    criterion = nn.MSELoss()
    optimizer = ch.optim.Adam(model.parameters(), lr=0.001)
    it = tqdm(range(20_000))
    for epoch in it:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y) 
        loss.backward()
        optimizer.step()
        it.set_description(f"Epoch {epoch} | Loss: {loss.item():.3f}")

    return model
