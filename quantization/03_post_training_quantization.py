import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

torch.manual_seed(0)
DEVICE = "cpu"
MODEL_PATH = "simplenet_ptq.pt"


class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu    = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu    = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28 * 28)
        x = self.quant(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return self.dequant(x)


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set  = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=10, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            out = model(x.to(DEVICE))
            correct += (out.argmax(dim=1) == y.to(DEVICE)).sum().item()
            total   += y.size(0)
    print(f"Accuracy: {correct / total:.3f}")


def model_size_kb(model):
    torch.save(model.state_dict(), "temp_delme.p")
    size = os.path.getsize("temp_delme.p") / 1e3
    os.remove("temp_delme.p")
    return size


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    net = VerySimpleNet().to(DEVICE)
    if Path(MODEL_PATH).exists():
        net.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded model from disk")
    else:
        train(net, train_loader, epochs=1)
        torch.save(net.state_dict(), MODEL_PATH)

    print(f"\nBefore quantization — size: {model_size_kb(net):.1f} KB")
    test(net, test_loader)

    # PTQ: copy weights, insert observers, calibrate, convert
    net_q = QuantizedVerySimpleNet().to(DEVICE)
    net_q.load_state_dict(net.state_dict())
    net_q.eval()
    net_q.qconfig = torch.ao.quantization.default_qconfig
    torch.ao.quantization.prepare(net_q, inplace=True)
    test(net_q, test_loader)  # calibration pass
    torch.ao.quantization.convert(net_q, inplace=True)

    print(f"\nAfter quantization — size: {model_size_kb(net_q):.1f} KB")
    test(net_q, test_loader)
