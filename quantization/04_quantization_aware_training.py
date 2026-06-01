import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

torch.manual_seed(0)
DEVICE = "cpu"


class VerySimpleNet(nn.Module):
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
    train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return (
        torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True),
        torch.utils.data.DataLoader(test_set,  batch_size=10, shuffle=False),
    )


def train(model, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            criterion(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            correct += (model(x.to(DEVICE)).argmax(dim=1) == y.to(DEVICE)).sum().item()
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

    # QAT: insert fake-quant observers before training
    net.qconfig = torch.ao.quantization.default_qconfig
    net.train()
    torch.ao.quantization.prepare_qat(net, inplace=True)

    train(net, train_loader, epochs=1)

    # Convert to real quantized model
    net.eval()
    torch.ao.quantization.convert(net, inplace=True)

    print(f"Size after QAT: {model_size_kb(net):.1f} KB")
    test(net, test_loader)
