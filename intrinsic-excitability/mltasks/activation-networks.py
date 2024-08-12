import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class LearnableActivation(nn.Module):
    def __init__(self):
        super(LearnableActivation, self).__init__()
        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.beta1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.1))
        self.beta3 = nn.Parameter(torch.tensor(0.5))

    def custom_activation1(self, x):
        return self.alpha1 * torch.tanh(self.beta1 * x)

    def custom_activation2(self, x):
        return self.alpha2 * F.relu(self.beta2 * x)

    def custom_activation3(self, x):
        return self.alpha3 * torch.sigmoid(self.beta3 * x)

    def mixed_activation(self, x):
        return self.custom_activation1(x) + self.custom_activation2(x) + self.custom_activation3(x)

    def forward(self, x):
        return self.mixed_activation(x)


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
        # Feedforward layers
        self.fc1 = nn.Linear(784, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)

        # Initialize weights and biases
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

        # Turn off gradients for weights, keep them on for biases
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = True

        # Initialize learnable parameters
        self.activation1 = LearnableActivation()
        self.activation2 = LearnableActivation()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch, log_interval=1, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()
                )
            )
            if dry_run:
                break
        yield loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    return test_loss, correct / len(test_loader.dataset)


def dataset(batch_size, test_batch_size, use_cuda):
    data_path = r"/Users/landauland/Documents/ML-Datasets"

    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST(data_path, train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


if __name__ == "__main__":
    batch_size = 128
    test_batch_size = 128
    lr = 1.0
    gamma = 0.7
    epochs = 2
    log_interval = 100

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    train_loader, test_loader = dataset(batch_size, test_batch_size, use_cuda)

    model_to_use = Activation
    model = model_to_use().to(device)
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        c_train_loss = train(model, device, train_loader, optimizer, epoch, log_interval=log_interval)
        c_test_loss, c_test_accuracy = test(model, device, test_loader)
        scheduler.step()

        train_losses.extend(c_train_loss)
        test_losses.append(c_test_loss)
        test_accuracies.append(c_test_accuracy)
