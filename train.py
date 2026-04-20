import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import CNNPrunableNet, PrunableLinear
from utils import sparsity_loss, calculate_sparsity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation (IMPORTANT BOOST)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

def train_model(lambda_val):
    model = CNNPrunableNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(15):
        model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            ce_loss = F.cross_entropy(output, target)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return 100 * correct / total


def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig("gate_distribution.png")
    plt.show()


if __name__ == "__main__":
    lambdas = [1e-5, 1e-4, 5e-4]
    results = []

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")
        model = train_model(lam)

        acc = evaluate(model)
        sparsity = calculate_sparsity(model)

        print(f"Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

    plot_gates(model)

    print("\nFinal Results:")
    for r in results:
        print(r)