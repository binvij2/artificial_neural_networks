import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 6)
        self.layer2 = nn.Linear(6, 6)
        self.output = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


# instantiate model
input_dim = X_train.shape[1]       # for example
model = BinaryClassifier(input_dim)

# binary cross-entropy loss
criterion = nn.BCELoss()

# Adam optimizer (default lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():
    preds = model(inputs)
    preds_class = (preds >= 0.5).float()
    acc = (preds_class == targets).float().mean()



# Assuming you have your data as NumPy arrays
# X_train: shape (num_samples, input_dim)
# y_train: shape (num_samples,)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (num_samples, 1)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, optimizer
input_dim = X_train.shape[1]
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item() * inputs.size(0)

        # Compute accuracy
        preds = (outputs >= 0.5).float()
        acc = (preds == targets).float().mean()
        epoch_acc += acc.item() * inputs.size(0)

    # Average loss and accuracy
    epoch_loss /= len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} \
          - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
