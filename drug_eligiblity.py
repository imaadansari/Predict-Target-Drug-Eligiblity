import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load and preprocess the data into a Pandas DataFrame
df = pd.read_csv('processed_data.csv')

# Split data into features (X) and labels (y)
X = df.drop('IS_ELIGIBLE', axis=1)  
y = df['IS_ELIGIBLE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = X_train
y = y_train
# Convert data into PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.long)  # Assuming labels are encoded as integers (0 or 1)

# Define ANN architecture
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Specify the dimensions for input, hidden, and output layers
input_size = X.shape[1]
hidden_size = 64
output_size = 1

model = ANN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the ANN model
num_epochs = 50
batch_size = 32

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels.unsqueeze(1).float())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the trained model
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Set model to evaluation mode
model.eval()

# Make predictions on the test data
with torch.no_grad():
    test_outputs = model(X_test)
    predicted_labels = test_outputs.round()

# Calculate classification metrics
accuracy = (predicted_labels == y_test.unsqueeze(1)).sum().item() / len(y_test)
precision = precision_score(y_test,predicted_labels)
f1_score = f1_score(y_test, predicted_labels)
recall = recall_score(y_test,predicted_labels)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# Save the model's state dictionary
torch.save(model.state_dict(), 'model_state_dict_2.pth')

# Save the entire model
torch.save(model, 'model_2.pth')

# # Load the model
# model = torch.load('model.pth')

# # Load the model's state dictionary
# state_dict = torch.load('model.pth')
# model.load_state_dict(state_dict)
