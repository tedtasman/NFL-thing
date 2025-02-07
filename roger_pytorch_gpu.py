'''
roger_pytorch_gpu.py
version 1.0.0

Theodore Tasman
2025-02-01

A PyTorch version of the NFL game prediction model.
Created for accelerated GPU training.
'''
print("Loading imports...")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the model
class RogerModel(nn.Module):

    def __init__(self, input_size=84, hidden_size_1=32, hidden_size_2=16, hidden_size_3=0, output_size=6):

        super(RogerModel, self).__init__()
        self.two_hidden = hidden_size_2
        self.three_hidden = hidden_size_3
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2 if self.two_hidden else output_size)

        if self.three_hidden:
            self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
            self.fc4 = nn.Linear(hidden_size_3, output_size)

        elif self.two_hidden:
            self.fc3 = nn.Linear(hidden_size_2, output_size)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        if self.two_hidden:
            x = self.relu(x)
            x = self.fc3(x)
        if self.three_hidden:
            x = self.relu(x)
            x = self.fc4(x)
        return x

# Load the games
def load_games(filename):
    games = pd.read_csv(filename)
    train_games, test_games = train_test_split(games, test_size=0.2, random_state=42)
    print("Training data shape:" + str(train_games.shape))
    print("Testing data shape:" + str(test_games.shape))
    return train_games, test_games

# Load the stats
def load_stats(filename):
    stats = pd.read_csv(filename)
    print("Stats shape:" + str(stats.shape))
    return stats

# Prepare the data
def prepare_data(game, stats):

    inputs = []
    targets = []

    for _, game in game.iterrows():
        
        # Extract the game data
        week = game["week"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        points_home = game["points_home"]
        points_away = game["points_away"]
        yards_home = game["yards_home"]
        turnovers_home = game["turnovers_home"]
        yards_away = game["yards_away"]
        turnovers_away = game["turnovers_away"]
        neutral = game["neutral"]


        
        # Extract the team stats
        home_stats = stats[stats["team_id"] == home_team]
        away_stats = stats[stats["team_id"] == away_team]

        # Skip if stats are missing
        if home_stats.empty or away_stats.empty:
            continue
        
        # Remove the team ID
        home_stats = home_stats.drop(["team_id"], axis=1)
        away_stats = away_stats.drop(["team_id"], axis=1)

        # Convert to numpy arrays
        home_stats_values = home_stats.values.flatten().astype(np.float32)
        away_stats_values = away_stats.values.flatten().astype(np.float32)
        neutral = np.array([neutral], dtype=np.float32)
        week = np.array([week], dtype=np.float32)

        # Concatenate the input data
        input_data = np.concatenate((home_stats_values, away_stats_values, neutral, week))

        # Ensure the data is numeric
        input_data = np.nan_to_num(input_data).astype(np.float32)

        # Extract the target
        target = game.drop(["week", "home_team", "away_team", "neutral"], axis=0).values.astype(np.float32)

        # Append the data
        inputs.append(input_data)
        targets.append(target)
    
    targets = np.array(targets)
    inputs = np.array(inputs)

    # Convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return inputs, targets

def train_model(model, train_loader, optimizer, criterion, device):
    
    model.train()

    for inputs, targets in train_loader:
        
        # Move the data to the device (GPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()
    
    return loss.item()
            

def test_model(model, test_loader, criterion, device):

    model.eval()
    test_loss = 0

    spread_diff = 0
    home_points_diff = 0
    away_points_diff = 0
    home_yards_diff = 0
    home_TO_diff = 0
    away_yards_diff = 0
    away_TO_diff = 0

    abs_spread_diff = 0
    home_abs_points_diff = 0
    away_abs_points_diff = 0
    abs_home_yards_diff = 0
    abs_home_TO_diff = 0
    abs_away_yards_diff = 0
    abs_away_TO_diff = 0

    correct = 0

    count = 0

    with torch.no_grad():

        for inputs, targets in test_loader:
            
            # Move the data to the device (GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            test_loss += criterion(outputs, targets).item()

            for output, target in zip(outputs, targets):
                # Calculate the spread of the prediction
                game_spread = output[1] - output[0]
                home_points = output[0]
                away_points = output[1]
                home_yards = output[2]
                home_turnovers = output[3]
                away_yards = output[4]
                away_turnovers = output[5]

                # Calculate the spread of the target
                target_spread = target[1] - target[0]
                target_home_points = target[0]
                target_away_points = target[1]
                target_home_yards = target[2]
                target_home_TO = target[3]
                target_away_yards = target[4]
                target_away_TO = target[5]

                # Calculate the difference
                sample_spread_diff = game_spread - target_spread
                sample_home_points_diff = home_points - target_home_points
                sample_away_points_diff = away_points - target_away_points
                sample_home_yards_diff = home_yards - target_home_yards
                sample_home_TO_diff = home_turnovers - target_home_TO
                sample_away_yards_diff = away_yards - target_away_yards
                sample_away_TO_diff = away_turnovers - target_away_TO

                # Update the metrics
                spread_diff += sample_spread_diff
                home_points_diff += sample_home_points_diff
                away_points_diff += sample_away_points_diff
                home_yards_diff += sample_home_yards_diff
                home_TO_diff += sample_home_TO_diff
                away_yards_diff += sample_away_yards_diff
                away_TO_diff += sample_away_TO_diff

                abs_spread_diff += abs(sample_spread_diff)
                home_abs_points_diff += abs(sample_home_points_diff)
                away_abs_points_diff += abs(sample_away_points_diff)
                abs_home_yards_diff += abs(sample_home_yards_diff)
                abs_home_TO_diff += abs(sample_home_TO_diff)
                abs_away_yards_diff += abs(sample_away_yards_diff)
                abs_away_TO_diff += abs(sample_away_TO_diff)

                # Calculate the accuracy
                if game_spread > 0 and target_spread > 0:
                    correct += 1
                elif game_spread < 0 and target_spread < 0:
                    correct += 1
                elif game_spread == 0 and target_spread == 0:
                    correct += 1
                
                count += 1
    
    # Calculate the average loss
    test_loss /= count

    # Calculate the average metrics
    spread_diff /= count
    home_points_diff /= count
    away_points_diff /= count
    home_yards_diff /= count
    home_TO_diff /= count
    away_yards_diff /= count
    away_TO_diff /= count

    abs_spread_diff /= count
    home_abs_points_diff /= count
    away_abs_points_diff /= count
    abs_home_yards_diff /= count
    abs_home_TO_diff /= count
    abs_away_yards_diff /= count
    abs_away_TO_diff /= count

    # Calculate the accuracy
    accuracy = correct / count

    header = ["Test Loss", "Accuracy", "Spread Diff", "Home Points Diff","Away Points Diff","Home Yards Diff", "Home TO Diff", "Away Yards Diff", "Away TO Diff", "Abs Spread Diff", "Abs Home Points Diff","Abs Away Points Diff","Abs Home Yards Diff", "Abs Home TO Diff", "Abs Away Yards Diff", "Abs Away TO Diff"]
    results = [test_loss, accuracy, spread_diff,home_points_diff, away_points_diff,home_yards_diff, home_TO_diff, away_yards_diff, away_TO_diff, abs_spread_diff, home_abs_points_diff,away_abs_points_diff, abs_home_yards_diff, abs_home_TO_diff, abs_away_yards_diff, abs_away_TO_diff]

    return dict(zip(header, results))



def train_model_version(device, train_inputs, test_inputs, train_targets, test_targets, epochs=200, learning_rate=0.003, batch_size=64, input_size=84, hidden_size_1=32, hidden_size_2=16, output_size=6):

    model = RogerModel(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        train_model(model, train_loader, optimizer, criterion, device)
    
    results = test_model(model, test_loader, criterion, device)
    return model, results


def prepare_data_spread(game, stats):
    
    inputs = []
    targets = []

    for _, game in game.iterrows():
        
        # Extract the game data
        week = game["week"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        #neutral = game["neutral"]
        spread = game["spread"]

        # Extract the team stats
        home_stats = stats[stats["team_id"] == home_team]
        away_stats = stats[stats["team_id"] == away_team]

        # Skip if stats are missing
        if home_stats.empty or away_stats.empty:
            continue

        # Remove the team ID
        home_stats = home_stats.drop(["team_id"], axis=1)
        away_stats = away_stats.drop(["team_id"], axis=1)

        # Calculate the difference between away and home stats
        diff_stats = away_stats.values - home_stats.values

        # Convert to numpy arrays
        diff_stats = diff_stats.flatten().astype(np.float32)
        #neutral = np.array([neutral], dtype=np.float32)
        week = np.array([week], dtype=np.float32)
        spread = np.array([spread], dtype=np.float32)

        # Concatenate the input data
        input_data = diff_stats
        output_data = spread

        # Ensure the data is numeric
        input_data = np.nan_to_num(input_data).astype(np.float32)
        output_data = np.nan_to_num(output_data).astype(np.float32)

        # Append the data
        inputs.append(input_data)
        targets.append(output_data)

    targets = np.array(targets)
    inputs = np.array(inputs)

    # Convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    return inputs, targets


def test_model_spread(model, test_loader, criterion, device):

    model.eval()
    test_loss = 0

    spread_diffs = []
    square_diffs = []

    with torch.no_grad():

        for inputs, targets in test_loader:
            
            # Move the data to the device (GPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            test_loss += criterion(outputs, targets).item()

            # Calculate the spread of the prediction
            game_spread = outputs
            target_spread = targets

            # Calculate the difference
            sample_spread_diff = game_spread - target_spread

            # Update the metrics
            spread_diffs.append(sample_spread_diff)
            square_diffs.append(sample_spread_diff ** 2)

    # Calculate the average loss
    test_loss /= len(spread_diffs)

    mean_spread_diff = sum(spread_diffs) / len(spread_diffs)
    mean_square_diff = sum(square_diffs) / len(square_diffs)

    variance_spread_diff = sum((diff - mean_spread_diff) ** 2 for diff in spread_diffs) / len(spread_diffs)
    variance_square_diff = sum((diff - mean_square_diff) ** 2 for diff in square_diffs) / len(square_diffs)

    # Convert spread_diffs and square_diffs to numpy arrays for easier manipulation
    spread_diffs = torch.cat(spread_diffs).cpu().numpy()
    square_diffs = torch.cat(square_diffs).cpu().numpy()

    print("Test Loss: " + str(test_loss))
    print("Average Spread Difference: " + str(mean_spread_diff))
    print("Average Square Difference: " + str(mean_square_diff))
    print("Variance of Spread Differences: " + str(variance_spread_diff))
    print("Variance of Square Differences: " + str(variance_square_diff))



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_games, test_games = load_games("nfl_data_spread.csv")
    stats = pd.read_csv("no_recursion_stats.csv")

    train_inputs, train_targets = prepare_data_spread(train_games, stats)
    test_inputs, test_targets = prepare_data_spread(test_games, stats)

    model = RogerModel(input_size=4, hidden_size_1=10, hidden_size_2=0, hidden_size_3=0, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=1024, shuffle=True)
    losses = []
    for epoch in range(400):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}", end="\r")
        losses.append(loss)
    import matplotlib.pyplot as plt

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=1, shuffle=False)
    test_model_spread(model, test_loader, criterion, device)

    # Save the model
    torch.save(model.state_dict(), "roger_model_spread.pth")
    torch.save(optimizer.state_dict(), "optimizer_spread.pth")
    print("Model saved as roger_model_spread.pth")

if __name__ == "__main__":
    main()