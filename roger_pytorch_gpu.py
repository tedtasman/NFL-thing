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

# Define the model
class RogerModel(nn.Module):

    def __init__(self):

        super(RogerModel, self).__init__()
        self.fc1 = nn.Linear(84, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 7)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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
        winner = game["Winner/tie"]
        loser = game["Loser/tie"]
        at = game["At"]
        week = game["Week"]
        neutral = [0]
        home_team = winner
        away_team = loser
        home_winner = 1

        # Determine the home and away teams
        if at == '@':
            home_team = loser
            away_team = winner
            home_winner = 0
        elif at == 'N':
            neutral = [1]
        
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
        neutral = np.array(neutral, dtype=np.float32)
        week = np.array([week], dtype=np.float32)

        # Concatenate the input data
        input_data = np.concatenate((home_stats_values, away_stats_values, neutral, week))

        # Ensure the data is numeric
        input_data = np.nan_to_num(input_data).astype(np.float32)

        # Extract the target
        target = game.drop(["Winner/tie", "Loser/tie", "At", "Week"], axis=0).values.astype(np.float32)
        home_winner = np.array([home_winner], dtype=np.float32)
        target = np.concatenate((home_winner, target))

        # Append the data
        inputs.append(input_data)
        targets.append(target)
    
    # Convert to tensors
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
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
    winner_yards_diff = 0
    winner_TO_diff = 0
    loser_yards_diff = 0
    loser_TO_diff = 0

    abs_spread_diff = 0
    abs_winner_yards_diff = 0
    abs_winner_TO_diff = 0
    abs_loser_yards_diff = 0
    abs_loser_TO_diff = 0

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
                winner_yards = output[2]
                winner_TO = output[3]
                loser_yards = output[4]
                loser_TO = output[5]

                # Calculate the spread of the target
                target_spread = target[1] - target[0]
                target_winner_yards = target[2]
                target_winner_TO = target[3]
                target_loser_yards = target[4]
                target_loser_TO = target[5]

                # Calculate the difference
                sample_spread_diff = game_spread - target_spread
                sample_winner_yards_diff = winner_yards - target_winner_yards
                sample_winner_TO_diff = winner_TO - target_winner_TO
                sample_loser_yards_diff = loser_yards - target_loser_yards
                sample_loser_TO_diff = loser_TO - target_loser_TO

                # Update the metrics
                spread_diff += sample_spread_diff
                winner_yards_diff += sample_winner_yards_diff
                winner_TO_diff += sample_winner_TO_diff
                loser_yards_diff += sample_loser_yards_diff
                loser_TO_diff += sample_loser_TO_diff

                abs_spread_diff += abs(sample_spread_diff)
                abs_winner_yards_diff += abs(sample_winner_yards_diff)
                abs_winner_TO_diff += abs(sample_winner_TO_diff)
                abs_loser_yards_diff += abs(sample_loser_yards_diff)
                abs_loser_TO_diff += abs(sample_loser_TO_diff)

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
    winner_yards_diff /= count
    winner_TO_diff /= count
    loser_yards_diff /= count
    loser_TO_diff /= count

    abs_spread_diff /= count
    abs_winner_yards_diff /= count
    abs_winner_TO_diff /= count
    abs_loser_yards_diff /= count
    abs_loser_TO_diff /= count

    # Calculate the accuracy
    accuracy = correct / count

    header = ["Test Loss", "Accuracy", "Spread Diff", "Winner Yards Diff", "Winner TO Diff", "Loser Yards Diff", "Loser TO Diff", "Abs Spread Diff", "Abs Winner Yards Diff", "Abs Winner TO Diff", "Abs Loser Yards Diff", "Abs Loser TO Diff"]
    results = [test_loss, accuracy, spread_diff, winner_yards_diff, winner_TO_diff, loser_yards_diff, loser_TO_diff, abs_spread_diff, abs_winner_yards_diff, abs_winner_TO_diff, abs_loser_yards_diff, abs_loser_TO_diff]

    return dict(zip(header, results))








def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RogerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    train_games, test_games = load_games("nfl_data.csv")
    stats = pd.read_csv("all_stats.csv")

    train_inputs, train_targets = prepare_data(train_games, stats)
    test_inputs, test_targets = prepare_data(test_games, stats)

    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    epochs = 200
    for epoch in range(epochs):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} completed\t Loss: {loss}")

    # Test the model
    results = test_model(model, test_loader, criterion, device)

    for key, value in results.items():
        print(f"{key}: {value}")

    # Save the model
    torch.save(model.state_dict(), "nfl_model.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")
    print("Model and optimizer state saved.")

if __name__ == "__main__":
    main()