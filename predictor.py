import torch
import torch.nn as nn
from roger_pytorch_gpu import load_stats, RogerModel
import numpy as np

# Create an instance of the model
model = RogerModel()

# Load the model state dictionary
model.load_state_dict(torch.load('nfl_model.pth'))

# Ensure the model is in evaluation mode
model.eval()

# Load the optimizer state dictionary
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(torch.load('optimizer.pth'))


def predict_matchup(data):
    """
    Predict the outcome of an NFL matchup given the input data.
    
    Args:
    data (torch.Tensor): The input data for the matchup.
    
    Returns:
    torch.Tensor: The prediction result.
    """
    with torch.no_grad():
        prediction = model(data)
    return prediction

# Example usage
if __name__ == "__main__":

    # Load the stats
    stats = load_stats("all_stats.csv")

    # Game data
    week = np.array([23], dtype=np.float32)
    neutral = np.array([1], dtype=np.float32)
    home_team = 'Kansas City Chiefs_2022'
    away_team = 'Philadelphia Eagles_2022'

    home_team, away_team = away_team, home_team
    
    # Extract the team stats
    home_stats = stats[stats["team_id"] == home_team]
    away_stats = stats[stats["team_id"] == away_team]
    
    # Remove the team ID
    home_stats = home_stats.drop(["team_id"], axis=1)
    away_stats = away_stats.drop(["team_id"], axis=1)

    # Convert to numpy arrays
    home_stats_values = home_stats.values.flatten().astype(np.float32)
    away_stats_values = away_stats.values.flatten().astype(np.float32)

    # Concatenate the input data
    input_data = np.concatenate((home_stats_values, away_stats_values, neutral, week))

    # Ensure the data is numeric
    input_data = np.nan_to_num(input_data).astype(np.float32)

    example_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Make a prediction
    result = predict_matchup(example_data)
    print(result)