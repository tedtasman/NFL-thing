import torch
import torch.nn as nn
from roger_pytorch_gpu import load_stats, RogerModel
import numpy as np

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the model
model = RogerModel(input_size=41, hidden_size_1=16, hidden_size_2=4, hidden_size_3=0, output_size=1).to(device)

# Load the model state dictionary
model.load_state_dict(torch.load('roger_model_spread.pth'))

# Ensure the model is in evaluation mode
model.eval()

# Load the optimizer state dictionary
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(torch.load('optimizer_spread.pth'))


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
    stats = load_stats("normalized_stats.csv")

    # Game data
    week = np.array([12], dtype=np.float32)
    neutral = np.array([0], dtype=np.float32)
    home_team = 'Philadelphia Eagles_2024'
    away_team = 'Optimal Team'

    home_team, away_team = away_team, home_team
    
    # Extract the team stats
    home_stats = stats[stats["team_id"] == home_team]
    away_stats = stats[stats["team_id"] == away_team]
    
    # Remove the team ID
    home_stats = home_stats.drop(["team_id"], axis=1)
    away_stats = away_stats.drop(["team_id"], axis=1)

    # Calculate the difference between away and home stats
    diff_stats = away_stats.values - home_stats.values

    # Convert to numpy arrays
    diff_stats = diff_stats.flatten().astype(np.float32)

    # Concatenate the input data
    input_data = diff_stats

    # Ensure the data is numeric
    input_data = np.nan_to_num(input_data).astype(np.float32)

    example_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Make a prediction
    result = predict_matchup(example_data)
    print(result)