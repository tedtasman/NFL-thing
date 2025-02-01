'''
roger.py
version 1.0.0

Theodore Tasman
2025-01-31

This is the ML model for predicting the outcome of NFL games.
Named after the great Roger Goodell.
'''
print("Loading imports...")
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(84,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(6, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load the games
def load_games(filename):
    games = pd.read_csv(filename)
    train_games, test_games = train_test_split(games, test_size=0.2, random_state=42)
    print(train_games.shape)
    print(test_games.shape)
    return train_games, test_games

# Load the stats
def load_stats(filename):
    stats = pd.read_csv(filename)
    print(stats.shape)
    return stats

def train_model(model, train_games, stats, epochs):

    for epoch in range(epochs):
        
        game_count = train_games.shape[0]
        count = 0

        for _, game in train_games.iterrows():

            winner = game["Winner/tie"]
            loser = game["Loser/tie"]
            at = game["At"]
            week = game["Week"]
            neutral = [0]
            home_team = winner
            away_team = loser

            if at == '@':
                home_team = loser
                away_team = winner
            elif at == 'N':
                neutral = [1]

            
            home_stats = stats[stats["team_id"] == home_team]
            away_stats = stats[stats["team_id"] == away_team]

            if home_stats.empty or away_stats.empty:
                continue

            home_stats = home_stats.drop(["team_id"], axis=1)
            away_stats = away_stats.drop(["team_id"], axis=1)

            # Ensure all data is numeric before concatenation
            home_stats_values = home_stats.values.flatten().astype(np.float32)
            away_stats_values = away_stats.values.flatten().astype(np.float32)
            neutral = np.array(neutral, dtype=np.float32)
            week = np.array([week], dtype=np.float32)

            input_data = np.concatenate((home_stats_values, away_stats_values, neutral, week))
            input_data = input_data.reshape(1, 84)
            input_data = np.nan_to_num(input_data)
            input_data = input_data.astype(np.float32)

            target = game.drop(["Winner/tie", "Loser/tie", "At", "Week"], axis=0).values.reshape(1, 6)
            
            target = target.astype(np.float32)

            if count % 10 == 0:
                print(f"Training step {count}/{game_count}...\r", end="")

            train_step(model, input_data, target)

            count += 1

        print(f"Epoch {epoch + 1}/{epochs} complete.")

@tf.function
def train_step(model, input_data, target):
    model.train_on_batch(input_data, target)

def test_model(model, test_games, stats):

    correct = 0
    total = 0

    for index, game in test_games.iterrows():

        winner = game["Winner/tie"]
        loser = game["Loser/tie"]
        at = game["At"]
        week = game["Week"]
        neutral = 0
        home_team = winner
        away_team = loser

        if at == '@':
            home_team = loser
            away_team = winner
        elif at == 'N':
            neutral = 1

        
        home_stats = stats[stats["team_id"] == home_team]
        away_stats = stats[stats["team_id"] == away_team]

        if home_stats.empty or away_stats.empty:
            continue

        home_stats = home_stats.drop(["team_id"], axis=1)
        away_stats = away_stats.drop(["team_id"], axis=1)

        input_data = np.concatenate([home_stats.values, away_stats.values, neutral, week], axis=1)
        input_data = input_data.reshape(1, 82)
        target = game.drop(["Winner/tie", "Loser/tie", "At", "Week"], axis=1).values

        prediction = model.predict(input_data)

        if prediction[0] - prediction[1] > 0 and target[0] - target[1] > 0:
            correct += 1
        elif prediction[0] - prediction[1] < 0 and target[0] - target[1] < 0:
            correct += 1
        total += 1

    return correct / total
            

def save_model(model, filename):
    model.save(filename)





def main():
    print("Creating model...")
    model = create_model()

    print("Loading data...")
    train_games, test_games = load_games("nfl_data.csv")
    stats = load_stats("all_stats.csv")

    print("Training model...")
    train_model(model, train_games, stats, 100)

    print("Testing model...")
    accuracy = test_model(model, test_games, stats)
    print(f"Accuracy: {accuracy}")

    save_model(model, "nfl_model.h5")

if __name__ == "__main__":
    main()