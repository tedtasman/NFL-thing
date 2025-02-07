import pandas as pd

def create_files(start_year, end_year):
    for year in range(start_year, end_year + 1):
        with open(f"data_files/nfl_data_{year}.csv", "w") as f:
            pass

def join_files(start_year, end_year, file_prefix, output_file):
    # Create the output file
    with open(output_file, "w") as outfile:
        
        # Write the header
        with open(f"{file_prefix}_{start_year}.csv", "r") as infile:
            lines = infile.readlines()
            outfile.write(lines[0])

        # Write the data
        for year in range(start_year, end_year + 1):

            # Read the data from the file
            with open(f"{file_prefix}_{year}.csv", "r") as infile:

                # read lines, skip the header
                lines = infile.readlines()
                for line in lines[1:]:
                    
                    # check for playoff data, encode it as a number
                    parts = line.split(",")

                    parts[0] = parts[0].strip() + f"_{year}"

                    if parts[0] == "WildCard":
                        new_line = "20," + ",".join(parts[1:])
                    elif parts[0] == "Division":
                        new_line = "21," + ",".join(parts[1:])
                    elif parts[0] == "ConfChamp":
                        new_line = "22," + ",".join(parts[1:])
                    elif parts[0] == "SuperBowl":
                        new_line = "23," + ",".join(parts[1:])
                    else:
                        new_line = ','.join(parts)
                    
                    # write the line to the output file
                    outfile.write(new_line)
                
                # write a newline to separate the years
                #outfile.write("\n")

def home_away_layout(input_file, output_file):

    with open(input_file, "r") as infile:
        lines = infile.readlines()
    
    with open(output_file, "w") as outfile:
        
        outfile.write("week,home_team,away_team,points_home,points_away,yards_home,turnovers_home,yards_away,turnovers_away,neutral\n")

        neutral_count = 0

        for line in lines[1:]:
            
            if line == "\n":
                continue

            parts = line.split(",")

            week = parts[0].strip()
            home_team = parts[1].strip()
            at = parts[2].strip()
            away_team = parts[3].strip()
            points_home = parts[4].strip()
            points_away = parts[5].strip()
            yards_home = parts[6].strip()
            turnovers_home = parts[7].strip()
            yards_away = parts[8].strip()
            turnovers_away = parts[9].strip()
            neutral = 0

            # if the winner was away, swap the teams
            # if the game was neutral, swap for even lines
            if at == "@" or (at == "N" and neutral_count % 2 == 1):
                home_team, away_team = away_team, home_team
                points_home, points_away = points_away, points_home
                yards_home, yards_away = yards_away, yards_home
                turnovers_home, turnovers_away = turnovers_away, turnovers_home
            
            # if the game was neutral, increment the counter
            if at == "N":
                neutral_count += 1
                neutral = 1

            # write the new line
            new_line = f"{week},{home_team},{away_team},{points_home},{points_away},{yards_home},{turnovers_home},{yards_away},{turnovers_away},{neutral}\n"
            outfile.write(new_line)

def normalize_column(column):
    """
    Normalize a column of data.
    
    Args:
    column (pd.Series): The column to normalize.
    
    Returns:
    pd.Series: The normalized column.
    """
    min_value = column.min()
    max_value = column.max()
    return (column - min_value) / (max_value - min_value)

def normalize_stats(input_file, output_file):
    """
    Normalize the stats in the input file and write the result to the output file.
    
    Args:
    input_file (str): The input file.
    output_file (str): The output file.
    """
    # Load the stats
    stats = pd.read_csv(input_file)
    
    # Normalize the stats
    for column in stats.columns:
        if column == "team_id":
            continue
        stats[column] = normalize_column(stats[column])
    
    # Write the normalized stats to the output file
    stats.to_csv(output_file, index=False)

def spread_layout(input_file, output_file):

    with open(input_file, "r") as infile:
        lines = infile.readlines()
    
    with open(output_file, "w") as outfile:
        
        outfile.write("week,home_team,away_team,neutral,spread\n")

        neutral_count = 0

        for line in lines[1:]:
            
            if line == "\n":
                continue

            parts = line.split(",")

            week = parts[0].strip()
            home_team = parts[1].strip()
            away_team = parts[2].strip()
            points_home = parts[3].strip()
            points_away = parts[4].strip()
            neutral = parts[9].strip()

            # calculate the spread
            spread = int(points_away) - int(points_home)

            # write the new line
            new_line = f"{week},{home_team},{away_team},{neutral},{spread}\n"
            outfile.write(new_line)

def moneyline_layout(input_file, output_file):
    
    with open(input_file, "r") as infile:
        lines = infile.readlines()
    
    with open(output_file, "w") as outfile:
        
        outfile.write("week,home_team,away_team,moneyline\n")

        neutral_count = 0

        for line in lines[1:]:
            
            if line == "\n":
                continue

            parts = line.split(",")

            week = parts[0].strip()
            home_team = parts[1].strip()
            away_team = parts[2].strip()
            spread = parts[4].strip()

            # calculate the moneyline
            moneyline = 0

            if int(spread) < 0:
                moneyline = 1
            else:
                moneyline = -1

            # write the new line
            new_line = f"{week},{home_team},{away_team},{moneyline}\n"
            outfile.write(new_line)

def no_recursion_stats(input_file, output_file):
    '''
    take the first 5 columns of the input file and write them to the output file
    '''
    with open(input_file, "r") as infile:
        lines = infile.readlines()
    
    with open(output_file, "w") as outfile:
        
        outfile.write("team_id,win_percentage,power_level,power_yardage,power_turnovers\n")

        for line in lines[1:]:
            
            if line == "\n":
                continue

            parts = line.split(",")

            team_id = parts[0].strip()
            win_percentage = parts[1].strip()
            power_level = parts[2].strip()
            power_yards = parts[3].strip()
            power_to = parts[4].strip()

            # write the new line
            new_line = f"{team_id},{win_percentage},{power_level},{power_yards},{power_to}\n"
            outfile.write(new_line)

def main():
    power_level_differentials = []
    moneylines = []

    # Example of opening a CSV file with pandas
    stats = pd.read_csv("no_recursion_stats.csv")
    games = pd.read_csv("nfl_data_spread.csv")

    for _, game in games.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        moneyline = game["spread"]

        home_stats = stats[stats["team_id"] == home_team]
        away_stats = stats[stats["team_id"] == away_team]

        home_power_level = home_stats["power_level"].values[0]
        away_power_level = away_stats["power_level"].values[0]

        power_level_differential = home_power_level - away_power_level

        power_level_differentials.append(power_level_differential)
        moneylines.append(moneyline)
    
    import matplotlib.pyplot as plt

    # Create a scatter plot
    plt.scatter(power_level_differentials, moneylines)
    plt.xlabel('Power Level Differential')
    plt.ylabel('Moneyline')
    plt.title('Power Level Differential vs Moneyline')
    plt.grid(True)
    plt.show()
        

if __name__ == "__main__":
    main()
                        