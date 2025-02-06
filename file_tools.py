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

def new_layout(input_file, output_file):

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


def main():
    new_layout("nfl_data.csv", "better_nfl_data.csv")
if __name__ == "__main__":
    main()
                        