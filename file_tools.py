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

def main():
    join_files(1970, 2024, "stats_files/stats", "all_stats.csv")

if __name__ == "__main__":
    main()
                        