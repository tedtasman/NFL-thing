'''
league_model.py
version 1.0.0

Theodore Tasman
2025-01-31

This module contains the class definitions used to model the league data.
'''

# HYPERPARAMETERS
WIN_WEIGHT = 1
TIE_WEIGHT = 0.5
LOSS_WEIGHT = -1
DIFFERENTIAL_LOSS_WEIGHT = 1 # differential already includes sign, preserve the sign of the result
RECURSIVE_LOSS_WEIGHT = 0 # can't be negative because it's recursive, sign will be determined by the recursion depth
DIFFERENCE_WEIGHT = 0.05
RECURSION_DEPTH = 5
DECAY_RATE = 0.9

class League:

    def __init__(self):
        '''
        Constructor for the League class.
        '''
        self.seasons = []

    def add_season(self, season):
        '''
        Adds a season to the league.
        season: Season object
        '''
        self.seasons.append(season)
    
    def load_season(self, year, filename):
        '''
        Adds a season to the league.
        year: int
        filename: str (path to the games file)
        '''
        season = Season(year)
        season.load_games(filename)
        self.add_season(season)
    
    def load_seasons(self, start_year, end_year):
        '''
        Adds a list of seasons to the league.
        start_year: int
        end_year: int
        '''
        for year in range(start_year, end_year + 1):
            self.load_season(year, f'nfl_data_{year}.csv')

    def generate_files(self, start_year, end_year, recursion_depth):
        '''
        Generates files with the season summaries.
        start_year: int
        end_year: int
        '''
        for year in range(start_year, end_year + 1):
            season = self.get_season(year)
            season.generate_file(f'stats_{year}.csv', recursion_depth)

class Season:

    def __init__(self, year):
        '''
        Constructor for the Season class.
        year: int (start year of the season)
        league: League object
        '''
        self.year = year
        self.teams = []
        self.games = []

    def __str__(self):
        '''
        Returns a string representation of the Season object.
        '''
        return self.win_percentages

    __repr__ = __str__

    def add_game(self, game):
        '''
        Adds a game to the season.
        game: Game object
        '''
        self.games.append(game)

        game.home_team.addWeek(game.home_team_week)
        game.away_team.addWeek(game.away_team_week)
    
    def load_games(self, filename):
        '''
        Adds a list of games to the season.
        filename: str (path to the games file)
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Iterate through the lines of the file, skipping the header
        for line in lines[1:]:

            line = line.strip().split(',')
            week = line[0]

            # track if the away team won the game
            away_winner = False

            # Determine home and away teams based on the '@' symbol
            if '@' in line[2]:
                home_team = self.get_team(line[3])
                away_team = self.get_team(line[1])

                # if the away team won mark the game as such
                away_winner = True

            else:
                home_team = self.get_team(line[1])
                away_team = self.get_team(line[3])

            # Determine the score based on the winner
            if away_winner:
                away_score = int(line[4])
                home_score = int(line[5])
                away_yardage = int(line[6])
                away_turnovers = int(line[7])
                home_yardage = int(line[8])
                home_turnovers = int(line[9])
            else:
                home_score = int(line[4])
                away_score = int(line[5])
                home_yardage = int(line[6])
                home_turnovers = int(line[7])
                away_yardage = int(line[8])
                away_turnovers = int(line[9])
            
            # Create the game object
            game = Game(week, home_team, away_team, home_score, away_score, home_yardage, away_yardage, home_turnovers, away_turnovers)
            self.add_game(game)

            # add the game to the teams' schedules
            home_team.add_week(game.home_team_week)
            away_team.add_week(game.away_team_week)
        
    def get_team(self, team_id):
        '''
        Returns the team object associated with the given team_id.
        team_id: str
        returns Team object
        '''

        # Check if the team already exists
        for team in self.teams:
            if team.team_id == team_id:
                return team
        
        # If the team does not exist create a new team
        team = Team(team_id)
        self.teams.append(team)
        return team

    @property
    def win_percentages(self):
        '''
        Returns a string representation of the Season object.
        '''
        teams = []
        for team in self.teams:
            teams.append((team.team_id, team.win_percentage))
        
        teams.sort(key=lambda x: x[1], reverse=True)
        return f'\n{self.year} Season Results:\n\n' + '\n'.join([f' {rank + 1}.\t{team_id} ({win_percentage:.3f})' for rank, (team_id, win_percentage) in enumerate(teams)]) + '\n'

    @property
    def power_rankings(self):
        '''
        Returns a string representation of the Season object.
        '''
        teams = []
        for team in self.teams:
            teams.append((team.team_id, team.power_level))
        
        teams.sort(key=lambda x: x[1], reverse=True)
        return f'\n{self.year} Power Rankings:\n\n' + '\n'.join([f' {rank + 1}.\t{team_id} ({power_level:.3f})' for rank, (team_id, power_level) in enumerate(teams)]) + '\n'


    def recursive_power_rankings(self, depth, decay_function=lambda x: x):
        '''
        Returns a string representation of the Season object.
        '''
        teams = []
        for team in self.teams:
            teams.append((team.team_id, 1000*team.recursive_power_level(depth, decay_function)))
        
        teams.sort(key=lambda x: x[1], reverse=True)
        return f'\n{self.year} Power Rankings:\n\n' + '\n'.join([f' {rank + 1}.\t{team_id} ({power_level:.3f})' for rank, (team_id, power_level) in enumerate(teams)]) + '\n'

    def power_differentials(self):
        '''
        Returns a string representation of the Season object.
        '''
        teams = []
        for team in self.teams:
            teams.append((team.team_id, team.power_differential()))
        
        teams.sort(key=lambda x: x[1], reverse=False)
        return f'\n{self.year} Power Differentials:\n\n' + '\n'.join([f' {rank + 1}.\t{team_id} ({power_diff:.3f})' for rank, (team_id, power_diff) in enumerate(teams)]) + '\n'

    def summaries(self):
        '''
        Returns a string representation of the Season object.
        '''
        teams = []
        for team in self.teams:
            teams.append((team.team_id, 
                          team.win_percentage, 
                          team.power_level, 
                          1000*team.recursive_power_level(RECURSION_DEPTH, lambda x: x*DECAY_RATE),
                          team.power_yardage(), 
                          team.power_turnovers(),
                          1000*team.recursive_power_differential(RECURSION_DEPTH, lambda x: x*DECAY_RATE),
                          1000*team.recursive_power_yardage(RECURSION_DEPTH, lambda x: x*DECAY_RATE),
                          1000*team.recursive_power_turnovers(RECURSION_DEPTH, lambda x: x*DECAY_RATE)
                        ))
        
        teams.sort(key=lambda x: x[1], reverse=True)

        return (f'\n{self.year} Season Summaries:\n\n' + 
                '\n'.join([f' {rank + 1}.\t{team_id}\n\tWin Percentage: {win_percentage:.3f}\n\tPower Level: {power_level:.3f}\n\tRecursive Power Level: {recursive_power_level:.3f}\n\tPower Yardage: {power_yardage:.3f}\n\tPower Turnovers: {power_turnovers:.3f}\n\tRecursive Power Differential: {recursive_power_differential:.3f}\n\tRecursive Power Yardage: {recursive_power_yardage:.3f}\n\tRecursive Power Turnovers: {recursive_power_turnovers:.3f}' 
                           for rank, (team_id, 
                                      win_percentage, 
                                      power_level, 
                                      recursive_power_level, 
                                      power_yardage, 
                                      power_turnovers,
                                      recursive_power_differential,
                                      recursive_power_yardage,
                                      recursive_power_turnovers
                                      ) in enumerate(teams)]) + '\n')
        
    def generate_file(self, filename, recursion_depth):
        '''
        Generates a file with the season summaries.
        filename: str
        '''
        output = ['team_id,win_percentage,power_level,power_yardage,power_turnovers,']
        for depth in range(2, recursion_depth + 1):
            output.append(f'recursive_power_level_{depth},recursive_power_differential_{depth},recursive_power_yardage_{depth},recursive_power_turnovers_{depth},')

        output.append('\n')

        for team in self.teams:
            output.append(f'{team.team_id},{team.win_percentage},{team.power_level},{team.power_yardage()},{team.power_turnovers()},')
            for depth in range(2, recursion_depth + 1):
                output.append(f'{1000*team.recursive_power_level(depth, lambda x: x*DECAY_RATE)},{1000*team.recursive_power_differential(depth, lambda x: x*DECAY_RATE)},{1000*team.recursive_power_yardage(depth, lambda x: x*DECAY_RATE)},{1000*team.recursive_power_turnovers(depth, lambda x: x*DECAY_RATE)},')

            output.append('\n')
        
        with open(filename, 'w') as f:
            f.writelines(output)

class Team:
    '''
    A class to represent a team in the league.
    A team object should only exist within the context of a single season, i.e. there is no team legacy.
    '''

    def __init__(self, team_id):
        '''
        Constructor for the Team class.
        team_id: int
        name: str
        city: str
        league: League object
        '''
        self.team_id = team_id
        self.weeks = []
        self.recursive_power_levels = []
    
    def add_week(self, week):
        '''
        Adds a week to the team's schedule.
        week: Team_Week object
        '''
        self.weeks.append(week)
    
    @property
    def win_percentage(self):
        '''
        Returns the team's win percentage.
        '''
        wins = 0
        ties = 0
        total = 0

        for week in self.weeks:

            if week.result < 0:
                wins += 1

            elif week.result == 0:
                ties += 1

            total += 1

        return (wins + 0.5 * ties) / total
    
    @property
    def power_level(self):
        '''
        Returns the team's power level.
        '''

        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.opponent.win_percentage
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.opponent.win_percentage
            elif week.result > 0:
                numerator += LOSS_WEIGHT * (1 - week.opponent.win_percentage) # 1 - win_percentage is the loss percentage
            
            denominator += 1
        
        return numerator / denominator
    

    def recursive_power_level(self, depth, decay_function=lambda x: x):
        '''
        Returns the team's power level.
        '''

        numerator = 0
        denominator = 0
        
        print(f'Calculating for {self.team_id} at level {depth + 1}')

        if depth == 0:
            return self.power_level
        
        for rpl in self.recursive_power_levels:
            if rpl[0] == depth:
                return rpl[1]

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result > 0:
                numerator += RECURSIVE_LOSS_WEIGHT * week.opponent.recursive_power_level(depth - 1, decay_function)
            
            denominator += 1
        self.recursive_power_levels.append((depth, decay_function(numerator / denominator)))
        return decay_function(numerator / denominator)


    def power_differential(self):

        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.result * week.opponent.power_level
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.result * week.opponent.power_level
            elif week.result > 0:
                numerator += DIFFERENTIAL_LOSS_WEIGHT * week.result * week.opponent.power_level
            
            denominator += 1
        
        return numerator / denominator
    
    def power_yardage(self):

        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.yardage * week.opponent.power_level
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.yardage * week.opponent.power_level
            elif week.result > 0:
                numerator += DIFFERENTIAL_LOSS_WEIGHT * week.yardage * week.opponent.power_level
            
            denominator += 1
        
        return numerator / denominator
    
    def power_turnovers(self):

        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.turnovers * week.opponent.power_level
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.turnovers * week.opponent.power_level
            elif week.result > 0:
                numerator += DIFFERENTIAL_LOSS_WEIGHT * week.turnovers * week.opponent.power_level
            
            denominator += 1
        
        return numerator / denominator
    
    def recursive_power_differential(self, depth, decay_function=lambda x: x):

        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.result * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.result * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result > 0:
                numerator += RECURSIVE_LOSS_WEIGHT * week.result * week.opponent.recursive_power_level(depth - 1, decay_function)
            
            denominator += 1
        
        return decay_function(numerator / denominator)
    
    def recursive_power_yardage(self, depth, decay_function=lambda x: x):
        
        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.yardage * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.yardage * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result > 0:
                numerator += RECURSIVE_LOSS_WEIGHT * week.yardage * week.opponent.recursive_power_level(depth - 1, decay_function)
            
            denominator += 1
        
        return decay_function(numerator / denominator)
    
    def recursive_power_turnovers(self, depth, decay_function=lambda x: x):
        
        numerator = 0
        denominator = 0

        for week in self.weeks:

            if week.result < 0:
                numerator += WIN_WEIGHT * week.turnovers * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result == 0:
                numerator += TIE_WEIGHT * week.turnovers * week.opponent.recursive_power_level(depth - 1, decay_function)
            elif week.result > 0:
                numerator += RECURSIVE_LOSS_WEIGHT * week.turnovers * week.opponent.recursive_power_level(depth - 1, decay_function)
            
            denominator += 1
        
        return decay_function(numerator / denominator)


class Team_Week:
    '''
    A class to represent a team's result in a given week.
    This is a one sided perspective of a game.
    '''

    def __init__(self, week, team, opponent, home_score, away_score, home, home_yardage, away_yardage, home_turnovers, away_turnovers):
        '''
        Constructor for the Team_Week class.
        week: str
        opponent: Team object
        score: str (<home_score>-<away_score>)
        home: bool
        '''
        self.week = week
        self.team = team
        self.opponent = opponent
        self.home = home
        self.result = self.get_result(home_score, away_score)
        self.yardage = self.get_yardage(home_yardage, away_yardage)
        self.turnovers = self.get_turnovers(home_turnovers, away_turnovers)
    
    def get_result(self, home_score, away_score):
        '''
        Converts the score string into a team result.
        home_score: int
        away_score: int
        returns score difference (int)
            (opponent score - team score)
            follows betting convention (negative is margin of victory, positive is margin of loss)
        '''

        if self.home:
            return away_score - home_score
        else:
            return home_score - away_score
        
    def get_yardage(self, home_yardage, away_yardage):
        '''
        Converts the score string into a team result.
        home_score: int
        away_score: int
        returns score difference (int)
            (opponent score - team score)
            follows betting convention (negative is margin of victory, positive is margin of loss)
        '''

        if self.home:
            return away_yardage - home_yardage
        else:
            return home_yardage - away_yardage
    
    def get_turnovers(self, home_turnovers, away_turnovers):
        '''
        Converts the score string into a team result.
        home_score: int
        away_score: int
        returns score difference (int)
            (opponent score - team score)
            follows betting convention (negative is margin of victory, positive is margin of loss)
        '''

        if self.home:
            return away_turnovers - home_turnovers
        else:
            return home_turnovers - away_turnovers


class Game:
    '''
    A class to represent a game between two teams.
    '''

    def __init__(self, week, home_team, away_team, home_score, away_score, home_yardage, away_yardage, home_turnovers, away_turnovers):
        '''
        Constructor for the Game class.
        week: str
        home_team: Team object
        away_team: Team object
        home_score: int
        away_score: int
        '''
        self.week = week

        self.home_team = home_team
        self.away_team = away_team

        self.score = (home_score, away_score)

        self.home_team_week = Team_Week(week, home_team, away_team, home_score, away_score, True, home_yardage, away_yardage, home_turnovers, away_turnovers)
        self.away_team_week = Team_Week(week, away_team, home_team, home_score, away_score, False, home_yardage, away_yardage, home_turnovers, away_turnovers)



def main():
    
    Season_2024 = Season(2024)
    Season_2024.load_games('nfl_data_2024.csv')

    Season_2024.generate_file('stats_2024.csv', RECURSION_DEPTH)

if __name__ == '__main__':
    main()