# Third party libraries
from datetime import datetime as dt
import pandas as pd
import numpy as np

ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final 4", "Championship"]


def fix_time_format(time_str: str):
    """Function to fix "a"/"p" to "AM"/"PM"""
    if time_str.endswith("a"):
        return time_str.replace("a", "AM")
    elif time_str.endswith("p"):
        return time_str.replace("p", "PM")
    return time_str  # If already correct


def set_score_entry(dct: dict, home_team: str, away_team: str, home_team_score: int, away_team_score: int,
                    date: str, time: str):
    """Set score entry

    Args:
        dct (dict): score dictionary
        home_team (str): home team name
        away_team (str): away team name
        home_team_score (int): home team score
        away_team_score (int): away team score
        date (str): date of matchup
        time (str): time of matchup

    Returns:
        dct (dict): Massey score dictionary
    """

    dct["Home"].append(home_team)
    dct["Away"].append(away_team)
    dct["Home_Score"].append(home_team_score)
    dct["Away_Score"].append(away_team_score)

    if home_team_score > away_team_score:
        dct["Winner"].append(home_team)
    else:
        dct["Winner"].append(away_team)

    # Fix time format
    time_str_fixed = fix_time_format(time)

    # Add date time
    # Convert date string to datetime object
    date_obj = dt.strptime(date, "%a, %b %d, %Y")

    # Convert time string to 24-hour format
    time_obj = dt.strptime(time_str_fixed, "%I:%M%p").time()

    # Combine date and time into one datetime object
    datetime_obj = dt.combine(date_obj, time_obj)
    dct["Date"].append(datetime_obj)

    return dct


def set_rating_data_frame(filename: str):
    """Set score data frame prior to rating calculation

    Args:
        filename (str): Name of JSON team file

    Returns:
        score_df (pd.DataFrame): score data frame
    """

    # Initialize score dictionary
    score_dict = {
        "Home": [],
        "Home_Score": [],
        "Away": [],
        "Away_Score": [],
        "Winner": [],
        "Date": []
    }

    # Read Stats
    teams_df = pd.read_json(filename)
    all_teams = list(teams_df.keys())

    for team in all_teams:

        team_df = teams_df[team]

        for i in range(len(team_df["Type"])):

            # Non Tournament games
            if team_df["Type"][i] != "NCAA" and team_df["Type"][i] != "CIT" and team_df["Tm"][i] is not None and \
                    team_df["Opp"][i] is not None:

                # Find which team is home/away (None = home, @ = away, N = neutral/assign home to winner?)
                if team_df["Site"][i] is None:
                    # Current team is home team
                    score_dict = set_score_entry(dct=score_dict, home_team=team, away_team=team_df["Opponent"][i],
                                                 home_team_score=int(team_df["Tm"][i]),
                                                 away_team_score=int(team_df["Opp"][i]),
                                                 date=team_df["Date"][i], time=team_df["Time"][i])
                elif team_df["Site"][i] == "@":
                    # Opponent team is away team
                    score_dict = set_score_entry(dct=score_dict, home_team=team_df["Opponent"][i], away_team=team,
                                                 home_team_score=int(team_df["Opp"][i]),
                                                 away_team_score=int(team_df["Tm"][i]),
                                                 date=team_df["Date"][i], time=team_df["Time"][i])
                else:  # Neutral site home team advantage goes to winner
                    if team_df["Outcome"][i] == "W":
                        # Current team is home team
                        score_dict = set_score_entry(dct=score_dict, home_team=team, away_team=team_df["Opponent"][i],
                                                     home_team_score=int(team_df["Tm"][i]),
                                                     away_team_score=int(team_df["Opp"][i]),
                                                     date=team_df["Date"][i], time=team_df["Time"][i])
                    else:
                        # Opponent team is away team
                        score_dict = set_score_entry(dct=score_dict, home_team=team_df["Opponent"][i], away_team=team,
                                                     home_team_score=int(team_df["Opp"][i]),
                                                     away_team_score=int(team_df["Tm"][i]),
                                                     date=team_df["Date"][i], time=team_df["Time"][i])

    # Zip all lists together and sort by date
    sorted_data = sorted(zip(score_dict["Date"],
                             score_dict["Home"],
                             score_dict["Home_Score"],
                             score_dict["Away"],
                             score_dict["Away_Score"],
                             score_dict["Winner"]), key=lambda x: x[0])

    # Unzip the sorted data back into lists
    score_dict["Date"], score_dict["Home"], score_dict["Home_Score"], score_dict["Away"], score_dict["Away_Score"], \
    score_dict["Winner"] = map(list, zip(*sorted_data))

    # Convert to data frame
    score_df = pd.DataFrame(score_dict)

    print(score_df.head())

    return score_df


def calculate_massey_ratings(score_df: pd.DataFrame, debug: bool=False):
    """Calculate Massey ratings for each team and sort in ranked order

    Args:
        score_df (pd.DataFrame): matchup score data frame
        debug (bool): flag to print debug statements

    Returns:
        massey_ratings (dict): dictionary of Massey ratings
    """

    # Get unique teams and index them
    teams = list(set(score_df["Home"]).union(set(score_df["Away"])))
    team_index = {team: i for i, team in enumerate(teams)}
    N = len(teams)

    # Initialize Massey matrix and score vector
    M = np.zeros((N, N))
    b = np.zeros(N)

    # Fill the matrix and score vector
    for _, row in score_df.iterrows():
        h, a = team_index[row["Home"]], team_index[row["Away"]]
        home_margin = row["Home_Score"] - row["Away_Score"]

        M[h, h] += 1
        M[a, a] += 1
        M[h, a] -= 1
        M[a, h] -= 1

        b[h] += home_margin
        b[a] -= home_margin

    # Replace last row to enforce sum constraint (makes matrix invertible)
    M[-1, :] = 1
    b[-1] = 0

    ratings = np.linalg.solve(M, b)

    # Convert ratings to a dictionary
    massey_ratings = {team: rating for team, rating in zip(teams, ratings)}

    # Sort and display rankings
    massey_rankings = sorted(massey_ratings.items(), key=lambda x: x[1], reverse=True)

    if debug:
        for rank, (team, rating) in enumerate(massey_rankings, 1):
            print(f"{rank}. {team}: {rating:.2f}")

    return massey_ratings


def calculate_colley_ratings(score_df: pd.DataFrame, debug: bool=False):
    """Calculates Colley rankings given a game results DataFrame.

    Args:
        score_df (pd.DataFrame): matchup score data frame
        debug (bool): flag to print debug statements

    Returns:
        colley_ratings (dict): dictionary of Massey ratings
    """

    # Get unique teams and index them
    teams = list(set(score_df["Home"]).union(set(score_df["Away"])))
    team_index = {team: i for i, team in enumerate(teams)}  # Map teams to indices
    N = len(teams)

    # Initialize Colley matrix (C) and RHS vector (b)
    C = np.eye(N) * 2  # Start with 2 on the diagonal
    b = np.ones(N)  # Initialize b with 1s

    # Populate matrix and vector using game results
    for _, row in score_df.iterrows():
        t1, t2, winner = row["Home"], row["Away"], row["Winner"]
        i, j = team_index[t1], team_index[t2]

        # Update matrix
        C[i, i] += 1  # Each team gets an additional game played
        C[j, j] += 1
        C[i, j] -= 1
        C[j, i] -= 1

        # Update b vector
        if winner == t1:
            b[i] += 0.5
            b[j] -= 0.5
        else:
            b[i] -= 0.5
            b[j] += 0.5

    # Solve for ratings
    ratings = np.linalg.solve(C, b)

    # Convert ratings to a dictionary
    colley_ratings = {team: rating for team, rating in zip(teams, ratings)}

    # Sort and display rankings
    colley_rankings = sorted(colley_ratings.items(), key=lambda x: x[1], reverse=True)

    if debug:
        for rank, (team, rating) in enumerate(colley_rankings, 1):
            print(f"{rank}. {team}: {rating:.2f}")

    return colley_ratings


def expected_outcome(r1, r2):
    """Calculate expected probability of team 1 winning against team 2."""
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def mov_multiplier(mov, blowout_factor=2.2):
    """Scale K-factor based on Margin of Victory (MOV)."""
    return np.log(abs(mov) + 1) * blowout_factor


def update_elo(r1: float, r2: float, outcome: int, mov: int, K: int=40, adjust_K: bool=True):
    """Update Elo ratings with MOV scaling

    Args:
        r1 (float): current Elo rating for team 1
        r2 (float): current Elo rating for team 2
        outcome (int): binary value for winner of matchup, 1: team 1 is winner, 0: team 2 is winner
        mov (int): margin of victory, ([team 1 score] - [team 2 score])
        K (int): rating adjustment factor (default 30)
        adjust_K (bool): use adjusted K value based on MOV

    Returns:
        r1_new (float): updated Elo rating for team 1
        r2_new (float): updated Elo rating for team 2
    """

    E1 = expected_outcome(r1, r2)
    E2 = 1 - E1  # Expected for team 2

    # Scale adjustment by MOV
    K_adj = K * mov_multiplier(mov) if adjust_K else K

    # Adjust ratings
    r1_new = r1 + K_adj * (outcome - E1)
    r2_new = r2 + K_adj * ((1 - outcome) - E2)

    return r1_new, r2_new


def calculate_elo_ratings(score_df: pd.DataFrame, initial_ratings: int=None, K: int=40, debug: bool=False,
                          adjust_K: bool=True):
    """Calculates Elo rankings given a game results DataFrame.

    Args:
        score_df (pd.DataFrame): matchup data frame
        initial_ratings (int): starting rating for all teams (default 1500)
        K (int): rating adjustment factor (default 30)
        debug (bool): flag to print debug statements
        adjust_K (bool): use adjusted K value based on MOV

    Returns:
        elo_ratings (dict): dictionary of Elo ratings
    """

    # Get unique teams and index them
    teams = list(set(score_df["Home"]).union(set(score_df["Away"])))
    elo_ratings = {team: initial_ratings.get(team, 1500) if initial_ratings else 1500 for team in teams}

    for _, row in score_df.iterrows():
        t1, t2, winner, mov = row["Home"], row["Away"], row["Winner"], row["Home_Score"] - row["Away_Score"]

        # Assign outcome (1 if t1 wins, 0 if t2 wins)
        outcome = 1 if winner == t1 else 0

        elo_ratings[t1], elo_ratings[t2] = update_elo(r1=elo_ratings[t1], r2=elo_ratings[t2],
                                                      outcome=outcome, mov=mov, K=K, adjust_K=adjust_K)

    # Sort and display rankings
    elo_rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    if debug:
        for rank, (team, rating) in enumerate(elo_rankings, 1):
            print(f"{rank}. {team}: {rating:.2f}")

    return elo_ratings


def simulate_next_round(tourney_dict: dict, ratings: dict, rd: int):
    """Simulate next round of tournament using current round of team matchups

    Args:
        tourney_dict (dict): tournament dictionary of current round matchups
        ratings (dict): dictionary of ratings
        rd (int): current round of tournament matchups

    Returns:
        tourney_dict (dict): tournament dictionary of current and next round matchups
    """
    game = 0

    # Convert dictionary to DataFrame, find indices matching current round
    df = pd.DataFrame(tourney_dict)
    matching_indices = df[df["Round"] == rd].index.tolist()

    for i in matching_indices:

        team1 = tourney_dict["Team1"][i]
        team2 = tourney_dict["Team2"][i]
        rating1 = ratings[team1]
        rating2 = ratings[team2]

        # Add 2nd round teams
        if tourney_dict["Rating1"][i] >= tourney_dict["Rating2"][i]:

            if i % 2 == 0:
                # Assign 1st team if even index
                tourney_dict["Team1"].append(team1)
                tourney_dict["Rating1"].append(rating1)
            else:
                # Assign 2nd team if odd index
                tourney_dict["Team2"].append(team1)
                tourney_dict["Rating2"].append(rating1)

        else:

            if i % 2 == 0:
                # Assign 1st team if even index
                tourney_dict["Team1"].append(team2)
                tourney_dict["Rating1"].append(rating2)
            else:
                # Assign 2nd team if odd index
                tourney_dict["Team2"].append(team2)
                tourney_dict["Rating2"].append(rating2)

        if i % 2 == 0:
            game += 1

            tourney_dict["Round"].append(rd + 1)
            tourney_dict["Game"].append(game)

    # Apply final round corrections
    if rd == 6:

        placeholder = "No Value"

        if len(tourney_dict["Team1"]) > len(tourney_dict["Team2"]):
            tourney_dict["Team2"].append(placeholder)
            tourney_dict["Rating2"].append(placeholder)
        else:
            tourney_dict["Team1"].append(placeholder)
            tourney_dict["Rating1"].append(placeholder)

    return tourney_dict


def calculate_correct_picks(tourney_dict: dict, tourney_df: pd.DataFrame, rd: int, debug: bool=True):
    """Assess number of correct picks based on teams in next round

    Args:
        tourney_dict (dict): tournament dictionary of current round matchups
        tourney_df (pd.DataFrame): tournament data frame
        rd (int): current round of tournament matchups
        debug (bool): flag to print debug statements

    Returns:
        correct_picks (int): number of correct picks in this round of tournament
        num_teams (int): number of teams in this round of tournament
    """

    # Need to add teams to correct pick checker
    correct_picks = 0

    # Convert dictionary to DataFrame, find indices matching current round
    df = pd.DataFrame(tourney_dict)
    matching_indices = df[df["Round"] == (rd + 1)].index.tolist()
    num_teams = len(df[df["Round"] == rd].index.tolist())

    for i in matching_indices:

        # Compare 1st round scores by checking 2nd round participants
        if tourney_df["Team1"][i] == tourney_dict["Team1"][i]:
            correct_picks += 1
        if tourney_df["Team2"][i] == tourney_dict["Team2"][i]:
            correct_picks += 1

    if debug:
        print(f"Round: {rd} / {ROUND_NAMES[rd - 1]} - Correct picks: {correct_picks} out of {num_teams}")

    return correct_picks, num_teams


def simulate_tournament(filename: str, ratings: dict, debug: bool=True):
    """Simulate tournament outcomes based on given rating system

    Args:
        filename (str): Name of CSV tournament team file
        ratings (dict): dictionary of ratings
        debug (bool): flag to print debug statements
    """

    # Load tournament CSV file into a DataFrame
    tourney_df = pd.read_csv(filename)

    tourney_dict = {
        "Round": [],
        "Game": [],
        "Team1": [],
        "Team2": [],
        "Rating1": [],
        "Rating2": []
    }

    # Add ratings to 1st round
    for i in range(32):
        team1 = tourney_df["Team1"][i]
        team2 = tourney_df["Team2"][i]
        rating1 = ratings[team1]
        rating2 = ratings[team2]

        tourney_dict["Round"].append(tourney_df["Round"][i])
        tourney_dict["Game"].append(tourney_df["Game"][i])
        tourney_dict["Team1"].append(team1)
        tourney_dict["Team2"].append(team2)
        tourney_dict["Rating1"].append(rating1)
        tourney_dict["Rating2"].append(rating2)

    total_correct_picks = 0
    total_num_teams = 0

    for rd in range(1, 7):
        tourney_dict = simulate_next_round(tourney_dict=tourney_dict,
                                           ratings=ratings,
                                           rd=rd)

        correct_picks, num_teams = calculate_correct_picks(tourney_dict=tourney_dict,
                                                           tourney_df=tourney_df,
                                                           rd=rd, debug=debug)

        total_correct_picks += correct_picks
        total_num_teams += num_teams

    if debug:
        print(f"\nTotal correct picks in tournament: {total_correct_picks} out of {total_num_teams}")

    return total_correct_picks
