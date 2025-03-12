# Third party libraries
import pandas as pd
import numpy as np


def set_massey_score_entry(dct: dict, team: str, opponent: str, team_score: int, opponent_score: int):
    """Set Massey score entry

    Args:
        dct (dict): Massey score dictionary
        team (str): current team name
        opponent (str): opponent team name
        team_score (int): current team score
        opponent_score (int): opponent team score

    Returns:
        dct (dict): Massey score dictionary
    """

    dct["Home"].append(team)
    dct["Away"].append(opponent)
    dct["Home_Score"].append(team_score)
    dct["Away_Score"].append(opponent_score)

    return dct


def set_massey_rating_data_frame(filename: str):
    """Set Massey score data frame prior to Massey rating calculation

    Args:
        filename (str): Name of JSON team file

    Returns:
        score_df (pd.DataFrame): Massey score data frame
    """

    # Initialize score dictionary
    score_dict = {
        "Home": [],
        "Home_Score": [],
        "Away": [],
        "Away_Score": []
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
                    score_dict = set_massey_score_entry(dct=score_dict, team=team, opponent=team_df["Opponent"][i],
                                                        team_score=int(team_df["Tm"][i]),
                                                        opponent_score=int(team_df["Opp"][i]))
                elif team_df["Site"][i] == "@":
                    # Opponent team is away team
                    score_dict = set_massey_score_entry(dct=score_dict, team=team_df["Opponent"][i], opponent=team,
                                                        team_score=int(team_df["Opp"][i]),
                                                        opponent_score=int(team_df["Tm"][i]))
                else:
                    if team_df["Outcome"][i] == "W":
                        # Current team is home team
                        score_dict = set_massey_score_entry(dct=score_dict, team=team, opponent=team_df["Opponent"][i],
                                                            team_score=int(team_df["Tm"][i]),
                                                            opponent_score=int(team_df["Opp"][i]))
                    else:
                        # Opponent team is away team
                        score_dict = set_massey_score_entry(dct=score_dict, team=team_df["Opponent"][i], opponent=team,
                                                            team_score=int(team_df["Opp"][i]),
                                                            opponent_score=int(team_df["Tm"][i]))

    # Convert to data frame
    score_df = pd.DataFrame(score_dict)

    return score_df


def calculate_massey_ratings(score_df: pd.DataFrame, debug: bool = False):
    """Calculate Massey ratings for each team and sort in ranked order

    Args:
        score_df (pd.DataFrame): Massey score data frame
        debug (bool): flag to print debug statements

    Returns:
        massey_rankings (dict): dictionary of Massey ratings
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
