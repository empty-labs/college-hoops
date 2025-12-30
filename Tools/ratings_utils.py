# Third party libraries
from datetime import datetime as dt
import numpy as np
import pandas as pd

ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final 4", "Championship"]
ROUND_POINTS = [10, 20, 40, 80, 160, 320]
ROUND_1_SEEDING = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]


def fix_time_format(time_str: str):
    """Function to fix "a"/"p" to "AM"/"PM"""
    if time_str.endswith("a"):
        return time_str.replace("a", "AM")
    elif time_str.endswith("p"):
        return time_str.replace("p", "PM")
    return time_str  # If already correct


def set_datetime_object(date: str, time: str):
    """Clean up date and time into datetime format"""

    # Fix time format
    if time is None:
        time = "12:00p"  # Generic time if none provided
    time_str_fixed = fix_time_format(time)

    # Add date time
    # Convert date string to datetime object
    date_obj = dt.strptime(date, "%a, %b %d, %Y")

    # Convert time string to 24-hour format
    time_obj = dt.strptime(time_str_fixed, "%I:%M%p").time()

    # Combine date and time into one datetime object
    datetime_obj = dt.combine(date_obj, time_obj)
    return datetime_obj


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

    datetime_obj = set_datetime_object(date=date, time=time)
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
        "Date": [],
        "Home": [],
        "Home_Score": [],
        "Away": [],
        "Away_Score": [],
        "Winner": []
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
    data_tuples = list(zip(score_dict["Date"],
                           score_dict["Home"],
                           score_dict["Home_Score"],
                           score_dict["Away"],
                           score_dict["Away_Score"],
                           score_dict["Winner"]))

    # Remove duplicates using set() and convert back to list
    unique_data = list(set(data_tuples))

    # Keep duplicates
    # unique_data = data_tuples

    # ✅ Sort the unique data by Date
    unique_data_sorted = sorted(unique_data, key=lambda x: x[0])

    # Unzip the sorted, unique data back into separate lists
    score_dict["Date"], score_dict["Home"], score_dict["Home_Score"], score_dict["Away"], score_dict["Away_Score"], \
    score_dict["Winner"] = map(list, zip(*unique_data_sorted))

    # Convert to data frame
    score_df = pd.DataFrame(score_dict)

    return score_df


def update_rating_matrix(mtx, i: int, j: int):
    """Update rating matrix"""
    mtx[i, i] += 1
    mtx[j, j] += 1
    mtx[i, j] -= 1
    mtx[j, i] -= 1

    return mtx


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
        h, a = row["Home"], row["Away"]
        i, j = team_index[h], team_index[a]
        home_margin = row["Home_Score"] - row["Away_Score"]

        M = update_rating_matrix(mtx=M, i=i, j=j)

        b[i] += home_margin
        b[j] -= home_margin

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
        h, a, winner = row["Home"], row["Away"], row["Winner"]
        i, j = team_index[h], team_index[a]

        # Update matrix
        C = update_rating_matrix(mtx=C, i=i, j=j)

        # Update b vector
        if winner == h:
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


def compile_srs_ratings(filename: str, debug: bool=False):
    """"""

    # Read Stats
    teams_df = pd.read_json(filename)
    teams = list(teams_df.keys())

    ratings = []

    for team in teams:

        team_df = teams_df[team]
        final_srs = -999

        for i in range(len(team_df["Type"])):

            # Non Tournament games
            if team_df["Type"][i] != "NCAA" and team_df["Type"][i] != "CIT" and team_df["Tm"][i] is not None and \
                    team_df["Opp"][i] is not None:
                final_srs = team_df["SRS"][i]

        ratings.append(float(final_srs))

    # Convert ratings to a dictionary
    srs_ratings = {team: rating for team, rating in zip(teams, ratings)}

    # Sort and display rankings
    srs_rankings = sorted(srs_ratings.items(), key=lambda x: x[1], reverse=True)

    if debug:
        for rank, (team, rating) in enumerate(srs_rankings, 1):
            print(f"{rank}. {team}: {rating:.2f}")

    return srs_ratings


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
        h, a, winner, mov = row["Home"], row["Away"], row["Winner"], row["Home_Score"] - row["Away_Score"]

        # Assign outcome (1 if h wins, 0 if a wins)
        outcome = 1 if winner == h else 0

        elo_ratings[h], elo_ratings[a] = update_elo(r1=elo_ratings[h], r2=elo_ratings[a],
                                                    outcome=outcome, mov=mov, K=K, adjust_K=adjust_K)

    # Sort and display rankings
    elo_rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    if debug:
        for rank, (team, rating) in enumerate(elo_rankings, 1):
            print(f"{rank}. {team}: {rating:.2f}")

    return elo_ratings


def add_ratings_per_game(score_df: pd.DataFrame, initial_ratings: int=None):
    """Calculate Massey, Colley, Elo ratings for each team for each game

    Args:
        score_df (pd.DataFrame): matchup score data frame
        initial_ratings (int): starting rating for all teams (default 1500)

    Returns:
        rating_score_df (dict): score data frame with Massey ratings
    """

    # Get unique teams and index them
    teams = list(set(score_df["Home"]).union(set(score_df["Away"])))
    team_index = {team: i for i, team in enumerate(teams)}
    N = len(teams)

    # Initialize Massey matrix and score vector
    M = np.zeros((N, N))
    mb = np.zeros(N)

    # Initialize Colley matrix (C) and RHS vector (b)
    C = np.eye(N) * 2  # Start with 2 on the diagonal
    cb = np.ones(N)  # Initialize b with 1s

    # Initialize Elo
    elo_ratings = {team: initial_ratings.get(team, 1500) if initial_ratings else 1500 for team in teams}
    adj_elo_ratings = {team: initial_ratings.get(team, 1500) if initial_ratings else 1500 for team in teams}

    rating_scores = []
    curr = 0
    curr_pct = 1

    # Fill the matrix and score vector
    for _, row in score_df.iterrows():

        # Team Prep
        h, a, winner = row["Home"], row["Away"], row["Winner"]
        i, j = team_index[h], team_index[a]
        hPts, aPts = row["Home_Score"], row["Away_Score"]
        home_margin = hPts - aPts

        # Copy current M, mb, C, cb
        M_copy = M.copy()
        mb_copy = mb.copy()
        C_copy = C.copy()
        cb_copy = cb.copy()

        # Fix singularity: replace last row with ones
        M_copy[-1, :] = 1
        mb_copy[-1] = 0

        # Solve Massey, Colley systems
        try:
            massey_ratings = np.linalg.solve(M_copy, mb_copy)
            colley_ratings = np.linalg.solve(C_copy, cb_copy)
        except np.linalg.LinAlgError:
            massey_ratings = np.zeros(N)  # fallback at very start
            colley_ratings = np.zeros(N)

        # Append to current pregame rating data
        rating_scores.append({
            "Date": row["Date"],
            "Home": h,
            "Home_Score": hPts,
            "Away": a,
            "Away_Score": aPts,
            "Winner": winner,
            "Home_Massey": massey_ratings[i],
            "Away_Massey": massey_ratings[j],
            "Home_Colley": colley_ratings[i],
            "Away_Colley": colley_ratings[j],
            "Home_Elo": elo_ratings[h],
            "Away_Elo": elo_ratings[a],
            "Home_Adj_Elo": adj_elo_ratings[h],
            "Away_Adj_Elo": adj_elo_ratings[a]
        })

        # Update Massey matrix
        M = update_rating_matrix(mtx=M, i=i, j=j)

        mb[i] += home_margin
        mb[j] -= home_margin

        # Update Colley matrix
        C = update_rating_matrix(mtx=C, i=i, j=j)

        # Update cb vector
        if winner == h:
            cb[i] += 0.5
            cb[j] -= 0.5
        else:
            cb[i] -= 0.5
            cb[j] += 0.5

        # Assign Elo outcome (1 if h wins, 0 if a wins)
        outcome = 1 if winner == h else 0

        # Traditional Elo rating
        elo_ratings[h], elo_ratings[a] = update_elo(r1=elo_ratings[h], r2=elo_ratings[a],
                                                    outcome=outcome, mov=home_margin, K=30, adjust_K=False)

        # Adjusted Elo rating
        adj_elo_ratings[h], adj_elo_ratings[a] = update_elo(r1=adj_elo_ratings[h], r2=adj_elo_ratings[a],
                                                    outcome=outcome, mov=home_margin, K=30, adjust_K=True)
        # Update status
        curr += 1
        pct = round(100 * (curr / len(score_df)), 3)
        if pct >= curr_pct:  # Print every % instead of every update
            print(f"Complete: {curr} / {len(score_df)} or {pct}%")
            curr_pct += 1

    # Convert to DataFrame
    rating_score_df = pd.DataFrame(rating_scores)

    return rating_score_df


def set_tournament_team_rating(i: int, tourney_dict: dict, team: str, rating: float):
    """Helper function to assign team and rating to next round of tournament

    Args:
        i (int): current index
        tourney_dict (dict): tournament dictionary of current and next round matchups
        team (str): winning team
        rating (float): team rating

    Returns:
        tourney_dict (dict): updated tournament dictionary of current and next round matchups
    """

    if i % 2 == 0:
        # Assign 1st team if even index
        tourney_dict["Team1"].append(team)
        tourney_dict["Rating1"].append(rating)
    else:
        # Assign 2nd team if odd index
        tourney_dict["Team2"].append(team)
        tourney_dict["Rating2"].append(rating)

    return tourney_dict


def simulate_next_round(tourney_dict: dict, rd: int, ratings: dict=None):
    """Simulate next round of tournament using current round of team matchups

    Args:
        tourney_dict (dict): tournament dictionary of current round matchups
        rd (int): current round of tournament matchups
        ratings (dict): dictionary of ratings (use chalk/seeding otherwise)

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
        rating1 = tourney_dict["Rating1"][i]
        rating2 = tourney_dict["Rating2"][i]

        # Rating system provided
        if ratings is not None:

            # Add 2nd round teams
            if rating1 >= rating2:
                tourney_dict = set_tournament_team_rating(i=i, tourney_dict=tourney_dict, team=team1, rating=rating1)

            else:
                tourney_dict = set_tournament_team_rating(i=i, tourney_dict=tourney_dict, team=team2, rating=rating2)

        else: # Chalk method

            # Add 2nd round teams
            if rating1 <= rating2:
                tourney_dict = set_tournament_team_rating(i=i, tourney_dict=tourney_dict, team=team1, rating=rating1)

            else:
                tourney_dict = set_tournament_team_rating(i=i, tourney_dict=tourney_dict, team=team2, rating=rating2)

        # Update round and game counters
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
        total_points (int): points based on round
        num_teams (int): number of teams in this round of tournament
        results (str): printed copy of tournament results
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

    # Map correct picks to total points
    total_points = correct_picks * ROUND_POINTS[rd - 1]
    total_possible_points = num_teams * ROUND_POINTS[rd - 1]

    results = f"Round: {rd} / {ROUND_NAMES[rd - 1]} - Correct picks: {correct_picks} out of {num_teams} - Total Points: {total_points} out of {total_possible_points}"

    if debug:
        print(results)

    return correct_picks, total_points, num_teams, results


def simulate_tournament(filename: str, ratings: dict=None, debug: bool=True):
    """Simulate tournament outcomes based on given rating system

    Args:
        filename (str): Name of CSV tournament team file
        ratings (dict): dictionary of ratings (use chalk/seeding otherwise)
        debug (bool): flag to print debug statements

    Returns:
        total_correct_picks (int): number of correct picks
        total_points (int): points based on round
        tourney_dict (dict): tournament dictionary of all matchups
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

    tourney_results = ""

    # Add ratings to 1st round
    for i in range(32):
        team1 = tourney_df["Team1"][i]
        team2 = tourney_df["Team2"][i]

        # Rating system provided
        if ratings is not None:

            if team1 not in list(ratings.keys()):
                ratings[team1] = -999

            if team2 not in list(ratings.keys()):
                ratings[team2] = -999

            tourney_dict["Rating1"].append(ratings[team1])
            tourney_dict["Rating2"].append(ratings[team2])

        else:  # Chalk method

            team1_seed = ROUND_1_SEEDING[(2*i) % 16]
            team2_seed = ROUND_1_SEEDING[(2*i + 1) % 16]

            tourney_dict["Rating1"].append(team1_seed)
            tourney_dict["Rating2"].append(team2_seed)

        tourney_dict["Round"].append(tourney_df["Round"][i])
        tourney_dict["Game"].append(tourney_df["Game"][i])
        tourney_dict["Team1"].append(team1)
        tourney_dict["Team2"].append(team2)

    total_correct_picks = 0
    total_points = 0
    total_num_teams = 0

    for rd in range(1, 7):
        tourney_dict = simulate_next_round(tourney_dict=tourney_dict,
                                           ratings=ratings,
                                           rd=rd)

        correct_picks, points, num_teams, results = calculate_correct_picks(
            tourney_dict=tourney_dict,
            tourney_df=tourney_df,
            rd=rd, debug=debug)

        tourney_results += results + "\n"

        total_correct_picks += correct_picks
        total_points += points
        total_num_teams += num_teams

    tourney_results += f"\nTotal correct picks in tournament: {total_correct_picks} out of {total_num_teams}"
    tourney_results += f"\nTotal points in tournament: {total_points} out of 1920"

    if debug:
        print(tourney_results)

    return total_correct_picks, total_points, tourney_dict, tourney_results


def compile_ratings_dict(score_df: pd.DataFrame):
    """Compile all rating systems together into one dictionary"""

    massey_ratings = calculate_massey_ratings(score_df=score_df,
                                              debug=False)
    colley_ratings = calculate_colley_ratings(score_df=score_df,
                                              debug=False)
    elo_ratings = calculate_elo_ratings(score_df=score_df,
                                        K=30,
                                        debug=False,
                                        adjust_K=False)
    adj_elo_ratings = calculate_elo_ratings(score_df=score_df,
                                            K=30,
                                            debug=False,
                                            adjust_K=True)

    ratings = {}

    for k in massey_ratings.keys():
        # Initialize dictionary entry
        ratings[k] = {}

        # Assign values to entry
        ratings[k]['Massey'] = massey_ratings[k]
        ratings[k]['Colley'] = colley_ratings[k]
        ratings[k]['Elo'] = elo_ratings[k]
        ratings[k]['Adj_Elo'] = adj_elo_ratings[k]

    return ratings


def simulate_tournament_with_all_ratings(filename: str, ratings: dict, model=None, debug: bool=True):
    """Simulate tournament outcomes based on given rating system

    Args:
        filename (str): Name of CSV tournament team file
        ratings (dict): dictionary of ratings
        model: classification model to predict results
        debug (bool): flag to print debug statements

    Returns:
        total_correct_picks (int): number of correct picks
        total_points (int): points based on round
        tourney_dict (dict): tournament dictionary of all matchups
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

    model_ratings = {}

    # Add ratings to 1st round
    for i in range(32):
        team1 = tourney_df["Team1"][i]
        team2 = tourney_df["Team2"][i]

        massey_diff = ratings[team1]['Massey'] - ratings[team2]['Massey']
        colley_diff = ratings[team1]['Colley'] - ratings[team2]['Colley']
        elo_diff = ratings[team1]['Elo'] - ratings[team2]['Elo']
        adj_elo_diff = ratings[team1]['Adj_Elo'] - ratings[team2]['Adj_Elo']

        x1_dict = {
            'Massey_diff': [massey_diff],
            'Colley_diff': [colley_diff],
            'Elo_diff': [elo_diff],
            'Adj_Elo_diff': [adj_elo_diff]
        }

        x1 = pd.DataFrame(x1_dict)

        # Grab probability of team 1 win
        model_ratings[team1] = model.predict_proba(x1)[:, 1][0]
        model_ratings[team2] = 1 - model_ratings[team1]

        tourney_dict["Round"].append(tourney_df["Round"][i])
        tourney_dict["Game"].append(tourney_df["Game"][i])
        tourney_dict["Team1"].append(team1)
        tourney_dict["Team2"].append(team2)
        tourney_dict["Rating1"].append(model_ratings[team1])
        tourney_dict["Rating2"].append(model_ratings[team2])

    total_correct_picks = 0
    total_points = 0
    total_num_teams = 0

    for rd in range(1, 7):
        tourney_dict = simulate_next_round(tourney_dict=tourney_dict,
                                           ratings=model_ratings,
                                           rd=rd)

        correct_picks, points, num_teams = calculate_correct_picks(tourney_dict=tourney_dict,
                                                                   tourney_df=tourney_df,
                                                                   rd=rd, debug=debug)

        total_correct_picks += correct_picks
        total_points += points
        total_num_teams += num_teams

    if debug:
        print(f"\nTotal correct picks in tournament: {total_correct_picks} out of {total_num_teams}")
        print(f"\nTotal points in tournament: {total_points} out of 1920")

    return total_correct_picks, total_points, tourney_dict


def normalize_ratings(ratings: dict, weight: float = 1):
    """Normalize rating values between 0 and 1

    Args:
        ratings (dict): dictionary of ratings
        weight (float): scaled value applied to normalized rating systems

    Returns:
        normalized_dict (dict): dictionary of normalized ratings
    """

    # Min-Max Scaling
    min_val = min(ratings.values())
    max_val = max(ratings.values())

    normalized_dict = {k: weight * (v - min_val) / (max_val - min_val) for k, v in ratings.items()}

    return normalized_dict


def apply_ratings_weights_to_maximize_correct_picks(massey_ratings: dict, colley_ratings: dict, adj_elo_ratings: dict,
                                                    elo_ratings: dict, srs_ratings: dict, tournament_filename: str):
    """Apply linear combination of ratings systems to maximize correct picks

    Args:
        massey_ratings (dict): Massey ratings dictionary
        colley_ratings (dict): Colley ratings dictionary
        adj_elo_ratings (dict): Adjusted Elo ratings dictionary
        elo_ratings (dict): Elo ratings dictionary
        srs_ratings (dict): SRS ratings dictionary
        tournament_filename (str): filepath for tournament results

    Returns:
        tourney_dict (dict): tournament dictionary of all matchups
    """

    # Iterate through all possible ratings weights
    iterations = list(np.arange(-1, 1.25, 0.25))
    total_iterations = 0

    weight_dict = {'Massey': [], 'Colley': [], 'Adjusted Elo': [], 'Elo': [], 'SRS': [], 'Correct': [], 'Points': []}

    # Loop through all possible iterations for each rating system
    # TODO: Make ratings loop more dynamic instead of hardcoded

    total = len(iterations) ** 5
    passed_integer = 1
    print(f"Total iterations: {total}")

    for w1 in iterations:
        for w2 in iterations:
            for w3 in iterations:
                for w4 in iterations:
                    for w5 in iterations:
                        total_iterations += 1

                        total_correct_picks, total_points, tourney_dict = apply_custom_weights(
                            massey_ratings=massey_ratings, colley_ratings=colley_ratings,
                            adj_elo_ratings=adj_elo_ratings, elo_ratings=elo_ratings, srs_ratings=srs_ratings,
                            tournament_filename=tournament_filename, massey_weight=w1, colley_weight=w2,
                            adj_elo_weight=w3, elo_weight=w4, srs_weight=w5, debug=False)

                        # Store weights in dictionary
                        weight_dict['Massey'].append(w1)
                        weight_dict['Colley'].append(w2)
                        weight_dict['Adjusted Elo'].append(w3)
                        weight_dict['Elo'].append(w4)
                        weight_dict['SRS'].append(w5)
                        weight_dict['Correct'].append(total_correct_picks)
                        weight_dict['Points'].append(total_points)

                        if (100 * total_iterations / total) > passed_integer:
                            print(f"{passed_integer} percent complete")
                            passed_integer += 1

    # Print max correct picks
    max_correct = max(weight_dict['Correct'])
    print(f"Max correct picks: {max_correct} in {total_iterations} iterations")

    # Print which combinations result in max correct picks
    num_max = 0

    for i in range(len(weight_dict['Correct'])):

        if weight_dict['Correct'][i] == max_correct:
            num_max += 1
            print(f"{num_max}. Massey: {weight_dict['Massey'][i]:.3f}, Colley: {weight_dict['Colley'][i]:.3f}, Adjusted Elo: {weight_dict['Adjusted Elo'][i]:.3f}, Elo: {weight_dict['Elo'][i]:.3f}, SRS: {weight_dict['SRS'][i]:.3f}")

    # Print max points total
    max_points = max(weight_dict['Points'])
    print(f"Max total points: {max_points} in {total_iterations} iterations")

    # Print which combinations result in max points
    num_max = 0

    for i in range(len(weight_dict['Points'])):

        if weight_dict['Points'][i] == max_points:
            num_max += 1
            print(f"{num_max}. Massey: {weight_dict['Massey'][i]:.3f}, Colley: {weight_dict['Colley'][i]:.3f}, Adjusted Elo: {weight_dict['Adjusted Elo'][i]:.3f}, Elo: {weight_dict['Elo'][i]:.3f}, SRS: {weight_dict['SRS'][i]:.3f}")

    return tourney_dict


def apply_custom_weights(massey_ratings: dict, colley_ratings: dict, adj_elo_ratings: dict, elo_ratings: dict,
                         srs_ratings: dict, tournament_filename: str, massey_weight: float, colley_weight: float,
                         adj_elo_weight: float, elo_weight: float, srs_weight: float, debug: bool=True):
    """Apply custom weights for all rating systems

    Args:
        massey_ratings (dict): Massey ratings dictionary
        colley_ratings (dict): Colley ratings dictionary
        adj_elo_ratings (dict): Adjusted Elo ratings dictionary
        elo_ratings (dict): Elo ratings dictionary
        srs_ratings (dict): SRS ratings dictionary
        tournament_filename (str): filepath for tournament results
        massey_weight (float): Massey ratings weight
        colley_weight (float): Colley ratings weight
        adj_elo_weight (float): Adjusted Elo ratings weight
        elo_weight (float): Elo ratings weight
        srs_weight (float): SRS ratings weight
        debug (bool): flag to print debug statements

    Returns:
        total_correct_picks (int): number of correct picks
        total_points (int): points based on round
        tourney_dict (dict): tournament dictionary of all matchups
    """

    # Normalize ratings by weight
    normalized_massey_ratings = normalize_ratings(ratings=massey_ratings, weight=massey_weight)
    normalized_colley_ratings = normalize_ratings(ratings=colley_ratings, weight=colley_weight)
    normalized_adj_elo_ratings = normalize_ratings(ratings=adj_elo_ratings, weight=adj_elo_weight)
    normalized_elo_ratings = normalize_ratings(ratings=elo_ratings, weight=elo_weight)
    normalized_srs_ratings = normalize_ratings(ratings=srs_ratings, weight=srs_weight)

    # Add ratings togather
    combined_ratings = {}

    for key, v in normalized_massey_ratings.items():
        combined_ratings[key] = v
    for key, v in normalized_colley_ratings.items():
        combined_ratings[key] += v
    for key, v in normalized_adj_elo_ratings.items():
        combined_ratings[key] += v
    for key, v in normalized_elo_ratings.items():
        combined_ratings[key] += v
    for key, v in normalized_srs_ratings.items():
        combined_ratings[key] += v

    total_correct_picks, total_points, tourney_dict = simulate_tournament(
        filename=tournament_filename, ratings=combined_ratings, debug=debug)

    return total_correct_picks, total_points, tourney_dict

def create_score_df(years: list):
    """Create score data frame based on seasons of interest

    Args:
        years (list): list of years

    Returns:
        score_df (pd.DataFrame): score dataframe
    """
    score_df = None

    for year in years:
        filename = f"Data/Seasons/data_{year}.json"

        if year is years[0]:
            # Create data frame for valid teams in the current season that can be used for tournament simulation
            score_df = set_rating_data_frame(filename=filename)
        else:
            # Concatenate
            new_season_score_df = set_rating_data_frame(filename=filename)
            score_df = pd.concat([score_df, new_season_score_df], ignore_index=True)

    return score_df
