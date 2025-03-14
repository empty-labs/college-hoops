import json
import os
import pandas as pd


def add_dictionary_to_json(dct: dict, filename: str):
    """Create/Append json team data

    Args:
        dct (dict): dictionary of new team data
        filename (str): Name of JSON team file
    """

    existing_data = None

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as file:
            existing_data = json.load(file)

    with open(filename, "w") as file:
        # Find current teams
        new_teams = list(dct.keys())

        # Compare against current list of teams
        if existing_data:
            existing_teams = list(existing_data.keys())

            # Add team if it doesn't already exist
            for t in new_teams:
                if t not in existing_teams:
                    existing_data[t] = dct[t]
                    print(f"Adding new data set for {t}.")
                print(f"Data set already exists for {t}.")

        else:  # Add new team data if JSON is empty
            existing_data = {}

            for t in new_teams:
                existing_data[t] = dct[t]
                print(f"Adding new data set for {t}.")

        if existing_data:
            json.dump(existing_data, file, indent=4)  # `indent=4` makes it more readable
            print("JSON updated")


def print_season_end_team_win_loss(filename: str):
    """Print team win-loss records

    Args:
        filename (str): Name of JSON team file
    """

    if os.path.exists(filename):
        # Read Stats
        df = pd.read_json(filename)

        for t in list(df.keys()):
            win = df[t]["W"][-1]
            loss = df[t]["L"][-1]
            print(f"{t}: {win}-{loss}")
