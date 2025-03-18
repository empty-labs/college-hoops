# Third party libraries
import csv
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


def write_tournament_to_csv(tourney_dict: dict, filename: str, rating_type: str):
    """Write tournament results to CSV

    Args:
        tourney_dict (dict): tournament dictionary of all matchups
        filename (str): Name of CSV tournament team file
        rating_type (str): name of rating system
    """
    csv_filename = filename.replace(".csv", f"_{rating_type}.csv")

    # Convert dictionary to a CSV-friendly format
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        keys = list(tourney_dict.keys())
        writer.writerow(keys)  # Header

        for i in range(len(tourney_dict[keys[0]])):
            row = []
            for key in keys:
                row.append(tourney_dict[key][i])
            writer.writerow(row)  # Combine team name with stats

        print(f"\nCSV written to {csv_filename}")
