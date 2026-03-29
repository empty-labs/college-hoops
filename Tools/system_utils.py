# Third party libraries
import csv
import pandas as pd
import sqlite3


def write_to_sql(df: pd.DataFrame, table_name: str):
    """Save matchups to SQL table

    Args:
        df (pd.DataFrame): Matchup dataframe for all teams in this season
        table_name (str): Name of table for matchups in this season
    """

    conn = sqlite3.connect('Data/Databases/matchups.db')

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()
    print(f'Finished writing to SQL table: {table_name}')


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
