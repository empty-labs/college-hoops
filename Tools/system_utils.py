# Third party libraries
import csv
import os


def write_to_parquet(spark, df, filename):
    """Save matchups to Parquet file if they don't already exist

    Args:
        spark: PySpark object
        df (pd.DataFrame): Matchup dataframe for all teams in this season
        filename (str): Name of Parquet matchup file
    """

    # Create DataFrame
    spark_df = spark.createDataFrame(df)

    # Write to Parquet
    if os.path.exists(filename):

        # File exists → read existing data, append, and remove duplicates
        df_existing = spark.read.parquet(filename)

        # Combine old + new
        df_combined = df_existing.union(spark_df)

        # Drop duplicates
        df_final = df_combined.dropDuplicates()

        # Overwrite with updated data
        df_final.write.mode('overwrite').parquet(filename)
        print('Data successfully updated.')

    else:

        # Write file with new data
        spark_df.write.mode('overwrite').parquet(filename)
        print('Data successfully created.')


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
