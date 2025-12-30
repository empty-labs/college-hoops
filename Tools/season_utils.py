SEASONS = [2021, 2022, 2023, 2024, 2025]


def convert_season_to_string(season: int):
    """season (str): season string"""
    return f"{season} - {season + 1}"


def convert_season_start_to_year(seasons: str):
    """seasons (str): seasons string"""
    return int(seasons[:4])  # Get start year


def convert_season_end_to_year(seasons: str):
    """seasons (str): seasons string"""
    return int(seasons[-4:]) - 1 # Get end year


def create_year_list(years):
    """Create list of years if not already created

    Args:
        years: season years

    Returns:
        years (list): list of years
    """
    if type(years) != list:
        years = [years]

    return years


def create_filenames(years):
    """Create filenames based on chosen season/years

    Args:
        years (list): list of years

    Returns:
        filename (str): filename string
        tournament_filename (str): tournament filename string
        picks_filename (str): picks filename string
        ratings_filename (str): ratings filename string
    """

    years = create_year_list(years)

    tournament_year = years[-1]
    filename_years = f"{years[0]}-{tournament_year}"

    filename = f"Data/Seasons/data_{tournament_year}.json"
    tournament_filename = f"Data/Tournaments/tournament_{tournament_year}.csv"
    picks_filename = f"Data/Tournament Picks/picks_{tournament_year}.csv"
    ratings_filename = f"Data/Season Ratings/data_{filename_years}.json"

    return filename, tournament_filename, picks_filename, ratings_filename
