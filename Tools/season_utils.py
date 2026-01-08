SEASONS = [2021, 2022, 2023, 2024, 2025]


def convert_season_to_string(season: int):
    """Convert season number to string (data filenames are based on year of final season game, e.g. 2021 = 2020-2021)

    Args:
        season (str): season string
    """
    return f"{season - 1} - {season}"


def convert_season_to_year(seasons: str):
    """seasons (str): seasons string"""
    return int(seasons[-4:]) # Get end year


def year_range(start_year: int, end_year: int):
    """Create list of years between start_year and end_year

    Args:
        start_year (int): start year
        end_year (int): end year
    """
    return list(range(start_year, end_year + 1))


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
        final_ratings_filename (str): final ratings filename string
    """

    years = create_year_list(years)

    tournament_year = years[-1]
    if years[0] == tournament_year:
        filename_years = years[0]
    else:
        filename_years = f"{years[0]}-{tournament_year}"

    filename = f"Data/Seasons/data_{tournament_year}.json"
    tournament_filename = f"Data/Tournaments/tournament_{tournament_year}.csv"
    picks_filename = f"Data/Tournament Picks/picks_{filename_years}.csv"
    ratings_filename = f"Data/Season Ratings/data_{filename_years}.json"
    final_ratings_filename = f"Data/Season Ratings/final_data_{filename_years}.json"

    return filename, tournament_filename, picks_filename, ratings_filename, final_ratings_filename


SEASONS_STR = [convert_season_to_string(season) for season in SEASONS]
