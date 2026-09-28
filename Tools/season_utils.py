SEASONS = [2021, 2022, 2023, 2024, 2025]


def convert_season_to_string(season: int):
    """Convert season number to string (data filenames are based on year of final season game, e.g. 2021 = 2020-2021)

    Args:
        season (str): season string
    """
    return f'{season - 1} - {season}'


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
        season_table_name_list (list): List of table names for matchups in this season
        tournament_filename (str): tournament filename string
        picks_filename (str): picks filename string
        ratings_table_name_list (list): List of table names for ratings in this season
        final_ratings_table_name_list (list): List of table names for final ratings in this season
    """

    years = create_year_list(years)

    # Set file name for single/multiple season(s)
    tournament_year = years[-1]
    if years[0] == tournament_year:
        filename_years = years[0]
    else:
        filename_years = f'{years[0]}-{tournament_year}'

    tournament_filename = f'Data/Tournaments/tournament_{tournament_year}.csv'
    picks_filename = f'Data/Tournament Picks/picks_{filename_years}.csv'

    # Grab list of all table names by year
    season_table_name_list = []
    ratings_table_name_list = []
    final_ratings_table_name_list = []

    for y in years:
        season_table_name_list.append(f'season_{y}')
        ratings_table_name_list.append(f'ratings_{y}')
        final_ratings_table_name_list.append(f'final_ratings_{y}')

    return season_table_name_list, tournament_filename, picks_filename, ratings_table_name_list, final_ratings_table_name_list


SEASONS_STR = [convert_season_to_string(season) for season in SEASONS]
