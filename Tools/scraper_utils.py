# Third party libraries
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import pandas as pd
import random
import requests
import time

# Local libraries
import Tools.system_utils as sys

session = requests.Session()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

URL_PREFIX = 'https://www.sports-reference.com'
AMP = '&amp;'


def skip_table_breaks(df: pd.DataFrame, header: str):
    """Skip breaks in the table where headers are being used in column data

    Args:
        df (pd.DataFrame): team schedule data frame
        header (str): name of column

    Return:
        df (pd.DataFrame): team schedule data frame
    """
    # Need to skip break in table
    df = df[df[header] != header]
    df = df.reset_index(drop=True)

    return df


def clean_school_name(df: pd.DataFrame):
    """Fix school names that use ranking in name that show '\xa0' in string

    Args:
        df (pd.DataFrame): team schedule data frame

    Return:
        df (pd.DataFrame): team schedule data frame
    """

    # Need to skip break in table
    school_names = []
    for i in range(len(df['Opponent'])):
        school_names.append(df['Opponent'][i].split('\xa0')[0])

    df['Opponent'] = school_names

    return df


def add_urls(table: str):
    """Extract URLs from HTML table string

    Args:
        table (str): HTML table data in string format

    Returns:
        urls (list): list of all team URLs
    """

    # Grab URLs from table manually
    table_urls = table.split('<a href="')
    urls = []

    for i in range(1, len(table_urls)):

        # URL
        url = URL_PREFIX + table_urls[i].split('">')[0]
        urls.append(url.strip())

    return urls


def scrape_team_list(url: str, debug: bool=False):
    """Scrape Sports-Reference site for list of all teams

    Args:
        url (str): site for all teams
        debug (bool): flag to print debug statements

    Returns:
        df (pd.DataFrame): team list data frame
    """

    response = session.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Check for rate limit error code
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            print(f'Retry after {retry_after} seconds.')
            return

    # Review table for 'id='
    if debug:
        tables = soup.find_all('table')
        print(tables)

    # Search for school name table
    table = str(soup.find('table', {'id': 'NCAAM_schools'}))

    # Wrap the HTML string in StringIO
    html_io = StringIO(table)

    # Convert the table to a DataFrame
    df = pd.read_html(html_io)[0]
    df = skip_table_breaks(df=df, header='School')

    # Add URLs to data frame
    df['URL'] = add_urls(table=table)

    if debug:
        for i in range(len(df['School'])):
            print(i, df['School'][i], df['URL'][i])

    return df




def scrape_team_schedule(url: str, debug: bool=False):
    """Scrape Sports-Reference site and apply correction for empty keys

    Args:
        url (str): site for this team's schedule of outcomes
        debug (bool): flag to print debug statements

    Returns:
        df (pd.DataFrame): team schedule data frame
    """

    response = session.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Check for rate limit error code
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            print(f'Retry after {retry_after} seconds.')
            return

    # Review table for 'id='
    if debug:
        tables = soup.find_all('table')
        print(tables)

    table = str(soup.find('table', {'id': 'schedule'}))

    # Wrap the HTML string in StringIO
    html_io = StringIO(table)

    df = None

    try:

        df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame
        df = skip_table_breaks(df=df, header='Opponent')
        df = clean_school_name(df=df)

        # Rename unnamed keys
        df.rename(columns={'Unnamed: 4': 'Site'}, inplace=True)
        df.rename(columns={'Unnamed: 8': 'Outcome'}, inplace=True)

    except ValueError as e:
        pass

    return df


def parse_team_matchups(team_list_df: pd.DataFrame, season_table_name: str, url_suffix: str):
    """Parse all team matchups and save to SQL file

    Args:
        team_list_df (pd.DataFrame): Team list data frame
        season_table_name (str): Name of table for matchups in this season
        url_suffix (str): suffix for URLs
    """

    team_matchups = []
    urls = team_list_df.loc[:, 'URL'] + url_suffix
    schools = team_list_df.loc[:, 'School']

    for i in range(len(team_list_df)):

        # Grab school name
        school = schools[i]
        print(f'{i + 1}. {school}')

        # Pull data for one team
        df = scrape_team_schedule(url=urls[i])

        if df is not None:

            # Add school name to matchup table
            df['School'] = school

            # Add to team matchup list
            team_matchups.append(df)

        else:

            print('--> No games recorded for this team')

        time.sleep(random.uniform(3.5, 5.5))

    # Concat all DataFrames
    team_matchups_df = pd.concat(team_matchups)

    # Write matchups to SQL table
    sys.write_matchups_to_sql(df=team_matchups_df, season_table_name=season_table_name)


def batch_parse_team_matchups(team_list_df: pd.DataFrame, season_table_name: str, url_suffix: str, batch_size: int=3):
    """Parse all team matchups and save to Parquet file

    STATUS: Trips rate limit, forces 1-hr (3600 sec) pause

    Args:
        team_list_df (pd.DataFrame): Team list data frame
        season_table_name (str): Name of table for matchups in this season
        url_suffix (str): suffix for URLs
        batch_size (int): number of teams to scrape
    """

    team_matchups = []
    urls = team_list_df.loc[:, 'URL'] + url_suffix
    schools = team_list_df.loc[:, 'School']

    for i in range(0, len(team_list_df), batch_size):

        # Set batch of team URL's
        batch_urls = urls[i:i+batch_size]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Grab team matchup tables in batch
            results = list(executor.map(scrape_team_schedule, batch_urls))

        for j, df in enumerate(results):

            # Grab school name
            school = schools[i + j]
            print(f'{i + j + 1}. {school}')

            if df is not None:

                # Add school name + year to matchup table
                df['Team'] = school

                # Add to team matchup list
                team_matchups.append(df)

            else:

                print('--> No games recorded for this team')

        time.sleep(random.uniform(3.5, 5.5))

    # Concat all DataFrames
    team_matchups_df = pd.concat(team_matchups)

    # Write matchups to SQL table
    sys.write_matchups_to_sql(df=team_matchups_df, season_table_name=season_table_name)
