# Third party libraries
from bs4 import BeautifulSoup
from io import StringIO
import json
import os
import pandas as pd
import requests
import time

# Local libraries
import Classes.Team as Team
import Tools.json_utils as ju

URL_PREFIX = "https://www.sports-reference.com"
AMP = "&amp;"
WAIT_TIME_SEC = 5


def skip_table_breaks(df: pd.DataFrame, header: str):
    """Skip breaks in the table where headers are being used in column data

    Args:
        df (pd.DataFrame): team schedule data frame
        header (str): name of column

    Return:
        df (pd.DataFrame): team schedule data frame
    """

    # Need to skip break in table
    drop_idxs = []
    for i in range(len(df[header])):
        if df[header][i] == header:
            drop_idxs.append(i)

    df = df.drop(drop_idxs)
    df.reset_index(drop=True, inplace=True)

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
    for i in range(len(df["Opponent"])):
        school_names.append(df["Opponent"][i].split('\xa0')[0])

    df["Opponent"] = school_names

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


def scrape_team_schedule(url: str, debug: bool=False):
    """Scrape Sports-Reference site and apply correction for empty keys

    Args:
        url (str): site for this team's schedule of outcomes
        debug (bool): flag to print debug statements

    Returns:
        df (pd.DataFrame): team schedule data frame
    """

    time.sleep(WAIT_TIME_SEC) # Mimic human behavior
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    time.sleep(WAIT_TIME_SEC) # Mimic human behavior

    # Review table for "id="
    if debug:
        tables = soup.find_all('table')
        print(tables)

    table = str(soup.find("table", {"id": "schedule"}))
    # Wrap the HTML string in StringIO
    html_io = StringIO(table)

    df = None

    try:

        df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame
        df = skip_table_breaks(df=df, header="Opponent")
        df = clean_school_name(df=df)

        # Rename unnamed keys
        df.rename(columns={'Unnamed: 4': 'Site'}, inplace=True)
        df.rename(columns={'Unnamed: 8': 'Outcome'}, inplace=True)
    except ValueError as e:
        print('Value Error')

    return df


def scrape_team_list(url: str, debug: bool=False):
    """Scrape Sports-Reference site for list of all teams

    Args:
        url (str): site for all teams
        debug (bool): flag to print debug statements

    Returns:
        df (pd.DataFrame): team list data frame
    """

    time.sleep(WAIT_TIME_SEC) # Mimic human behavior
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            print(f"Retry after {retry_after} seconds.")
            return

    soup = BeautifulSoup(response.content, "html.parser")
    time.sleep(WAIT_TIME_SEC) # Mimic human behavior

    # Review table for "id="
    if debug:
        tables = soup.find_all('table')
        print(tables)

    table = str(soup.find("table", {"id": "NCAAM_schools"}))
    # Wrap the HTML string in StringIO
    html_io = StringIO(table)

    df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame
    df = skip_table_breaks(df=df, header="School")

    # Add URLs to data frame
    df["URL"] = add_urls(table=table)

    if debug:
        for i in range(len(df["School"])):
            print(i, df["School"][i], df["URL"][i])

    return df


def add_team_to_dictionary(team_url: str, school: str, filename: str):
    """Add data if file doesn't exist already

    Args:
        team_url (str): URL for team matchup data
        school (str): team involved in matchup
        filename (str): Name of JSON team file
    """

    team = Team.Team(url=team_url)

    if team.df is not None:
        dct = {}
        dct[school] = team.df.to_dict(orient="list")
        ju.add_dictionary_to_json(dct=dct,
                                  filename=filename)
        print(f"{school} added to JSON successfully.")
    else:
        print(f"{school} has no records.")


def add_teams_to_json(team_list, filename: str, url_suffix: str):
    """Add all teams to the JSON file if they don't already exist

    Args:
        team_list: class holding data frame of all teams
        filename (str): Name of JSON team file
        url_suffix (str): suffix for URLs
    """

    for i in range(len(team_list.df["URL"])):
        team_url = team_list.df["URL"][i] + url_suffix
        school = team_list.df["School"][i]
        print(f"{i}. {school}")

        existing_data = None

        # Check if file exists already
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as file:
                existing_data = json.load(file)

        if existing_data:

            existing_teams = list(existing_data.keys())

            # Add data if it doesn't exist already
            if school not in existing_teams:
                add_team_to_dictionary(team_url=team_url,
                                       school=school,
                                       filename=filename)
            else:
                print(f"{school} already exists in current JSON.")

        else:
            add_team_to_dictionary(team_url=team_url,
                                   school=school,
                                   filename=filename)
