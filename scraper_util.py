import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd


def scrape_team_schedule(url: str):
    """Scrape Sports-Reference site and apply correction for empty keys

    Args:
        url (str): site for this team's schedule of outcomes

    Returns:
        df (pd.DataFrame): team schedule data frame
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # # Review table for "id="
    # tables = soup.find_all('table')
    # print(tables)

    table = soup.find("table", {"id": "schedule"})
    # Wrap the HTML string in StringIO
    html_io = StringIO(str(table))
    df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame

    # Rename unnamed keys
    df.rename(columns={'Unnamed: 4': 'Site'}, inplace=True)
    df.rename(columns={'Unnamed: 8': 'Outcome'}, inplace=True)

    return df


def scrape_team_list(url: str):
    """Scrape Sports-Reference site for list of all teams

    Args:
        url (str): site for all teams

    Returns:
        df (pd.DataFrame): team list data frame
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # # Review table for "id="
    # tables = soup.find_all('table')
    # print(tables)

    table = soup.find("table", {"id": "NCAAM_schools"})
    # Wrap the HTML string in StringIO
    html_io = StringIO(str(table))
    df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame

    return df

