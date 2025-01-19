import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

AMP = "&amp;"


def scrape_team_schedule(url: str):
    """Scrape Sports-Reference site and apply correction for empty keys

    Args:
        url (str): site for this team's schedule of outcomes

    Returns:
        df (pd.DataFrame): team schedule data frame
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Review table for "id="
    # tables = soup.find_all('table')
    # print(tables)

    table = str(soup.find("table", {"id": "schedule"}))
    # Wrap the HTML string in StringIO
    html_io = StringIO(table)
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

    # Review table for "id="
    # tables = soup.find_all('table')
    # print(tables)

    table = str(soup.find("table", {"id": "NCAAM_schools"}))
    # Wrap the HTML string in StringIO
    html_io = StringIO(table)

    df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame

    # Need to skip School School
    drop_idxs = []
    for i in range(len(df["School"])):
        if df["School"][i] == "School":
            drop_idxs.append(i)

    df = df.drop(drop_idxs)
    df.reset_index(drop=True, inplace=True)

    # Grab URLs from table manually
    table_urls = table.split('<a href="')
    urls = []
    schools_by_url = []

    for i in range(1, len(table_urls)):

        # URL
        url = table_urls[i].split('">')[0]
        urls.append(url.strip())

        # School Name near URL
        school_by_url = table_urls[i].split('</a')[0].split('">')[1]

        # Need to handle &amp; for School from URL
        if AMP in school_by_url:
            school_by_url = school_by_url.replace(AMP, "&")

        schools_by_url.append(school_by_url.strip())

    for i in range(len(df["School"])):
        # URL
        print("URL", i, df["School"][i], "==", schools_by_url[i], urls[i])

        if df["School"][i] != schools_by_url[i]:
            print("MTMT  cat", i)

    return df

