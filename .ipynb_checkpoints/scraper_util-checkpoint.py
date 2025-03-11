import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd


def scrape_team_url(url: str):
    """Scrape Sports-Reference site and apply correction for empty keys

    Args:
        url (str): site for this team's schedule of outcomes
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", {"id": "schedule"})
    # Wrap the HTML string in StringIO
    html_io = StringIO(str(table))
    df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame

    # Rename 'old_key' to 'new_key'
    df.rename(columns={'Unnamed: 4': 'Site'}, inplace=True)
    df.rename(columns={'Unnamed: 8': 'Outcome'}, inplace=True)

    return df

