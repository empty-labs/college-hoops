import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

url = "https://www.sports-reference.com/cbb/schools/kansas/2024-schedule.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

table = soup.find("table", {"id": "schedule"})
# Wrap the HTML string in StringIO
html_io = StringIO(str(table))
df = pd.read_html(html_io)[0]  # Convert the table to a DataFrame
print(df.head())