import scraper_util


class Team():

    def __init__(self, url: str):
        """Scrape Sports-Reference site and apply correction for empty keys

        Args:
            url (str): site for this team's schedule of outcomes
        """
        df = scraper_util.scrape_team_schedule(url=url)
        keys = df.keys()
        for k in keys:
            print(k, df[k][0], df[k][1], df[k][2], df[k][3], df[k][10])
