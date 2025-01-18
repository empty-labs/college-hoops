import scraper_util


class TeamList():

    def __init__(self, url: str):
        """Scrape Sports-Reference site for list of all teams

        Args:
            url (str): site for all teams
        """
        df = scraper_util.scrape_team_list(url=url)
        keys = df.keys()
        for k in keys:
            print(k, df[k][0], df[k][1], df[k][2], df[k][3], df[k][10])
