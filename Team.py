import scraper_util


class Team():

    def __init__(self, url: str):
        """Scrape Sports-Reference site and apply correction for empty keys

        Args:
            url (str): site for this team's schedule of outcomes
        """
        self.df = scraper_util.scrape_team_schedule(url=url)
