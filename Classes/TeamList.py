from Tools import scraper_utils


class TeamList:

    def __init__(self, url: str, debug: bool=False):
        """Scrape Sports-Reference site for list of all teams

        Args:
            url (str): site for all teams
            debug (bool): flag to print debug statements
        """

        self.df = scraper_utils.scrape_team_list(url=url,
                                                 debug=debug)
