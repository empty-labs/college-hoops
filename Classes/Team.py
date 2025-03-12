from Tools import scraper_utils


class Team:

    def __init__(self, url: str, debug: bool=False):
        """Scrape Sports-Reference site and apply correction for empty keys

        Args:
            url (str): site for this year's tournament
            debug (bool): flag to print debug statements
        """

        self.df = scraper_utils.scrape_team_schedule(url=url,
                                                     debug=debug)
