from Tools import scraper_util


class Tournament:

    def __init__(self, url: str, debug: bool=False):
        """Scrape Sports-Reference site and apply correction for empty keys

        Args:
            url (str): site for this team's schedule of outcomes
            debug (bool): flag to print debug statements
        """

        self.df = scraper_util.scrape_tournament(url=url,
                                                 debug=debug)
