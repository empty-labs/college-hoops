# College Hoops

This is a bracketology project seeking to understand whether we can predict the winners for each March Madness matchup based on their season and conference tournament results.  Primarily, we're interested in just the team scores and home advantage to predict.  Future projects might include player stats and use injury status to create a more nuanced prediction tool.

# Scraping Input Data

In order to create the input data sets for this project, I opted to scrape the data from "https://www.sports-reference.com" using the "BeautifulSoup" Python package (see Conda env section below for details on how to install).  For the original scope of this project I've included results for seasons 2021-2025 (2025 may not include NCAA tournament), however you can always use the "Team Scraper" notebook to grab more seasons as needed.

#### Note: If you create a season data set only a few games into an ongoing season, you might have to delete the data set and re-run the scraper to include additional games.  It's highly encouraged to create the data set at the end of the conference tournament stage of the season if you want to use it for an ongoing season prior to filling out your brackets.

# Ratings Systems

The rating systems applied in the matchup prediction are as follows:
1. Colley
2. Massey
3. Adjusted Elo
4. Elo
5. SRS
6. Combined (using numerical analysis)

TODO: Difference between Elo and Adjusted Elo.

# Manually Creating Tournament CSV's

The tournament data sets must be manually created if you choose to run the tool for seasons outside the orignal scope of this project (2021-2025 included here).

TODO: Here's an idea of how to do so using the ESPN bracket:

## Conda environment

1. Create new conda environment
```
conda env create --name college-hoops
```
2. Add packages to conda
```
conda install anaconda::pandas
```
```
conda install -c anaconda beautifulsoup4
```
```
conda install anaconda::lxml
```
```
conda install anaconda::html5lib
```
3. Set up jupyter for conda environment ([sauce](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook))

```
pip install jupyter ipykernel
```
```
python -m ipykernel install --user --name college-hoops --display-name "college-hoops"
```