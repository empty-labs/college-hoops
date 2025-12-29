# Local libraries
import Tools.ratings_utils as ru
import Tools.system_utils as sys

# Third party packages
import streamlit as st


# Page config
st.set_page_config(
    page_title="March Madness Bracketology",
    layout="centered"
)

# Title & description
st.title("🏀 March Madness Bracketology Simulator")
st.markdown("""
Simulate NCAA tournament brackets using team ratings, seeding, and
custom assumptions (chalk, upset bias, etc.).
""")

st.divider()

# Seasons
seasons = [2021, 2022, 2023, 2024, 2025]
def convert_season_to_string(season: int):
    return f"{season} - {season + 1}"

seasons_str = [convert_season_to_string(season) for season in seasons]

# Choose Season Start
season_start = st.selectbox(
    "Season Start",
    options=seasons_str,
    help="Choose what season to start simulation"
)
# Choose Season End
season_end = st.selectbox(
    "Season End",
    options=seasons_str,
    help="Choose what season to end simulation"
)

# Simulation Method
simulation_method = st.selectbox(
    "Simulation Method",
    options=[
        "Pure Ratings",
        "Ratings + Seeding (Chalk)",
        "Ratings + Upset Bias"
    ],
    help="Choose how game outcomes are determined"
)

# TODO: Break into utility script for simulator by year
year = season_start[:4]  # Get baseline of start year
FILENAME = f"Data/Seasons/data_{year}.json"
TOURNAMENT_FILENAME = f"Data/Tournaments/tournament_{year}.csv"
PICKS_FILENAME = f"Data/Tournament Picks/picks_{year}.csv"

# Run simulation
run_button = st.button("Run Tournament Simulation")

if run_button:
    st.subheader("Simulation Results")

    with st.spinner("Simulating tournament..."):
        # Placeholder for your real logic

        # Create data frame for valid teams in the current season that can be used for tournament simulation
        score_df = ru.set_rating_data_frame(filename=FILENAME)

        _, _, tourney_dict, results = ru.simulate_tournament(filename=TOURNAMENT_FILENAME,
                                                             ratings=None,
                                                             debug=True)

    st.success("Simulation complete!")

    st.markdown("### Tournament Results")
    st.write(results)