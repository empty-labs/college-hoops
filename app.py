# Local libraries
import Tools.ratings_utils as ru
import Tools.season_utils as su

# Third party packages
import streamlit as st


RATINGS_OPTIONS = [
    "Seeding (Chalk)",
    "Massey Ratings",
    "Colley Ratings",
    "Elo Ratings",
    "Adj Elo Ratings",
    "SRS Ratings"
]

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

# Choose Season Start
season_start = st.selectbox(
    "Season Start",
    options=su.SEASONS_STR,
    help="Choose what season to start simulation"
)
st.session_state.season_start= season_start
# Choose Season End
season_end = st.selectbox(
    "Season End",
    options=[y for y in su.SEASONS_STR if y >= st.session_state.season_start],
    help="Choose what season to end simulation"
)

# Simulation Method
simulation_method = st.selectbox(
    "Simulation Method",
    options=RATINGS_OPTIONS,
    help="Choose how game outcomes are determined"
)

# Convert season selections to data set filenames
start_year = su.convert_season_to_year(season_start)
end_year = su.convert_season_to_year(season_end)
years = su.year_range(start_year, end_year)
filename, tournament_filename, _, _ = su.create_filenames(years=years)

# Run simulation
run_button = st.button("Run Tournament Simulation")

if run_button:
    st.subheader("Simulation Results")

    with st.spinner("Simulating tournament..."):

        ratings = None

        if simulation_method != "Seed (Chalk)":
            # Create data frame for valid teams in the current season that can be used for tournament simulation
            score_df = ru.create_score_df(years=years)

        if simulation_method == "Massey Ratings":
            ratings = ru.calculate_massey_ratings(
                score_df=score_df, debug=False)
        elif simulation_method == "Colley Ratings":
            ratings = ru.calculate_colley_ratings(
                score_df=score_df, debug=False)
        elif simulation_method == "Elo Ratings":
            ratings = ru.calculate_elo_ratings(
                score_df=score_df, K=30, debug=False, adjust_K=False)
        elif simulation_method == "Adj Elo Ratings":
            ratings = ru.calculate_elo_ratings(
                score_df=score_df, K=30, debug=False, adjust_K=True)
        elif simulation_method == "SRS Ratings":
            ratings = ru.compile_srs_ratings(
                filename=filename, debug=False)

        _, _, tourney_dict, results = ru.simulate_tournament(filename=tournament_filename,
                                                             ratings=ratings)

    st.success("Simulation complete!")

    st.markdown("### Tournament Results")
    st.markdown(f"#### Simulation Method: {simulation_method}")
    st.markdown(f"#### Seasons: {years}")
    st.markdown(results.replace("\n", "  \n"))  # Replace newlines with streamlit-friendly newlines
