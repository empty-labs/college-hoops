# Local libraries
import Tools.ratings_utils as ru
import Tools.season_utils as su


# Third party packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import xgboost as xgb


ML_OPTIONS = [
    "Logistic Regression",
    "XGBoost",
    "Random Forest"
]

# Page config
st.set_page_config(
    page_title="March Madness Bracketology",
    layout="centered"
)

# Title & description
st.title("🏀 March Madness Bracketology Simulator")
st.markdown("""
Simulate NCAA tournament brackets using machine learning models based on team ratings and point differentials.
""")

st.divider()
col1, col2 = st.columns(2)

# Choose Season Start
with col1:
    season_start = st.selectbox(
        "Season Start",
        options=su.SEASONS_STR,
        help="Choose what season to start simulation"
    )
    st.session_state.season_start = season_start

# Choose Season End
with col2:
    season_end = st.selectbox(
        "Season End",
        options=[y for y in su.SEASONS_STR if y >= st.session_state.season_start],
        help="Choose what season to end simulation"
    )

# Simulation Method
simulation_method = st.selectbox(
    "Simulation Method",
    options=ML_OPTIONS,
    help="Choose how game outcomes are determined"
)

# Convert season selections to data set filenames
start_year = su.convert_season_to_year(season_start)
end_year = su.convert_season_to_year(season_end)
years = su.year_range(start_year, end_year)
_, tournament_filename, picks_filename, ratings_filename, final_ratings_filename = su.create_filenames(years=years)
#

# Run simulation
run_button = st.button("Run Tournament Simulation")

if run_button:
    st.subheader("Simulation Results")

    with st.spinner("Simulating tournament..."):

        # Read data from JSON
        rating_score_df = pd.read_json(ratings_filename)
        # Set data frame and target variable
        df = rating_score_df.copy()
        df["y"] = (df["Winner"] == df["Home"]).astype(int)

        # Add feature columns
        # TODO Move this to mid-season ratings_per_game function
        df = ru.derive_features(df=df, final_ratings_filename=final_ratings_filename)

        # Set features
        features = ru.ML_FEATURES

        # Create X, y data frames
        X = df[features]
        y = df["y"]

        # Split train/test data sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set up final ratings for tournament
        ratings_dict = ru.compile_ratings_dict(final_ratings_filename=final_ratings_filename)

        model = None

        if simulation_method == "Logistic Regression":

            log_model = LogisticRegression()
            log_model.fit(X_train, y_train)

            model = log_model

        elif simulation_method == "XGBoost":

            # Train XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False
            )

            xgb_model.fit(X_train, y_train)
            model = xgb_model

        elif simulation_method == "Random Forest":

            # Define Random Forest model
            rf_model = RandomForestClassifier(
                n_estimators=500,  # number of trees
                max_depth=None,  # let trees go deep until pure
                min_samples_split=2,  # default
                min_samples_leaf=1,  # default
                max_features="sqrt",  # good for classification
                random_state=42,
                n_jobs=-1  # use all cores
            )

            rf_model.fit(X_train, y_train)

            model = rf_model

        _, _, tourney_dict, results = ru.simulate_tournament_with_all_ratings(
            filename=tournament_filename,
            ratings=ratings_dict,
            model=model)

    st.success("Simulation complete!")

    st.markdown("### Tournament Results")
    st.markdown(f"#### Simulation Method: {simulation_method}")
    st.markdown(f"#### Seasons: {years}")
    st.markdown(results.replace("\n", "  \n"))  # Replace newlines with streamlit-friendly newlines

    # Clean up tournament dictionary with fewer columns and revised output
    revised_tourney_df = pd.DataFrame(tourney_dict)

    # TODO: Add streamlit_utils.py to handle common formatting changes and streamlit UI
    revised_tourney_df['Team1'] = (
            "(" + revised_tourney_df['Seed1'].astype(str) + ") " + revised_tourney_df['Team1']
    )
    revised_tourney_df['Team2'] = (
            "(" + revised_tourney_df['Seed2'].astype(str) + ") " + revised_tourney_df['Team2']
    )

    revised_tourney_df['Rating1'] = (
        pd.to_numeric(revised_tourney_df['Rating1'], errors='coerce')
        .round(3)
    )
    revised_tourney_df['Rating2'] = (
        pd.to_numeric(revised_tourney_df['Rating2'], errors='coerce')
        .round(3)
    )

    revised_tourney_df = revised_tourney_df[['Round Name', 'Team1', 'Team2', 'Rating1', 'Rating2']]

    st.markdown("### Tournament Dictionary")
    st.dataframe(revised_tourney_df)