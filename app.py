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

# Run simulation
run_button = st.button("Run Tournament Simulation")

if run_button:
    st.subheader("Simulation Results")

    with st.spinner("Simulating tournament..."):
        # Placeholder for your real logic
        # results = simulate_tournament(
        #     method=simulation_method,
        #     n_sims=num_sims,
        #     seed=random_seed
        # )

        # Fake output for now
        results = {
            "Champion": "UConn",
            "Final Four": ["UConn", "Houston", "Purdue", "Tennessee"],
            "Avg Upsets per Bracket": 6.3
        }

    st.success("Simulation complete!")

    st.markdown("### 🏆 Champion")
    st.write(results["Champion"])

    st.markdown("### 🔥 Final Four")
    st.write(results["Final Four"])

    st.markdown("### 📊 Summary Stats")
    st.metric(
        label="Avg Upsets per Bracket",
        value=results["Avg Upsets per Bracket"]
    )