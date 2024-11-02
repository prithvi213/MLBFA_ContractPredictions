import streamlit as st
import pandas as pd
from pybaseball import fangraphs

# Create the dataset using Pandas
free_agents = pd.read_csv('data/export_table#table_11_1_2024.csv')
player_names = list(free_agents['PLAYER (50)'])
st.title("MLB Free Agent Contract Predictor")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
