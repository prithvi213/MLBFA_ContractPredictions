import streamlit as st
import sqlite3
import pandas as pd

conn = sqlite3.connect('/Users/prithvia05/Desktop/MLBFA_ContractPredictions/db/fa-contracts.db')
conn.cursor()

current_free_agents = pd.read_sql_query("SELECT * FROM fa_contracts;", conn)
player_names = list(current_free_agents['PLAYER'])

st.title("MLB Free Agent Contract Projector")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
index = player_names.index(player)
years, value = current_free_agents.loc[index, 'YRS'], current_free_agents.loc[index, 'VALUE']
format_value = f"{value:,}"
st.write(str(years), ' YEAR(S), $', format_value)