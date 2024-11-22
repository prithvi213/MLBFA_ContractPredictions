import streamlit as st
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('./db/fa-contracts.db')
cursor = conn.cursor()

# Select all the player contracts from the newly calculated contracts db
cfa_contracts = pd.read_sql_query("SELECT * FROM fa_contracts;", conn)
player_names = list(cfa_contracts['PLAYER'])

st.title("MLB Free Agent Contract Projector")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
index = player_names.index(player)

# Select particular statistics and write them to the screen for each case
if cfa_contracts.loc[index, 'POS'] == 'SP':
    query = "SELECT * FROM curr_starters WHERE PLAYER = ?"
    cursor.execute(query, (player,))
    stats = cursor.fetchone()
    st.write('Total Games Played: ', stats[4])
    st.write('Total Game Starts: ', stats[5])
    st.write('Total Innings Pitched: ', stats[6])
    st.write('Total fWAR: ', stats[7])
    st.write('Weighted Average ERA: ', stats[8])
    st.write('Weighted Average SIERA: ', stats[9])
    st.write('Weighted Average FIP: ', stats[10])
    st.write('Weighted Average fWAR: ', stats[11])
    st.write('Weighted Average K/9: ', stats[12])
    st.write('Weighted Average K/BB Ratio: ', stats[13])
    st.write('Weighted Average K%: ', 100 * stats[15])
    
elif cfa_contracts.loc[index, 'POS'] == 'RP':
    query = "SELECT * FROM curr_relievers WHERE PLAYER = ?"
    cursor.execute(query, (player,))
    stats = cursor.fetchone()
    st.write('Total Games Played: ', stats[4])
    st.write('Total Innings Pitched: ', stats[6])
    st.write('Total fWAR: ', stats[7])
    st.write('Weighted Average ERA: ', stats[8])
    st.write('Weighted Average FIP: ', stats[10])
    st.write('Weighted Average fWAR: ', stats[11])
    st.write('Weighted Average K/9: ', stats[12])
    st.write('Weighted Average K-BB%: ', stats[14])
    st.write('Weighted Average K%: ', round(100 * stats[15], 1))
    st.write('Weighted Average Saves: ', stats[16])
else:
    query = "SELECT * FROM curr_batters WHERE PLAYER = ?"
    cursor.execute(query, (player,))
    stats = cursor.fetchone()
    st.write('Total Games Played: ', stats[4])
    st.write('Total Plate Appearances: ', stats[5])
    st.write('Weighted Average HRs: ', int(stats[6]))
    st.write('Weighted Average wRC+: ', stats[7])
    st.write('Weighted Average Off: ', stats[8])
    st.write('Weighted Average Def: ', stats[9])
    st.write('Total fWAR: ', stats[10])
    st.write('Weighted Average fWAR: ', stats[11])

# Get the values of projected years and contract value
years, value = cfa_contracts.loc[index, 'YRS'], cfa_contracts.loc[index, 'VALUE']
format_value = f"{value:,}"

# Write it to screen and close connection
st.write('PROJECTED CONTRACT: ', str(years), ' YEAR(S), $', format_value)
conn.close()