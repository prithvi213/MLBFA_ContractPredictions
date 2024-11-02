import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import re

# Split Position by Slash Token
def split_positions(position):
    positions = re.split(r'\/', position[0])
    return [pos for pos in positions if pos]

# Create the dataset using Pandas
previous_free_agents = pd.read_csv('data/export_table#table_11_2_2024.csv')

# Cleaning the training dataset
previous_free_agents = previous_free_agents.drop(0)
previous_free_agents = previous_free_agents.drop(columns=['TEAMFROM', 'TEAMTO', 'WAR', 'YOE', 'ARMBAT/THROW'])

index_of_player1972 = previous_free_agents[previous_free_agents['RANK'] == 'PLAYER (1972)'].index[0]
previous_free_agents = previous_free_agents.iloc[:index_of_player1972 - 1]
previous_free_agents.dropna(subset=['POS'], inplace=True)
previous_free_agents[['YEAR', 'YRS']] = previous_free_agents[['YEAR', 'YRS']].astype(int)
previous_free_agents.fillna('$0', inplace=True)
previous_free_agents[['VALUE', 'AAV']] = previous_free_agents[['VALUE', 'AAV']].replace({'\$': '', ',': ''}, regex=True).astype(int)
previous_free_agents = previous_free_agents[(previous_free_agents['YRS'] > 0) &
                                            (previous_free_agents['VALUE'] > 0) &
                                            (previous_free_agents['AAV'] > 0) &
                                            (previous_free_agents['YEAR'] > 2011)]

previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos: [pos])
previous_free_agents['POS'] = previous_free_agents['POS'].apply(split_positions)
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: list(set(pos_list)))
previous_free_agents['POS'][1] = ['TWP']

previous_free_agents['AGE'] = pd.to_numeric(previous_free_agents['AGE'], errors='coerce')
previous_free_agents['AGE'] = np.round(previous_free_agents['AGE'] - 1)
previous_free_agents['AGE'] = previous_free_agents['AGE'].astype(int)

print(previous_free_agents)

current_free_agents = pd.read_csv('data/export_table#table_11_1_2024.csv')
current_free_agents = current_free_agents.drop(columns=['YOE', 'ARMBAT/THROW', 'TEAM', 'PREV AAV', 'TYPE', 'MARKET VALUE', 'WAR'])
current_free_agents['AGE'] = pd.to_numeric(current_free_agents['AGE'], errors='coerce')
current_free_agents['AGE'] = np.round(current_free_agents['AGE'] + 0.2)
current_free_agents['AGE'] = current_free_agents['AGE'].astype(int)


print(current_free_agents)
player_names = list(current_free_agents['PLAYER (50)'])
st.title("MLB Free Agent Contract Predictor")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
