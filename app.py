import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pybaseball import fg_batting_data, fg_pitching_data
import numpy as np
import re

avg_GP_list = []
avg_PA_list = []
avg_WRC_PLUS_list = []
avg_OFF_list = []
avg_DEF_list = []
tot_BATWAR_list = []
avg_BATWAR_list = []

# Split Position by Slash Token
def split_positions(position):
    positions = re.split(r'\/', position[0])
    return [pos for pos in positions if pos]

def calculate_avg_stats(season_stats, player, year, positions):
    if 'SP' not in positions and 'RP' not in positions:
        player_stats = season_stats[(season_stats['Name'] == player) & (season_stats['Season'].isin([year - 2, year - 1]))]
        player_stats = player_stats[['Season', 'G', 'PA', 'wRC+', 'Off', 'Def', 'WAR']]

        if len(player_stats) == 0:
            return pd.Series([0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
    
        player_stats.index = [0] if len(player_stats) == 1 else [0, 1]
        index_2020 = player_stats.index[player_stats['Season'] == 2020].tolist()

        if len(index_2020) > 0:
            player_stats.at[index_2020[0], 'G'] = round(player_stats.at[index_2020[0], 'G'] * 162 / 60)
            player_stats.at[index_2020[0], 'PA'] = round(player_stats.at[index_2020[0], 'PA'] * 162 / 60)
            player_stats.at[index_2020[0], 'Off'] = round(player_stats.at[index_2020[0], 'Off'] * 162 / 60)
            player_stats.at[index_2020[0], 'Def'] = round(player_stats.at[index_2020[0], 'Def'] * 162 / 60)
            player_stats.at[index_2020[0], 'WAR'] = round(player_stats.at[index_2020[0], 'WAR'] * 162 / 60)

        if len(player_stats) == 1:
            avg_WRC_PLUS = round(player_stats.loc[0, 'wRC+'])
            avg_BATWAR = round(player_stats.loc[0, 'WAR'], 1)
        else:
            avg_WRC_PLUS = round(((player_stats.loc[0, 'PA'] * player_stats.loc[0, 'wRC+']) + (player_stats.loc[1, 'PA'] * player_stats.loc[1, 'wRC+'])) / player_stats['PA'].sum())
            avg_BATWAR = round(((player_stats.loc[0, 'PA'] * player_stats.loc[0, 'WAR']) + (player_stats.loc[1, 'PA'] * player_stats.loc[1, 'WAR'])) / player_stats['PA'].sum(), 1)

        tot_GP = round(player_stats['G'].sum())
        tot_PA = round(player_stats['PA'].sum())
        avg_OFF = round(player_stats['Off'].mean(), 1)
        avg_DEF = round(player_stats['Def'].mean(), 1)
        total_BATWAR = round(player_stats['WAR'].sum(), 1)
        return pd.Series([tot_GP, tot_PA, avg_WRC_PLUS, avg_OFF, avg_DEF, total_BATWAR, avg_BATWAR])

# League average MLB Salary in millions to help adjust for inflation
lg_avg = {
    2009: 3,
    2010: 3.01,
    2011: 3.1,
    2012: 3.21,
    2013: 3.39,
    2014: 3.69,
    2015: 3.84,
    2016: 4.38,
    2017: 4.45,
    2018: 4.41,
    2019: 4.38,
    2020: 4.43,
    2021: 4.17,
    2022: 4.41,
    2023: 4.9
}

# Create the dataset using Pandas
previous_free_agents = pd.read_csv('data/export_table#table_11_2_2024.csv', encoding='utf-8')

# Cleaning the training dataset
previous_free_agents = previous_free_agents.drop(0)
previous_free_agents = previous_free_agents.drop(columns=['TEAMTO', 'WAR', 'YOE', 'ARMBAT/THROW'])

index_of_player1972 = previous_free_agents[previous_free_agents['RANK'] == 'PLAYER (1972)'].index[0]
previous_free_agents = previous_free_agents.drop(columns=['RANK'])
previous_free_agents = previous_free_agents.iloc[:index_of_player1972 - 1]
previous_free_agents.dropna(subset=['POS'], inplace=True)
previous_free_agents[['YEAR', 'YRS']] = previous_free_agents[['YEAR', 'YRS']].astype(int)
previous_free_agents.fillna('$0', inplace=True)
previous_free_agents[['VALUE', 'AAV']] = previous_free_agents[['VALUE', 'AAV']].replace({'\$': '', ',': ''}, regex=True).astype(float)
previous_free_agents = previous_free_agents[(previous_free_agents['YRS'] > 0) &
                                            (previous_free_agents['VALUE'] > 0) &
                                            (previous_free_agents['AAV'] > 0) &
                                            (previous_free_agents['TEAMFROM'] != 'JPN') &
                                            (previous_free_agents['TEAMFROM'] != 'KOR') &
                                            (previous_free_agents['TEAMFROM'] != 'CUB')]
previous_free_agents = previous_free_agents.drop(columns=['TEAMFROM'])

previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].astype('string')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('é', 'e')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('á', 'a')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ó', 'o')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ú', 'u')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('í', 'i')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('Á', 'A')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ñ', 'n')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace(r'\.(.)\.', r'\1', regex=True)
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace(' Jr.', '')

previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos: [pos])
previous_free_agents['POS'] = previous_free_agents['POS'].apply(split_positions)
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: list(set(pos_list)))
previous_free_agents.loc[1, 'POS'] = ['TWP']

previous_free_agents['AGE'] = pd.to_numeric(previous_free_agents['AGE'], errors='coerce')
previous_free_agents['AGE'] = np.round(previous_free_agents['AGE'] - 1)
previous_free_agents['AGE'] = previous_free_agents['AGE'].astype(int)

previous_free_agents['VALUE'] = previous_free_agents['VALUE'] * (4.98 / previous_free_agents['YEAR'].apply(lambda c: lg_avg.get(c - 1)))
previous_free_agents['VALUE'] = np.round(previous_free_agents['VALUE'])
previous_free_agents['VALUE'] = previous_free_agents['VALUE'].astype(int)

previous_free_agents['AAV'] = previous_free_agents['VALUE'] / previous_free_agents['YRS']
previous_free_agents['AAV'] = np.round(previous_free_agents['AAV'])
previous_free_agents['AAV'] = previous_free_agents['AAV'].astype(int)

#print(previous_free_agents)

season_stats = pd.read_csv('data/fangraphs-leaderboards.csv')
season_stats = season_stats[['Season', 'Name', 'G', 'PA', 'wRC+', 'Off', 'Def', 'WAR']]
season_stats['Name'] = season_stats['Name'].astype('string')
season_stats['Name'] = season_stats['Name'].str.replace('é', 'e')
season_stats['Name'] = season_stats['Name'].str.replace('á', 'a')
season_stats['Name'] = season_stats['Name'].str.replace('ó', 'o')
season_stats['Name'] = season_stats['Name'].str.replace('ú', 'u')
season_stats['Name'] = season_stats['Name'].str.replace('í', 'i')
season_stats['Name'] = season_stats['Name'].str.replace('Á', 'A')
season_stats['Name'] = season_stats['Name'].str.replace('ñ', 'n')
season_stats['Name'] = season_stats['Name'].str.replace(' Jr.', '')
season_stats['Name'] = season_stats['Name'].str.replace(r'\.(.)\.', r'\1', regex=True)
season_stats['Name'] = season_stats['Name'].str.replace('BJ', 'Melvin')

previous_free_agents[['TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_Off (2 Yrs)', 'AVG_Def (2 Yrs)', 'Tot_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']] = previous_free_agents.apply(
    lambda row: calculate_avg_stats(season_stats, row['PLAYER (2000)'], row['YEAR'], row['POS']), axis=1
)

print(previous_free_agents)

current_free_agents = pd.read_csv('data/export_table#table_11_1_2024.csv')
current_free_agents = current_free_agents.drop(columns=['YOE', 'ARMBAT/THROW', 'TEAM', 'PREV AAV', 'TYPE', 'MARKET VALUE', 'WAR'])
current_free_agents['AGE'] = pd.to_numeric(current_free_agents['AGE'], errors='coerce')
current_free_agents['AGE'] = np.round(current_free_agents['AGE'] + 0.2)
current_free_agents['AGE'] = current_free_agents['AGE'].astype(int)

#print(current_free_agents)

player_names = list(current_free_agents['PLAYER (50)'])
#print(current_free_agents)

st.title("MLB Free Agent Contract Predictor")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
