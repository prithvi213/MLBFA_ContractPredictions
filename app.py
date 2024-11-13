import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime
from pybaseball import playerid_lookup
import matplotlib.pyplot as plt

conn = sqlite3.connect('/Users/prithvia05/Desktop/MLBFA_ContractPredictions/db/fa-contracts.db')
conn.cursor()

previous_free_agents = pd.read_csv('data/current-free-agents.csv')
batting_stats = pd.read_csv('data/batting-leaderboards.csv')
pitching_stats = pd.read_csv('data/basic-pitching-stats.csv')
advanced_stats = pd.read_csv('data/advanced-pitching-stats.csv')
pitching_stats = pitching_stats.merge(advanced_stats, on=['Season', 'Name', 'Team'], how='left', suffixes=('', '_dup'))
pitching_stats = pitching_stats.loc[:, ~pitching_stats.columns.str.endswith('_dup')]
current_free_agents = pd.read_csv('data/2024_25_free_agents.csv')

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

# Split Position by Slash Token
def split_positions(position):
    positions = re.split(r'\/', position[0])
    return [pos for pos in positions if pos]

def filter_positions(position):
    return [re.sub(r'SP\d+', 'SP', pos) for pos in position]

def calculate_batting_stats(batting_stats, player, year):
    player_stats = batting_stats[(batting_stats['Name'] == player) & (batting_stats['Season'].isin([year - 2, year - 1]))]
    player_stats = player_stats[['Season', 'G', 'PA', 'wRC+', 'Off', 'Def', 'WAR']]

    if len(player_stats) == 0:
        return pd.Series([0 for i in range(7)])

    player_stats.index = [0] if len(player_stats) == 1 else [0, 1]
    index_2020 = player_stats.index[player_stats['Season'] == 2020].tolist()

    if len(index_2020) > 0:
        player_stats.at[index_2020[0], 'G'] = round(player_stats.at[index_2020[0], 'G'] * 162 / 60)
        player_stats.at[index_2020[0], 'PA'] = round(player_stats.at[index_2020[0], 'PA'] * 162 / 60)
        player_stats.at[index_2020[0], 'Off'] = round(player_stats.at[index_2020[0], 'Off'] * 162 / 60)
        player_stats.at[index_2020[0], 'Def'] = round(player_stats.at[index_2020[0], 'Def'] * 162 / 60)
        player_stats.at[index_2020[0], 'WAR'] = round(player_stats.at[index_2020[0], 'WAR'] * 162 / 60)

    tot_GP = round(player_stats['G'].sum())
    tot_PA = round(player_stats['PA'].sum())
    tot_BATWAR = round(player_stats['WAR'].sum(), 1)

    if len(player_stats) == 1:
        avg_WRC_PLUS = round(player_stats.loc[0, 'wRC+'])
        avg_OFF = round(player_stats.loc[0, 'Off'], 1)
        avg_DEF = round(player_stats.loc[0, 'Def'], 1)
        avg_BATWAR = round(player_stats.loc[0, 'WAR'], 1)
    else:
        avg_WRC_PLUS = round(((0.25*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'wRC+'])) + (0.75*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'wRC+']))) / ((0.25*(player_stats.loc[0, 'PA'])) + (0.75*(player_stats.loc[1, 'PA']))))
        avg_OFF = round(((0.25*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'Off'])) + (0.75*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'Off']))) / ((0.25*(player_stats.loc[0, 'PA'])) + (0.75*(player_stats.loc[1, 'PA']))), 1)
        avg_DEF = round(((0.25*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'Def'])) + (0.75*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'Def']))) / ((0.25*(player_stats.loc[0, 'PA'])) + (0.75*(player_stats.loc[1, 'PA']))), 1)
        avg_BATWAR = round((0.25*((player_stats.loc[0, 'PA'] * player_stats.loc[0, 'WAR'])) + (0.75*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'WAR']))) / ((0.25*(player_stats.loc[0, 'PA'])) + (0.75*(player_stats.loc[1, 'PA']))), 1)
    
    return pd.Series([tot_GP, tot_PA, avg_WRC_PLUS, avg_OFF, avg_DEF, tot_BATWAR, avg_BATWAR])

def calculate_pitching_stats(pitching_stats, player, year):
    player_stats = pitching_stats[(pitching_stats['Name'] == player) & (pitching_stats['Season'].isin([year - 2, year - 1]))]
    player_stats = player_stats[['Season', 'G', 'GS', 'IP', 'K/9', 'BB/9', 'ERA', 'WAR', 'FIP', 'SIERA', 'K%', 'K-BB%', 'SV', 'PlayerId']]

    if len(player_stats) == 0:
        return pd.Series([0 for i in range(11)])
    
    if player == 'Luis Garcia':
        player_stats = player_stats[player_stats['PlayerId'] == 6984]

    player_stats.index = [0] if len(player_stats) == 1 else [0, 1]

    index_2020 = player_stats.index[player_stats['Season'] == 2020].tolist()
    player_stats['Decimal'] = player_stats['IP'] - np.floor(player_stats['IP'])
    player_stats['IP'] = np.floor(player_stats['IP']) + (player_stats['Decimal'] / 0.3)

    if len(index_2020) > 0:
        player_stats.at[index_2020[0], 'G'] = round(player_stats.at[index_2020[0], 'G'] * 162 / 60)
        player_stats.at[index_2020[0], 'GS'] = round(player_stats.at[index_2020[0], 'GS'] * 162 / 60)
        player_stats.at[index_2020[0], 'IP'] = round(player_stats.at[index_2020[0], 'IP'] * 162 / 60)
        player_stats.at[index_2020[0], 'SV'] = round(player_stats.at[index_2020[0], 'SV'] * 162 / 60)
        player_stats.at[index_2020[0], 'WAR'] = round(player_stats.at[index_2020[0], 'WAR'] * 162 / 60)
    
    tot_GP = round(player_stats['G'].sum())
    tot_GS = round(player_stats['GS'].sum())
    tot_IP = round(player_stats['IP'].sum(), 1)
    tot_PITCHWAR = round(player_stats['WAR'].sum(), 1)

    if len(player_stats) == 1:
        avg_SIERA = round(player_stats.loc[0, 'SIERA'], 2)
        avg_FIP = round(player_stats.loc[0, 'FIP'], 2)
        avg_PITCHWAR = round(player_stats.loc[0, 'WAR'], 1)

        if player_stats.loc[0, 'BB/9'] > 0:
            avg_K_PER_BB = round(player_stats.loc[0, 'K/9'] / player_stats.loc[0, 'BB/9'], 1)
        else:
            avg_K_PER_BB = round(player_stats.loc[0, 'K/9'] / 9, 1)

        avg_K_PER_9 = round(player_stats.loc[0, 'K/9'], 1)
        avg_K_MINUS_BB = round(player_stats.loc[0, 'K-BB%'], 1)
        avg_K_RATE = round(player_stats.loc[0, 'K%'], 1)
        avg_SAVES = round(player_stats.loc[0, 'SV'])
        avg_ERA = round(player_stats.loc[0, 'ERA'])
    else:
        avg_ERA = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'ERA']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'ERA'])) / tot_IP, 2)
        avg_SIERA = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'SIERA']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'SIERA'])) / tot_IP, 2)
        avg_FIP = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'FIP']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'FIP'])) / tot_IP, 2)
        avg_PITCHWAR = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'WAR']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'WAR'])) / tot_IP, 1)
        avg_K_PER_9 = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K/9']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K/9'])) / tot_IP, 1)

        if player_stats.loc[0, 'BB/9'] > 0 and player_stats.loc[1, 'BB/9'] > 0:
            avg_K_PER_BB = round(((player_stats.loc[0, 'IP'] * (player_stats.loc[0, 'K/9'] / player_stats.loc[0, 'BB/9'])) + (player_stats.loc[1, 'IP'] * (player_stats.loc[1, 'K/9'] / player_stats.loc[1, 'BB/9']))) / tot_IP, 1)
        else:
            avg_K_PER_BB = round(player_stats.loc[0, 'K/9'] / 9, 1)

        avg_K_MINUS_BB = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K-BB%']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K-BB%'])) / tot_IP, 1)
        avg_K_RATE = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K%']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K%'])) / tot_IP, 1)
        avg_SAVES = round(((player_stats.loc[0, 'IP'] * player_stats.loc[0, 'SV']) + (player_stats.loc[1, 'IP'] * player_stats.loc[1, 'SV'])) / tot_IP)
    
    return pd.Series([tot_GP, tot_GS, tot_IP, tot_PITCHWAR, avg_ERA, avg_SIERA, avg_FIP, avg_PITCHWAR, avg_K_PER_9, avg_K_PER_BB, avg_K_MINUS_BB, avg_K_RATE, avg_SAVES])

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
                                            (previous_free_agents['TEAMFROM'] != 'CUB') &
                                            (previous_free_agents['YEAR'] > 2011)]
previous_free_agents = previous_free_agents.drop(columns=['TEAMFROM'])

previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].astype('string')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('Á', 'A')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('á', 'a')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('é', 'e')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('í', 'i')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ó', 'o')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ú', 'u')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace('ñ', 'n')
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace(r'\.(.)\.', r'\1', regex=True)
previous_free_agents['PLAYER (2000)'] = previous_free_agents['PLAYER (2000)'].str.replace(' Jr.', '')

previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos: [pos])
previous_free_agents['POS'] = previous_free_agents['POS'].apply(split_positions)
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: list(pos_list))
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: [re.sub(r'SP\d+', 'SP', pos) for pos in pos_list])
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: [re.sub('OF', 'RF', pos) for pos in pos_list])
previous_free_agents['POS'] = previous_free_agents['POS'].apply(lambda pos_list: pos_list[0])
previous_free_agents.loc[1, 'POS'] = 'TWP'

previous_free_agents['AGE'] = pd.to_numeric(previous_free_agents['AGE'], errors='coerce')
previous_free_agents['AGE'] = np.round(previous_free_agents['AGE'] - 1)
previous_free_agents['AGE'] = previous_free_agents['AGE'].astype(int)

previous_free_agents['VALUE'] = previous_free_agents['VALUE'] * (4.98 / previous_free_agents['YEAR'].apply(lambda c: lg_avg.get(c - 1)))
previous_free_agents['VALUE'] = np.round(previous_free_agents['VALUE'])
previous_free_agents['VALUE'] = previous_free_agents['VALUE'].astype(float)

previous_free_agents['AAV'] = previous_free_agents['VALUE'] / previous_free_agents['YRS']
previous_free_agents['AAV'] = np.round(previous_free_agents['AAV'])
previous_free_agents['AAV'] = previous_free_agents['AAV'].astype(float)

batting_stats['Name'] = batting_stats['Name'].astype('string')
batting_stats['Name'] = batting_stats['Name'].str.replace('Á', 'A')
batting_stats['Name'] = batting_stats['Name'].str.replace('á', 'a')
batting_stats['Name'] = batting_stats['Name'].str.replace('é', 'e')
batting_stats['Name'] = batting_stats['Name'].str.replace('í', 'i')
batting_stats['Name'] = batting_stats['Name'].str.replace('ó', 'o')
batting_stats['Name'] = batting_stats['Name'].str.replace('ú', 'u')
batting_stats['Name'] = batting_stats['Name'].str.replace('ñ', 'n')
batting_stats['Name'] = batting_stats['Name'].str.replace(' Jr.', '')
batting_stats['Name'] = batting_stats['Name'].str.replace(r'\.(.)\.', r'\1', regex=True)

pitching_stats['Name'] = pitching_stats['Name'].astype('string')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('Á', 'A')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('á', 'a')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('é', 'e')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('í', 'i')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('ó', 'o')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('ú', 'u')
pitching_stats['Name'] = pitching_stats['Name'].str.replace('ñ', 'n')
pitching_stats['Name'] = pitching_stats['Name'].str.replace(' Jr.', '')
pitching_stats['Name'] = pitching_stats['Name'].str.replace(r'\.(.)\.', r'\1', regex=True)

previous_free_agents.to_sql('prev_free_agents', conn, if_exists='replace', index=False)
batter_free_agents = pd.read_sql_query("SELECT * FROM prev_free_agents WHERE POS NOT IN ('SP', 'RP')", conn)
pitcher_free_agents = pd.read_sql_query("SELECT * FROM prev_free_agents WHERE POS IN ('SP', 'RP', 'TWP')", conn)

batter_free_agents[['TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']] = batter_free_agents.apply(
    lambda row: calculate_batting_stats(batting_stats, row['PLAYER (2000)'], row['YEAR']), axis=1
)

value = batter_free_agents.at[0, 'VALUE']
batter_free_agents.at[0, 'VALUE'] = (0.5 * ((10.2 / 18.2) + (6.5 / 8.9))) * value
batter_free_agents.at[0, 'AAV'] = batter_free_agents.at[0, 'VALUE'] / batter_free_agents.at[0, 'YRS']
batter_free_agents['VALUE'] = batter_free_agents['VALUE'].astype(int)
batter_free_agents['AAV'] = batter_free_agents['AAV'].astype(int)
batter_free_agents['TOT_GP (2 Yrs)'] = batter_free_agents['TOT_GP (2 Yrs)'].astype(int)
batter_free_agents['TOT_PA (2 Yrs)'] = batter_free_agents['TOT_PA (2 Yrs)'].astype(int)
batter_free_agents['AVG_wRC+ (2 Yrs)'] = batter_free_agents['AVG_wRC+ (2 Yrs)'].astype(int)

pitcher_free_agents[['TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'Avg_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/9 (2 Yrs)', 'AVG_K/BB (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']] = pitcher_free_agents.apply(
    lambda row: calculate_pitching_stats(pitching_stats, row['PLAYER (2000)'], row['YEAR']), axis=1
)

value = pitcher_free_agents.at[0, 'VALUE']
pitcher_free_agents.at[0, 'VALUE'] = (0.5 * ((8.0 / 18.2) + (2.4 / 8.9))) * value
pitcher_free_agents.at[0, 'AAV'] = pitcher_free_agents.at[0, 'VALUE'] / pitcher_free_agents.at[0, 'YRS']
pitcher_free_agents['VALUE'] = pitcher_free_agents['VALUE'].astype(int)
pitcher_free_agents['AAV'] = pitcher_free_agents['AAV'].astype(int)
pitcher_free_agents['TOT_GP (2 Yrs)'] = pitcher_free_agents['TOT_GP (2 Yrs)'].astype(int)
pitcher_free_agents['TOT_GS (2 Yrs)'] = pitcher_free_agents['TOT_GS (2 Yrs)'].astype(int)
pitcher_free_agents['AVG_SAVES (2 Yrs)'].fillna(0.0, inplace=True)
pitcher_free_agents['AVG_SAVES (2 Yrs)'] = pitcher_free_agents['AVG_SAVES (2 Yrs)'].astype(int)

#print(batter_free_agents)
#print(pitcher_free_agents)

current_free_agents = current_free_agents.drop(columns=['YOE', 'ARMBAT/THROW', 'TEAM', 'PREV AAV', 'TYPE', 'MARKET VALUE', 'WAR'])
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].astype('string')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('Á', 'A')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('á', 'a')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('é', 'e')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('í', 'i')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('ó', 'o')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('ú', 'u')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace('ñ', 'n')
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace(r'\.(.)\.', r'\1', regex=True)
current_free_agents['PLAYER (198)'] = current_free_agents['PLAYER (198)'].str.replace(' Jr.', '')
current_free_agents['POS'] = current_free_agents['POS'].apply(lambda pos: str(pos))
current_free_agents['AGE'] = pd.to_numeric(current_free_agents['AGE'], errors='coerce')
current_free_agents['AGE'] = np.round(current_free_agents['AGE'] + 0.2)
current_free_agents['AGE'] = current_free_agents['AGE'].astype(int)

current_free_agents.to_sql('curr_free_agents', conn, if_exists='replace', index=False)
current_batter_agents = pd.read_sql_query("SELECT * FROM curr_free_agents WHERE POS NOT IN ('SP', 'RP')", conn)
current_batter_agents['YEAR'] = 2025
current_pitcher_agents = pd.read_sql_query("SELECT * FROM curr_free_agents WHERE POS IN ('SP', 'RP')", conn)
current_pitcher_agents['YEAR'] = 2025

current_batter_agents[['TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']] = current_batter_agents.apply(
    lambda row: calculate_batting_stats(batting_stats, row['PLAYER (198)'], 2025), axis=1
)

current_batter_agents['TOT_GP (2 Yrs)'] = current_batter_agents['TOT_GP (2 Yrs)'].astype(int)
current_batter_agents['TOT_PA (2 Yrs)'] = current_batter_agents['TOT_PA (2 Yrs)'].astype(int)
current_batter_agents['AVG_wRC+ (2 Yrs)'] = current_batter_agents['AVG_wRC+ (2 Yrs)'].astype(int)


current_pitcher_agents[['TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'Avg_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/9 (2 Yrs)', 'AVG_K/BB (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']] = current_pitcher_agents.apply(
    lambda row: calculate_pitching_stats(pitching_stats, row['PLAYER (198)'], 2025), axis=1
)

current_pitcher_agents['TOT_GP (2 Yrs)'] = current_pitcher_agents['TOT_GP (2 Yrs)'].astype(int)
current_pitcher_agents['TOT_GS (2 Yrs)'] = current_pitcher_agents['TOT_GS (2 Yrs)'].astype(int)
current_pitcher_agents['AVG_SAVES (2 Yrs)'].fillna(0.0, inplace=True)
current_pitcher_agents = current_pitcher_agents[current_pitcher_agents['TOT_GP (2 Yrs)'] > 0]
current_pitcher_agents['AVG_SAVES (2 Yrs)'] = current_pitcher_agents['AVG_SAVES (2 Yrs)'].astype(int)

#print(previous_free_agents.head(50))
print(batter_free_agents.head(50))
#print(pitcher_free_agents.head(50))
print(current_batter_agents.head(50))
#print(current_pitcher_agents.head(50))
#print(current_batter_agents.tail(50))
#print(current_pitcher_agents.tail(50))
#print(current_batter_agents.head(50))

batter_free_agents = batter_free_agents.sample(frac=1, random_state=42).reset_index(drop=True)
current_batter_agents = current_batter_agents.sample(frac=1, random_state=42).reset_index(drop=True)

starter_free_agents = pitcher_free_agents[pitcher_free_agents['POS'] == 'SP']
starter_free_agents = starter_free_agents.sample(frac=1, random_state=42).reset_index(drop=True)
current_starter_agents = current_pitcher_agents[current_pitcher_agents['POS'] == 'SP']
current_starter_agents = current_starter_agents.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bat = batter_free_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']]

y_train_bat = batter_free_agents[['YRS', 'AAV']]
X_test_bat = current_batter_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']]

conditions = [
    (X_train_bat['AGE'] < 27) & (X_train_bat['AVG_WAR (2 Yrs)'] > 5.0),
    (X_train_bat['AGE'] < 30) & (X_train_bat['AVG_WAR (2 Yrs)'] > 4.0),
    (X_train_bat['AGE'] > 30) & (X_train_bat['AVG_WAR (2 Yrs)'] > 3.0)
]

weights = [2.5, 2.0, 1.5]
sample_weights = np.select(conditions, weights, default=1.0)

#X_train_pitch = starter_free_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/BB (2 Yrs)']]
#y_train_pitch = starter_free_agents[['YRS', 'VALUE']]
#X_test_pitch = current_starter_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/BB (2 Yrs)']]

model_1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_1.fit(X_train_bat, y_train_bat['YRS'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_1 = model_1.predict(X_test_bat)

#X_train_bat['AVG_wRC+ (2 Yrs)'] *= 1.1

conditions = [
    (X_train_bat['AVG_WAR (2 Yrs)'] > 5.0),
    (X_train_bat['AGE'] < 30) & (X_train_bat['AVG_DEF (2 Yrs)'] > 7.5) & (X_train_bat['AVG_wRC+ (2 Yrs)'] > 100),
    (X_train_bat['AVG_wRC+ (2 Yrs)'] > 125),
    (X_train_bat['AVG_wRC+ (2 Yrs)'] < 100)
]

weights = [2.5, 2.0, 1.5, 0.9]
sample_weights = np.select(conditions, weights)

# Train the model for target_column_2
model_2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_2.fit(X_train_bat, y_train_bat['AAV'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_2 = model_2.predict(X_test_bat)

# Combine the predictions
predictions = pd.DataFrame({'YRS': predictions_1, 'AAV': predictions_2})

importance_scores = model_1.feature_importances_
feature_names = X_train_bat.columns
feature_importance_dict = dict(zip(feature_names, importance_scores))

# Print feature importance
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

"""model_3 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.02)
model_3.fit(X_train_pitch, y_train_pitch['YRS'])

# Make predictions on the test set
predictions_3 = model_3.predict(X_test_pitch)

# Train the model for target_column_2
model_4 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.02)
model_4.fit(X_train_pitch, y_train_pitch['VALUE'])

# Make predictions on the test set
predictions_4 = model_4.predict(X_test_pitch)

# Combine the predictions
predictions = pd.DataFrame({'YRS': predictions_3, 'VALUE': predictions_4})"""

for idx, (v1, v2) in enumerate(zip(predictions_1, predictions_2)):
    print(current_batter_agents.loc[idx, 'PLAYER (198)'], round(v1), round(v2))

#for idx, (v3, v4) in enumerate(zip(predictions_3, predictions_4)):
#    print(current_starter_agents.loc[idx, 'PLAYER (198)'], round(v3), round(v4))

player_names = list(current_free_agents['PLAYER (198)'])

st.title("MLB Free Agent Contract Predictor")

# Player Dropdown for Free Agent
player = st.selectbox('Select a FA', player_names)
