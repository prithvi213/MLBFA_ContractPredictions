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

w1, w2 = 0.25, 0.75

# League average MLB Salary in millions to help adjust for inflation
lg_avg = {
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

# Home Run Ratio
total_hrs = {
    2010: 4613,
    2011: 4552,
    2012: 4934,
    2013: 4661,
    2014: 4186,
    2015: 4909,
    2016: 5610,
    2017: 6105,
    2018: 5585,
    2019: 6776,
    2020: 6218,
    2021: 5940,
    2022: 5215,
    2023: 5868,
    2024: 5453
}

# Split Position by Slash Token
def split_positions(position):
    positions = re.split(r'\/', position[0])
    return [pos for pos in positions if pos]

def filter_positions(position):
    return [re.sub(r'SP\d+', 'SP', pos) for pos in position]

def calculate_batting_stats(batting_stats, player, year):
    player_stats = batting_stats[(batting_stats['Name'] == player) & (batting_stats['Season'].isin([year - 2, year - 1]))]
    player_stats = player_stats[['Season', 'G', 'PA', 'HR', 'wRC+', 'Off', 'Def', 'WAR']]

    if len(player_stats) == 0:
        return pd.Series([0 for i in range(7)])

    player_stats.index = [0] if len(player_stats) == 1 else [0, 1]
    index_2020 = player_stats.index[player_stats['Season'] == 2020].tolist()

    if len(index_2020) > 0:
        player_stats.at[index_2020[0], 'G'] = round(player_stats.at[index_2020[0], 'G'] * 162 / 60)
        player_stats.at[index_2020[0], 'PA'] = round(player_stats.at[index_2020[0], 'PA'] * 162 / 60)
        player_stats.at[index_2020[0], 'HR'] = round(player_stats.at[index_2020[0], 'HR'] * 162 / 60)
        player_stats.at[index_2020[0], 'Off'] = round(player_stats.at[index_2020[0], 'Off'] * 162 / 60)
        player_stats.at[index_2020[0], 'Def'] = round(player_stats.at[index_2020[0], 'Def'] * 162 / 60)
        player_stats.at[index_2020[0], 'WAR'] = round(player_stats.at[index_2020[0], 'WAR'] * 162 / 60)

    tot_GP = round(player_stats['G'].sum())
    tot_PA = round(player_stats['PA'].sum())
    tot_BATWAR = round(player_stats['WAR'].sum(), 1)

    if len(player_stats) == 1:
        s0 = player_stats.loc[0, 'Season']
        avg_HR = round(player_stats.loc[0, 'HR'] * (5453 / total_hrs[s0]))
        avg_WRC_PLUS = round(player_stats.loc[0, 'wRC+'])
        avg_OFF = round(player_stats.loc[0, 'Off'], 1)
        avg_DEF = round(player_stats.loc[0, 'Def'], 1)
        avg_BATWAR = round(player_stats.loc[0, 'WAR'], 1)
    else:
        s0, s1 = player_stats.loc[0, 'Season'], player_stats.loc[1, 'Season']
        avg_HR = round(((w1*(5453 / total_hrs[s0])*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'HR'])) + (w2*(5453 / total_hrs[s1])*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'HR']))) / ((w1*(player_stats.loc[0, 'PA'])) + (w2*(player_stats.loc[1, 'PA']))))
        avg_WRC_PLUS = round(((w1*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'wRC+'])) + (w2*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'wRC+']))) / ((w1*(player_stats.loc[0, 'PA'])) + (w2*(player_stats.loc[1, 'PA']))))
        avg_OFF = round(((w1*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'Off'])) + (w2*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'Off']))) / ((w1*(player_stats.loc[0, 'PA'])) + (w2*(player_stats.loc[1, 'PA']))), 1)
        avg_DEF = round(((w1*(player_stats.loc[0, 'PA'] * player_stats.loc[0, 'Def'])) + (w2*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'Def']))) / ((w1*(player_stats.loc[0, 'PA'])) + (w2*(player_stats.loc[1, 'PA']))), 1)
        avg_BATWAR = round((w1*((player_stats.loc[0, 'PA'] * player_stats.loc[0, 'WAR'])) + (w2*(player_stats.loc[1, 'PA'] * player_stats.loc[1, 'WAR']))) / ((w1*(player_stats.loc[0, 'PA'])) + (w2*(player_stats.loc[1, 'PA']))), 1)
    
    return pd.Series([tot_GP, tot_PA, avg_HR, avg_WRC_PLUS, avg_OFF, avg_DEF, tot_BATWAR, avg_BATWAR])

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
        avg_ERA = round(((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'ERA'])) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'ERA']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 2)
        avg_SIERA = round((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'SIERA']) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'SIERA']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 2)
        avg_FIP = round((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'FIP']) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'FIP']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 2)
        avg_PITCHWAR = round((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'WAR']) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'WAR']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 1)
        avg_K_PER_9 = round((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K/9']) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K/9']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 1)

        if player_stats.loc[0, 'BB/9'] > 0 and player_stats.loc[1, 'BB/9'] > 0:
            avg_K_PER_BB = round(((w1*(player_stats.loc[0, 'IP'] * (player_stats.loc[0, 'K/9'] / player_stats.loc[0, 'BB/9']))) + (w2*(player_stats.loc[1, 'IP'] * (player_stats.loc[1, 'K/9'] / player_stats.loc[1, 'BB/9'])))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 1)
        else:
            avg_K_PER_BB = round(player_stats.loc[0, 'K/9'] / 9, 1)

        avg_K_MINUS_BB = round(((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K-BB%'])) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K-BB%']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 1)
        avg_K_RATE = round(((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'K%'])) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'K%']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))), 1)
        avg_SAVES = round(((w1*(player_stats.loc[0, 'IP'] * player_stats.loc[0, 'SV'])) + (w2*(player_stats.loc[1, 'IP'] * player_stats.loc[1, 'SV']))) / ((w1*(player_stats.loc[0, 'IP'])) + (w2*(player_stats.loc[1, 'IP']))))
    
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
                                            (previous_free_agents['YEAR'] >= 2012)]
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

batter_free_agents[['TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_HR (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']] = batter_free_agents.apply(
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

pitcher_free_agents[['TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/9 (2 Yrs)', 'AVG_K/BB (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']] = pitcher_free_agents.apply(
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
pitcher_free_agents = pitcher_free_agents[pitcher_free_agents['PLAYER (2000)'] != 'Trevor Bauer']

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

current_batter_agents[['TOT_GP (2 Yrs)', 'TOT_PA (2 Yrs)', 'AVG_HR (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_WAR (2 Yrs)']] = current_batter_agents.apply(
    lambda row: calculate_batting_stats(batting_stats, row['PLAYER (198)'], 2025), axis=1
)

current_batter_agents['TOT_GP (2 Yrs)'] = current_batter_agents['TOT_GP (2 Yrs)'].astype(int)
current_batter_agents['TOT_PA (2 Yrs)'] = current_batter_agents['TOT_PA (2 Yrs)'].astype(int)
current_batter_agents['AVG_wRC+ (2 Yrs)'] = current_batter_agents['AVG_wRC+ (2 Yrs)'].astype(int)


current_pitcher_agents[['TOT_GP (2 Yrs)', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'TOT_WAR (2 Yrs)', 'AVG_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/9 (2 Yrs)', 'AVG_K/BB (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']] = current_pitcher_agents.apply(
    lambda row: calculate_pitching_stats(pitching_stats, row['PLAYER (198)'], 2025), axis=1
)

current_pitcher_agents['TOT_GP (2 Yrs)'] = current_pitcher_agents['TOT_GP (2 Yrs)'].astype(int)
current_pitcher_agents['TOT_GS (2 Yrs)'] = current_pitcher_agents['TOT_GS (2 Yrs)'].astype(int)
current_pitcher_agents['AVG_SAVES (2 Yrs)'].fillna(0.0, inplace=True)
current_pitcher_agents = current_pitcher_agents[current_pitcher_agents['TOT_GP (2 Yrs)'] > 0]
current_pitcher_agents['AVG_SAVES (2 Yrs)'] = current_pitcher_agents['AVG_SAVES (2 Yrs)'].astype(int)

#print(previous_free_agents.head(50))
#print(batter_free_agents.head(50))
#print(pitcher_free_agents.head(50))
#print(current_batter_agents.head(50))
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

reliever_free_agents = pitcher_free_agents[pitcher_free_agents['POS'] == 'RP']
reliever_free_agents = reliever_free_agents.sample(frac=1, random_state=42).reset_index(drop=True)
current_reliever_agents = current_pitcher_agents[current_pitcher_agents['POS'] == 'RP']
current_reliever_agents = current_reliever_agents.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bat = batter_free_agents[['AGE', 'YEAR', 'TOT_PA (2 Yrs)', 'AVG_HR (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'AVG_WAR (2 Yrs)']]
y_train_bat = batter_free_agents[['YRS', 'AAV']]
X_test_bat = current_batter_agents[['AGE', 'YEAR', 'TOT_PA (2 Yrs)', 'AVG_HR (2 Yrs)', 'AVG_wRC+ (2 Yrs)', 'AVG_OFF (2 Yrs)', 'AVG_DEF (2 Yrs)', 'AVG_WAR (2 Yrs)']]

conditions = [
    (X_train_bat['AGE'] < 27) & (X_train_bat['AVG_WAR (2 Yrs)'] > 5.0),
    (X_train_bat['AGE'] < 30) & ((X_train_bat['AVG_WAR (2 Yrs)'] > 4.0) | (X_train_bat['AVG_DEF (2 Yrs)'] > 7.5)),
    (X_train_bat['AGE'] > 30) & (X_train_bat['AVG_WAR (2 Yrs)'] > 3.0),
]

weights = [2.5, 2.2, 1.15]
sample_weights = np.select(conditions, weights, default=1.0)

model_1 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_1.fit(X_train_bat, y_train_bat['YRS'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_1 = model_1.predict(X_test_bat)

#X_train_bat['AVG_wRC+ (2 Yrs)'] *= 1.1

conditions = [
    (X_train_bat['AVG_WAR (2 Yrs)'] > 5.0),
    (X_train_bat['AVG_DEF (2 Yrs)'] > 7.5) & (X_train_bat['AVG_wRC+ (2 Yrs)'] > 100),
    ((X_train_bat['AVG_wRC+ (2 Yrs)'] >= 120) | (X_train_bat['AVG_HR (2 Yrs)'] >= 25)),
    (X_train_bat['AVG_wRC+ (2 Yrs)'] < 100)
]

weights = [2.5, 1.2, 1.35, 0.9]
sample_weights = np.select(conditions, weights, default=1.0)

# Train the model for target_column_2
model_2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_2.fit(X_train_bat, y_train_bat['AAV'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_2 = model_2.predict(X_test_bat)

# Combine the predictions
predictions = pd.DataFrame({'YRS': predictions_1, 'AAV': predictions_2})
# --------------------------------------------------------------------------------------------
X_train_pitch = starter_free_agents[['AGE', 'YEAR', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'AVG_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/BB (2 Yrs)']]
y_train_pitch = starter_free_agents[['YRS', 'AAV']]
X_test_pitch = current_starter_agents[['AGE', 'YEAR', 'TOT_GS (2 Yrs)', 'TOT_IP (2 Yrs)', 'AVG_ERA (2 Yrs)', 'AVG_SIERA (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K/BB (2 Yrs)']]

conditions = [
    (X_train_pitch['AGE'] <= 31) & (X_train_pitch['AVG_FIP (2 Yrs)'] <= 4.0) & (X_train_pitch['TOT_IP (2 Yrs)'] >= 250),
]

weights = [12.5]
sample_weights = np.select(conditions, weights, default=1.0)

model_3 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_3.fit(X_train_pitch, y_train_pitch['YRS'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_3 = model_3.predict(X_test_pitch)

conditions = [
    (X_train_pitch['AVG_FIP (2 Yrs)'] <= 4.0) & (X_train_pitch['TOT_IP (2 Yrs)'] >= 250),
]

weights = [12.5]
sample_weights = np.select(conditions, weights, default=1.0)

# Train the model for target_column_2
model_4 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_4.fit(X_train_pitch, y_train_pitch['AAV'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_4 = model_4.predict(X_test_pitch)

# Combine the predictions
predictions = pd.DataFrame({'YRS': predictions_3, 'AAV': predictions_4})

importance_scores = model_3.feature_importances_
feature_names = X_train_pitch.columns
feature_importance_dict = dict(zip(feature_names, importance_scores))

player_to_contract = {}

# --------------------------------------------------------------------------------------------
X_train_relief = reliever_free_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_IP (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']]
y_train_relief = reliever_free_agents[['YRS', 'AAV']]
X_test_relief = current_reliever_agents[['AGE', 'YEAR', 'TOT_GP (2 Yrs)', 'TOT_IP (2 Yrs)', 'AVG_FIP (2 Yrs)', 'AVG_WAR (2 Yrs)', 'AVG_K-BB% (2 Yrs)', 'AVG_K% (2 Yrs)', 'AVG_SAVES (2 Yrs)']]

conditions = [
    (X_train_relief['AVG_FIP (2 Yrs)'] <= 3.0),
    (X_train_relief['AVG_FIP (2 Yrs)'] <= 3.5),
    (X_train_relief['AVG_SAVES (2 Yrs)'] >= 20)
]

weights = [2.5, 1.5, 1.2]
sample_weights = np.select(conditions, weights, default=1.0)

model_5 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_5.fit(X_train_relief, y_train_relief['YRS'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_5 = model_5.predict(X_test_relief)

conditions = [
    (X_train_relief['AVG_FIP (2 Yrs)'] <= 3.0),
    (X_train_relief['AVG_FIP (2 Yrs)'] <= 3.5),
]

weights = [2.5, 1.5]
sample_weights = np.select(conditions, weights, default=1.0)

# Train the model for target_column_2
model_6 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
model_6.fit(X_train_relief, y_train_relief['AAV'], sample_weight=sample_weights)

# Make predictions on the test set
predictions_6 = model_6.predict(X_test_relief)

player_projected_contracts = pd.DataFrame(columns=['PLAYER', 'YRS', 'VALUE'])

for idx, (v1, v2) in enumerate(zip(predictions_1, predictions_2)):
    if((current_batter_agents.loc[idx, 'AGE'] >= 37 and current_batter_agents.loc[idx, 'AVG_WAR (2 Yrs)'] <= 1.7)):
        v1 = 1
        v2 *= 0.5
    
    if(round(v1) == 0):
        v1 = 1

    #print(current_batter_agents.loc[idx, 'PLAYER (198)'], round(v1), ' YRS, ', (round((round(v1) * round(v2)) / 1000000)), 'MILLION')
    new_row = {"PLAYER": current_batter_agents.loc[idx, 'PLAYER (198)'], "YRS": round(v1), "VALUE": 1000000 * (round((round(v1) * round(v2)) / 1000000))}
    player_projected_contracts = pd.concat([player_projected_contracts, pd.DataFrame([new_row])], ignore_index=True)

for idx, (v3, v4) in enumerate(zip(predictions_3, predictions_4)):
    if(round(v3)) == 0:
        v3 = 1
    #print(current_starter_agents.loc[idx, 'PLAYER (198)'], round(v3), 'YRS, ', (round((round(v3) * round(v4)) / 1000000)), 'MILLION')
    new_row = {"PLAYER": current_starter_agents.loc[idx, 'PLAYER (198)'], "YRS": round(v3), "VALUE": 1000000 * (round((round(v3) * round(v4)) / 1000000))}
    player_projected_contracts = pd.concat([player_projected_contracts, pd.DataFrame([new_row])], ignore_index=True)

for idx, (v5, v6) in enumerate(zip(predictions_5, predictions_6)):
    if(current_reliever_agents.loc[idx, 'TOT_IP (2 Yrs)'] <= 60 or round(v5) == 0):
        v5 = 1
        v6 *= 0.5

    #print(current_reliever_agents.loc[idx, 'PLAYER (198)'], round(v5), 'YRS, ', (round((round(v5) * round(v6)) / 1000000)), 'MILLION')
    new_row = {"PLAYER": current_reliever_agents.loc[idx, 'PLAYER (198)'], "YRS": round(v5), "VALUE": 1000000 * (round((round(v5) * round(v6)) / 1000000))}
    player_projected_contracts = pd.concat([player_projected_contracts, pd.DataFrame([new_row])], ignore_index=True)

player_projected_contracts.to_sql('fa_contracts', conn, if_exists='replace', index=False)
conn.close()
