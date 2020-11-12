import csv
import re
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs


def get_pokemon_gen1():
    df_raw = pd.read_csv("../dependencies/pokemon.csv", nrows=166)
    features = ['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    df_pruned = df_raw[features]
    for idx, row in df_pruned.iterrows():
        if 'Mega ' in row.Name:
            df_pruned = df_pruned.drop(idx, axis=0)
    df_pruned = df_pruned.set_index('#')
    df_pruned.to_csv("../dependencies/gen1.csv")


def get_moves():
    df_pokemon = pd.read_csv('../dependencies/gen1.csv')
    url_head = 'https://bulbapedia.bulbagarden.net/wiki/'
    url_tail = '_(Pok%C3%A9mon)/Generation_I_learnset#By_leveling_up'
    header = ['Move', 'Type', 'Power', 'Accuracy', 'PP', 'Pokemon']
    rows = []

    for name in df_pokemon['Name']:
        url = url_head + name + url_tail     

        with requests.get(url) as response:
            soup = bs(response.text, "html.parser")
        tables = soup.find_all('table')

        rows += level_moves(tables[3], name)
        rows += tm_moves(tables[4], name)
        
    df_moves = pd.DataFrame(rows, columns=header) 
    df_moves.to_csv('../dependencies/gen1_moves.csv')


def clean_row(row):
    try:
        row[2] = re.sub(r'—+', '0', row[2])
        row[3] = re.sub(r'[—*}*%*]+', '', row[3])
        row[3] = '0' + row[3]
        row[2:5] = [int(row[i]) for i in range(2, 5)]
    except:
        print(row)
        return None
        
    return row


def level_moves(table, name):
    rows = []
    for tr in table.find_all('tr')[1:]:
        row = []
        children = list(tr.children)

        if len(children)==12:
            children = children[3::2]
        elif len(children)==14:
            children = children[5::2]
        else:
            continue

        for child in children:
            if child.span:
                child = child.span
            row.append(child.text.strip())
        row.append(name)
        row = clean_row(row)
        if row:
            rows.append(row)

    return rows


def tm_moves(table, name):
    rows = []
    for tr in table.find_all('tr')[5:]:
        row = []
        children = list(tr.children)
        if len(children)<14:
            continue
        children = children[5::2]

        for child in children:
            if child.span:
                child = child.span
            row.append(child.text.strip())
        row.append(name)
        row = clean_row(row)
        rows.append(row)

    return rows


get_moves()
