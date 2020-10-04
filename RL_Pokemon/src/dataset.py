import csv
import re
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs


df_raw = pd.read_csv("../dependencies/pokemon.csv", nrows=166)
features = ['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
df_pruned = df_raw[features]
for idx, row in df_pruned.iterrows():
    if 'Mega ' in row.Name:
        df_pruned = df_pruned.drop(idx, axis=0)
df_pruned = df_pruned.set_index('#')
df_pruned.to_csv("../dependencies/gen1.csv")


url = "https://pokemondb.net/move/generation/1"
physicals = ["Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"]
specials = ["Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"]

with requests.get(url) as response:
    soup = bs(response.text, "html.parser")

rows = []
row = []

for tr in soup.find_all("tr"):
    for td in tr.find_all("td"):
        if td.string:
            row.append(td.string)
        else:
            row.append(" ")
    rows.append(row)
    try:
        row[2] = (row[3] != "—") * (1 * row[1] in physicals) - (1 * row[1] in specials)
    except:
        pass
    row = []
row = row[1:]

header = [i.string for i in soup.find("tr").contents if i!=" " and i!="\n"]

with io.open("../dependencies/moves.csv", 'w', newline="", encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
