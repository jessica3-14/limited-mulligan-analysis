import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import norm


import random
from datetime import date, timedelta
from requests import get
from json import loads
import json
import time
import math
import os


def gaussian_from_trunc(cutoff, kept_wr, emp_mull_rate):
    
    p = emp_mull_rate
    
    z = norm.ppf(p)                 # Φ^{-1}(p)
    phi = norm.pdf(z)               # φ(z)
    
    lam = phi / (1 - p)             # inverse Mills ratio
    
    sigma = (kept_wr - cutoff) / (lam - z)
    mu = cutoff - sigma * z
    
    return mu, sigma

def get_card_list(search_query):
    next_page = f"https://api.scryfall.com/cards/search?q={search_query}"
    
    card_list = []
    while next_page:
        cards = loads(get(next_page).text)
        next_page = cards.get('next_page')
        if('data' not in cards):
            continue
        card_list += cards['data']
        time.sleep(0.2)
    card_list = [card['name'] for card in card_list]
    

    return card_list




def calculate_set_win_rate(directory='.'):
    """
    Reads CSV files, normalizes the data to include both Player and Opponent
    perspectives, and calculates the unified win rate based ONLY on Play/Draw status, 
    restricted ONLY to Game 1, aggregated across ALL sets.
    """
    
    # Required columns for normalization and grouping
    required_cols = [
    "on_play",
    "num_mulligans",
    "opp_num_mulligans",
    "won",
    "candidate_hand_1",
    "candidate_hand_2",
    "candidate_hand_3",
    "candidate_hand_4",
    "candidate_hand_5",
    "candidate_hand_6",
    "candidate_hand_7",
    "game_number",
    "user_turn_3_eot_user_lands_in_play",
    "user_turn_4_eot_user_lands_in_play",
    "user_turn_5_eot_user_lands_in_play",
    "user_turn_1_cards_drawn",
    "user_turn_2_cards_drawn",
    "user_turn_3_cards_drawn",
    "user_turn_4_cards_drawn",
    "user_turn_5_cards_drawn"
    ]
    dtypes = {"on_play":int,
    "num_mulligans":int,
    "opp_num_mulligans":int,
    "won":int, "candidate_hand_1": str,"candidate_hand_2": str,"candidate_hand_3": str,"candidate_hand_4": 
              str,"candidate_hand_5": str,"candidate_hand_6": str,"candidate_hand_7": str, "game_number": int,"user_turn_3_eot_user_lands_in_play":str,
              "user_turn_4_eot_user_lands_in_play":str,
              "user_turn_5_eot_user_lands_in_play":str,"user_turn_1_cards_drawn":str,
    "user_turn_2_cards_drawn":str,
    "user_turn_3_cards_drawn":str,
    "user_turn_4_cards_drawn":str,
    "user_turn_5_cards_drawn":str}
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not all_files:
        print(f"🛑 No CSV files found in the directory: {os.path.abspath(directory)}")
        return pd.DataFrame()

    print(f"✅ Found {len(all_files)} CSV files. Combining data...")


    df = pd.read_csv("./other/cards.csv")


    #correction for mh3 mdfcs - still broken
    #set_names = ['eoe', 'eos', 'tdm','dft', 'fdn','blb','otj','otp','big','ktk','woe','wot','sir','sis','dmu','scn']
    #cycler sets: fin/fca, dsk, mkm(just one common),lci,ltr,mom
    #sets missing cols: afr and stx
    #land_names=(get_card_list(search_query="t:land set:mh3"))

    id_marked = (
    df.set_index("id")["types"]
    .str.contains("Land", na=False)
    .astype(int)
    .to_dict()
    )

    #ids_to_mark = df.loc[df["name"].isin(land_names), "id"]

    #for card_id in ids_to_mark:
    #    id_marked[card_id] = 1



    df_list=[]

    third_df = pd.read_csv("lands.csv")

# Convert the 'name' column to a list
    third_lands = third_df["name"].dropna()


    names_set = set(third_lands)  # faster lookup

    ids = df.loc[df["name"].isin(names_set), "id"].tolist()

    third_land_cards = set(ids)
    
    
    for filename in all_files:
        try:
            # 1. Load the necessary columns
            df = pd.read_csv(filename, usecols=required_cols,dtype=dtypes)
            
            df = df[df['game_number'] == 1]
            
            hand_cols = [f"candidate_hand_{i}" for i in range(1,8)]

            drawn_cols = [f"user_turn_{i}_cards_drawn" for i in range(1,6)]

            for col in hand_cols:

                cards = df[col].str.split("|", expand=True).astype(float).fillna(0).astype(int)

                lands = cards.stack().map(id_marked).unstack().fillna(0)

                df[col + "_lands"] = lands.sum(axis=1).astype("int8")

            for col in drawn_cols:

                cards = df[col].str.split("|", expand=True).astype(float).fillna(0).astype(int)

                lands = cards.stack().map(id_marked).unstack().fillna(0) 

                df[col+ "_lands"] = lands.sum(axis=1).astype("int8")
            
            land_cols = [f"{c}_lands" for c in hand_cols]

            land_array = df[land_cols].to_numpy()

            kept_index = df["num_mulligans"].clip(0,6).to_numpy()

            df["kept_lands"] = land_array[np.arange(len(df)), kept_index]
            df["hand1_lands"] = df["candidate_hand_1_lands"]

            

            df["kept_7"] = (df["num_mulligans"] == 0).astype(int)

            print(df.loc[(df["candidate_hand_1_lands"] == 0) & (df["kept_7"] == 1), "candidate_hand_1"].to_list())

            df["turn_1_lands"] = df["kept_lands"]+df["user_turn_1_cards_drawn_lands"]
            for i in range(2,6):
                df["turn_"+str(i)+"_lands"]=df["turn_"+str(i-1)+"_lands"]+df["user_turn_"+str(i)+"_cards_drawn_lands"]

            df_list.append(df)
            print(df["turn_5_lands"][0:20])

            print("finished "+filename)

            
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
   
    combined_df = pd.concat(df_list, ignore_index=True)
    df=combined_df
    
    mean_wr = df['won'].mean()
    print("mean wr: "+str(mean_wr))
    rescaling_wr = 0.5/mean_wr
    df["won"] = df["won"] * rescaling_wr
    
    df_sub = df[
    (df["candidate_hand_1_lands"] == 2) &
    (df["num_mulligans"] == 0)
    ].copy()

    df_sub["lands_t3"] = (
    df_sub["user_turn_3_eot_user_lands_in_play"]
    .fillna("")
    .apply(lambda x: len(x.split("|")) if x else 0)
    )
    df_sub["lands_t4"] = (
    df_sub["user_turn_4_eot_user_lands_in_play"]
    .fillna("")
    .apply(lambda x: len(x.split("|")) if x else 0)
    )
    df_sub["lands_t5"] = (
    df_sub["user_turn_5_eot_user_lands_in_play"]
    .fillna("")
    .apply(lambda x: len(x.split("|")) if x else 0)
    )


    df_sub["turn_hit_land"] = np.select(
    [
        df_sub["lands_t3"] >= 3,
        df_sub["lands_t4"] >= 3,
        df_sub["lands_t5"] >= 3,
    ],
    [3, 4, 5],
    default=6  # never reached by turn 5
    )

    result = (
    df_sub
    .groupby(["on_play", "turn_hit_land"])["won"]
    .mean()
    .reset_index()
    )

    result2 = (
    df_sub
    .groupby(["on_play", df_sub["lands_t3"] >= 3])["won"]
    .mean()
    .reset_index()
    )

    t3_lands = (
    df
    .groupby(["on_play", df_sub["turn_3_lands"]])["won"]
    .mean()
    .reset_index()
    )

    t4_lands = (
    df
    .groupby(["on_play", df_sub["turn_4_lands"]])["won"]
    .mean()
    .reset_index()
    )

    t5_lands = (
    df
    .groupby(["on_play", df_sub["turn_5_lands"]])["won"]
    .mean()
    .reset_index()
    )

    counts = df["turn_5_lands"].value_counts()

    print(result)
    print(result2)
    print(t3_lands)
    print(t4_lands)
    print(t5_lands)
    print(counts)

    return
    


# --- Execution ---
if __name__ == '__main__':
    calculate_set_win_rate(directory='./trad_replays/') 
    
    