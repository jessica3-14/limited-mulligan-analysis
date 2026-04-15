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
    "game_number"
    ]
    dtypes = {"on_play":int,
    "num_mulligans":int,
    "opp_num_mulligans":int,
    "won":int, "candidate_hand_1": str,"candidate_hand_2": str,"candidate_hand_3": str,"candidate_hand_4": 
              str,"candidate_hand_5": str,"candidate_hand_6": str,"candidate_hand_7": str, "game_number": int}
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not all_files:
        print(f"🛑 No CSV files found in the directory: {os.path.abspath(directory)}")
        return pd.DataFrame()

    print(f"✅ Found {len(all_files)} CSV files. Combining data...")


    df = pd.read_csv("./other/cards.csv")


    #mh3 excluded for mdfcs
    set_names = ['eoe', 'tdm','dft', 'fdn','blb','otj','ktk','woe','sir','dmu','fin','dsk','mkm','lci','ltr','mom']

    

    id_marked = (
    df.set_index("id")["types"]
    .str.contains("Land", na=False)
    .astype(int)
    .to_dict()
    )


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

            for col in hand_cols:

                cards = df[col].str.split("|", expand=True).astype(float).fillna(0).astype(int)

                lands = cards.stack().map(id_marked).unstack().fillna(0)

                df[col + "_lands"] = lands.sum(axis=1).astype("int8")
            
            

                if(col=="candidate_hand_1"):
                    has_third_land = cards.isin(third_land_cards).any(axis=1)
                    is_two_lands = df[col + "_lands"] == 2

                    mask = is_two_lands & has_third_land

    # Count changes
                    total_changed =0
                    changed_count = mask.sum()
                    total_changed += changed_count

    # Apply update
                    total_rows = len(df['candidate_hand_1_lands']==2)
                    df.loc[mask, col + "_lands"] = 3
                    
                    percent_changed = (total_changed / total_rows) * 100

                    print(f"Total rows changed: {total_changed}")
                    print(f"Percent of rows changed: {percent_changed:.2f}%")

            land_cols = [f"{c}_lands" for c in hand_cols]

            land_array = df[land_cols].to_numpy()

            kept_index = df["num_mulligans"].clip(0,6).to_numpy()

            df["kept_lands"] = land_array[np.arange(len(df)), kept_index]
            df["hand1_lands"] = df["candidate_hand_1_lands"]

            lands_2 = df[df["candidate_hand_1_lands"] == 2]

            df["kept_7"] = (df["num_mulligans"] == 0).astype(int)

            print(df.loc[(df["candidate_hand_1_lands"] == 0) & (df["kept_7"] == 1), "candidate_hand_1"].to_list())

            df_list.append(df)

            print("finished "+filename)

            
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
   
    combined_df = pd.concat(df_list, ignore_index=True)
    df=combined_df

    #TODO total wr fix to 2-5 lands?
    
    mean_wr = df['won'].mean()
    print("mean wr: "+str(mean_wr))
    rescaling_wr = 0.5/mean_wr
    
    mull_df = df[df['num_mulligans'] > 0]
    
    play_draw_mull_winrate = mull_df.groupby("on_play")['won'].mean()*rescaling_wr

    keep_df = df[(df['num_mulligans'] == 0) & (df['candidate_hand_1_lands'] < 6) & (df['candidate_hand_1_lands'] > 1)]
    
    total_keep_winrate = keep_df.groupby("on_play")['won'].mean()*rescaling_wr

    total_draw_mullrate = len(df[(df['num_mulligans'] > 0) & (df['on_play']==0)])/len(df[df['on_play']==0])
    total_play_mullrate = len(df[(df['num_mulligans'] > 0) & (df['on_play']==1)])/len(df[df['on_play']==1])
    print('Total mullrate on draw: '+str(total_draw_mullrate))
    print('Total mullrate on play: '+str(total_play_mullrate))

    
    print(f"play draw mull winrate: {play_draw_mull_winrate}")
    print(f"play draw keep winrate: {total_keep_winrate}")

    mulligan_stats = (
            df.groupby(["on_play", "candidate_hand_1_lands"])
            .agg(
            games=("num_mulligans", "size"),
            mulligans=("num_mulligans", lambda x: (x > 0).sum())
            )
            .reset_index()
            )

    mulligan_stats["mulligan_rate"] = mulligan_stats["mulligans"] / mulligan_stats["games"]

    mulligan_stats["play_draw"] = mulligan_stats["on_play"].map({1: "Play", 0: "Draw"})

    mulligan_stats = mulligan_stats.sort_values(
            ["play_draw", "candidate_hand_1_lands"]
            )
    

    winrate_lands = (
                df[df["num_mulligans"] == 0].groupby("candidate_hand_1_lands")
                .agg(
                games=("won","size"),
                winrate=("won","mean")
                )
            )
    winrate_lands["winrate"] *= rescaling_wr
    playdraw_winrate = (
            df[df["num_mulligans"] == 0].groupby(["on_play","candidate_hand_1_lands"])
            .agg(
            games=("won","size"),
            winrate=("won","mean")
            )
            .reset_index()
            )   
    playdraw_winrate["play_draw"] = playdraw_winrate["on_play"].map({1:"Play",0:"Draw"})
    playdraw_winrate["winrate"] *= rescaling_wr
    
    decision_stats = (
            df.groupby(["on_play", "candidate_hand_1_lands", "kept_7"])
            .agg(
            games=("won", "size"),
            winrate=("won", "mean")
            )
            .reset_index()
            )

    decision_table = decision_stats.pivot_table(
            index=["on_play", "candidate_hand_1_lands"],
            columns="kept_7",
            values="winrate"
            ).reset_index()

    decision_table = decision_table.rename(columns={
            0: "winrate_if_mulligan",
            1: "winrate_if_keep"
            })
    
    decision_table["winrate_if_keep"] *= rescaling_wr
    decision_table["winrate_if_mulligan"] *= rescaling_wr
    
    decision_table["keep_advantage"] = (
    decision_table["winrate_if_keep"] -
    decision_table["winrate_if_mulligan"]
        )

    decision_table["play_draw"] = decision_table["on_play"].map({1: "Play", 0: "Draw"})

    decision_table = decision_table.sort_values(
            ["play_draw", "candidate_hand_1_lands"]
        )
    

    stats = playdraw_winrate.merge(
    mulligan_stats[["on_play", "candidate_hand_1_lands", "mulligan_rate"]],
    on=["on_play", "candidate_hand_1_lands"]
    )
    
    stats['cutoff'] = stats['on_play'].map(play_draw_mull_winrate)

    
    print(f"stats: {stats}")
    
    stats[["mean", "stddev"]] = stats.apply(
    lambda r: gaussian_from_trunc(
        r["cutoff"],
        r["winrate"],
        r["mulligan_rate"]
    ),
    axis=1,
    result_type="expand"
    )

    table = stats.pivot(
    index="candidate_hand_1_lands",
    columns="on_play",
    values=["mean", "stddev"]
    )

   
    keep_mull = (df.groupby([df["num_mulligans"]>0,df["opp_num_mulligans"]>0,"on_play"])["won"].mean().reset_index())
    print(keep_mull)
    #keep_mull["won"]*=rescaling_wr
    #print("Rescaled wrs")
    #print(keep_mull)

    play_df = df[df["on_play"] == 1]

    play_result = (
    play_df.assign(
        player_mulligan = play_df["num_mulligans"] > 0,
        opp_mulligan = play_df["opp_num_mulligans"] > 0
    )
    .groupby("player_mulligan")["opp_mulligan"]
    .mean()
    .rename("percent_opp_mulligan")
    .reset_index()
)

    print(play_result)

    

    draw_df = df[df["on_play"]==0]
    draw_result = (
    draw_df.assign(
        player_mulligan = draw_df["num_mulligans"] > 0,
        opp_mulligan = draw_df["opp_num_mulligans"] > 0
    )
    .groupby("opp_mulligan")["player_mulligan"]
    .mean()
    .rename("percent_player_mulligan")
    .reset_index()
)

    print(draw_result)

 

    opp_draw_mullrate = len(df[(df['opp_num_mulligans'] > 0) & (df['on_play']==1)])/len(df[df['on_play']==1])
    opp_play_mullrate = len(df[(df['opp_num_mulligans'] > 0) & (df['on_play']==0)])/len(df[df['on_play']==0])
    print('Opp total mullrate on draw: '+str(opp_draw_mullrate))
    print('Opp total mullrate on play: '+str(opp_play_mullrate))
    

    print(f"table: {table}")

    print(f"winrate lands: {winrate_lands}")
    print(f"playdraw winrate: {playdraw_winrate}")
    print(f"mulligan stats: {mulligan_stats}")
    print(f"decision table: {decision_table}")
           

    plt.plot(winrate_lands.index, winrate_lands["winrate"])
    plt.xlabel("number of lands")
    plt.ylabel("Winrate")
    plt.title("Winrate of kept 7s by number of lands")
    plt.show()

    return
    


# --- Execution ---
if __name__ == '__main__':
    calculate_set_win_rate(directory='./trad_replays/') 
    
    