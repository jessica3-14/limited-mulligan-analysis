import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import seaborn as sns

import random
from datetime import date, timedelta
from requests import get
from json import loads
import json
import time
import math
import os

def plot_ideal_vs_actual_mulligan(regression_df):
    """
    Plots the average mulligan rate vs the 'ideal' (predicted high-skill) 
    mulligan rate across different land counts, split by Play/Draw.
    """
    import matplotlib.pyplot as plt
    
    # 1. Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Define colors and scenarios
    scenarios = [
        {'on_play': 1, 'ax': ax1, 'title': 'On the Play', 'color_avg': '#377eb8', 'color_ideal': '#a6cee3'},
        {'on_play': 0, 'ax': ax2, 'title': 'On the Draw', 'color_avg': '#ff7f00', 'color_ideal': '#fdbf6f'}
    ]

    for sc in scenarios:
        subset = regression_df[regression_df['on_play'] == sc['on_play']].sort_values('lands')
        ax = sc['ax']
        
        # Plot Actual Average Mulligan Rate
        ax.plot(subset['lands'], subset['avg_mull_rate'], 
                marker='o', linestyle='-', linewidth=3, 
                color=sc['color_avg'], label='Actual Avg (All Players)')
        
        # Plot Ideal (Predicted 70% WR) Mulligan Rate
        ax.plot(subset['lands'], subset['predicted_70'], 
                marker='s', linestyle='--', linewidth=2, 
                color=sc['color_ideal'], label='Ideal (Predicted 70% WR)')
        
        # Fill the "Efficiency Gap"
        ax.fill_between(subset['lands'], subset['avg_mull_rate'], subset['predicted_70'], 
                        color=sc['color_avg'], alpha=0.1, label='Skill Gap')

        # Formatting
        ax.set_title(sc['title'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Lands in Opening Hand', fontsize=12)
        ax.set_xticks(subset['lands'].unique())
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend()

    ax1.set_ylabel('Mulligan Rate', fontsize=12)
    plt.suptitle('Actual vs. Ideal Mulligan Decisions by Land Count', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("ideal_vs_actual_mulligan.png")
    plt.show()

def player_strength_regression(directory_replays='.', directory_games='.'):
    """
    Reads CSV files, zips together play and replay data, then 
    """
    
    # Required columns for normalization and grouping
    required_cols_replay = [
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
    "match_number",
    "draft_id"
    ]
    required_cols_game = [
    "game_number",
    "match_number",
    "draft_id",
    "user_game_win_rate_bucket"
    ]
    dtypes_replay = {"on_play":int,
    "num_mulligans":int,
    "opp_num_mulligans":int,
    "won":int, "candidate_hand_1": str,"candidate_hand_2": str,"candidate_hand_3": str,"candidate_hand_4": 
              str,"candidate_hand_5": str,"candidate_hand_6": str,"candidate_hand_7": str, "game_number": int, "match_number":int,
              "draft_id":str}
    dtypes_game = {"game_number": int, "match_number":int,
              "draft_id":str, "user_game_win_rate_bucket": float}
    replay_files = glob.glob(os.path.join(directory_replays, "*.csv"))
    game_files = glob.glob(os.path.join(directory_games, "*.csv"))
    
    if not replay_files:
        print(f"🛑 No CSV replay files found in the directory: {os.path.abspath(directory_replays)}")
        return pd.DataFrame()
    if not game_files:
        print(f"🛑 No CSV game files found in the directory: {os.path.abspath(directory_games)}")
        return pd.DataFrame()

    print(f"✅ Found {len(replay_files)} CSV files. Combining data...")


    df = pd.read_csv("./other/cards.csv")


    #mh3 excluded for mdfcs
    # set_names = ['eoe', 'eos', 'tdm','dft','mh3', 'fdn','blb','otj','otp','big','ktk','woe','wot','sir','sis','dmu','fin','fca','dsk','mkm','lci','ltr','mom']
    set_names = ['blb', 'dft', 'dsk', 'eoe', 'fdn', 'fin', 'mkm', 'otj', 'tdm']
    

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

    
    replay_files.sort()
    game_files.sort()
    
    for filename_replay, filename_game in zip(replay_files, game_files):
        try:
            # 1. Load the necessary columns
            df_replay = pd.read_csv(filename_replay, usecols=required_cols_replay,dtype=dtypes_replay)
            df_game = pd.read_csv(filename_game, usecols=required_cols_game,dtype=dtypes_game)
            
            df_replay = df_replay[df_replay['game_number'] == 1]
            df_game = df_game[df_game['game_number'] == 1]
            # print(f"len df replay: {len(df_replay)}")
            # print(f"len df game: {len(df_game)}")
            # print(df_replay.head(1))
            # print(df_game.head(1))
            
            df = pd.merge(df_replay, df_game, on=['draft_id', 'match_number'], how='inner')
            # print(f"len df here: {len(df)}")
            
            hand_cols = [f"candidate_hand_{i}" for i in range(1,8)]

            for col in hand_cols:
                # print(f"len df there: {len(df)}")
                # print(f"{filename_replay}, {col}")

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
                    # print(f"len df: {len(df)}")
                    # print(f"candidate_hand_1_lands==2: {df['candidate_hand_1_lands']==2}")
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

            print("finished "+filename_replay)

            
            
        except Exception as e:
            print(f"❌ Error reading {filename_replay}: {e}")
    
   
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
            df.groupby(["on_play", "candidate_hand_1_lands", "user_game_win_rate_bucket"])
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
    
    
    
    print(mulligan_stats)
    
    print("\n=== Linear Regression Analysis ===")
    
    # Remove game threshold filtering entirely - use weighting instead
    mulligan_stats_filtered = mulligan_stats[mulligan_stats['games'] >= 0]
    
    # Store regression results
    regression_results = []
    
    # Perform regression for each combination of on_play and candidate_hand_1_lands (2-5 lands only)
    for on_play in [0, 1]:
        for lands in sorted(mulligan_stats_filtered['candidate_hand_1_lands'].unique()):
            if lands < 2 or lands > 5:  # Skip hands with less than 2 or more than 5 lands
                continue
                
            subset = mulligan_stats_filtered[
                (mulligan_stats_filtered['on_play'] == on_play) & 
                (mulligan_stats_filtered['candidate_hand_1_lands'] == lands)
            ]
            
            if len(subset) >= 3:  # Need at least 3 points for meaningful regression
                X = subset['user_game_win_rate_bucket'].values.reshape(-1, 1)
                y = subset['mulligan_rate'].values
                weights = subset['games'].values  # Add sample weights
                
                reg = LinearRegression().fit(X, y, sample_weight=weights)
                slope = reg.coef_[0]
                intercept = reg.intercept_
                r_squared = reg.score(X, y, sample_weight=weights)
                
                # Calculate weighted average mulligan rate for this group
                avg_mull_rate = np.average(subset['mulligan_rate'].values, weights=weights)
                
                # Calculate predicted mulligan rate at 70% winrate
                predicted_70 = reg.predict([[0.7]])[0]
                print(on_play, lands, avg_mull_rate, predicted_70)
                diff_from_avg = predicted_70 - avg_mull_rate
                
                regression_results.append({
                    'on_play': on_play,
                    'play_draw': 'Play' if on_play == 1 else 'Draw',
                    'lands': lands,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'avg_mull_rate': avg_mull_rate,
                    'predicted_70': predicted_70,
                    'diff_from_avg': diff_from_avg,
                    'n_points': len(subset)
                })
    
    regression_df = pd.DataFrame(regression_results)
    print("\nRegression Results:")
    print(regression_df)
    
    # Create the plots
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot 1: Slope vs number of lands
    for on_play in [0, 1]:
        subset = regression_df[regression_df['on_play'] == on_play]
        label = 'On Play' if on_play == 1 else 'On Draw'
        color = '#377eb8' if on_play == 1 else '#ff7f00'
        ax1.plot(subset['lands'], subset['slope'], marker='o', label=label, color=color, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Lands in Opening Hand')
    ax1.set_ylabel('Regression Slope (Mulligan Rate vs Win Rate)')
    ax1.set_title('How Player Skill Affects Mulligan Decisions by Land Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("regression_plot.png")
    plt.show()
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot 1: Slope vs number of lands
    for on_play in [0, 1]:
        subset = regression_df[regression_df['on_play'] == on_play]
        label = 'On Play' if on_play == 1 else 'On Draw'
        color = '#377eb8' if on_play == 1 else '#ff7f00'
        ax1.plot(subset['lands'], subset['slope']/subset['avg_mull_rate'], marker='o', label=label, color=color, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Lands in Opening Hand')
    ax1.set_ylabel('Regression Slope (Mulligan Rate vs Win Rate) over Avg Mullrate')
    ax1.set_title('How Player Skill Affects Mulligan Decisions by Land Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("regression_plot_rescaled.png")
    plt.show()
    
    # Print some summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average slope on play: {regression_df[regression_df['on_play'] == 1]['slope'].mean():.4f}")
    print(f"Average slope on draw: {regression_df[regression_df['on_play'] == 0]['slope'].mean():.4f}")
    print(f"Average R² on play: {regression_df[regression_df['on_play'] == 1]['r_squared'].mean():.4f}")
    print(f"Average R² on draw: {regression_df[regression_df['on_play'] == 0]['r_squared'].mean():.4f}")
    
    # NEW: Plot individual regressions for each land count (2-5 lands only)
    unique_lands = sorted([l for l in regression_df['lands'].unique() if 2 <= l <= 5])
    n_lands = len(unique_lands)
    
    # Create subplots - adjust layout based on number of land counts
    cols = min(4, n_lands)
    rows = (n_lands + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_lands == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if n_lands == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for idx, lands in enumerate(unique_lands):
        ax = axes[idx]
        
        # Plot data points and regression lines for both on_play conditions
        for on_play in [0, 1]:
            # Get the original data points (filtered to 2-5 lands)
            subset_data = mulligan_stats_filtered[
                (mulligan_stats_filtered['on_play'] == on_play) & 
                (mulligan_stats_filtered['candidate_hand_1_lands'] == lands) &
                (mulligan_stats_filtered['candidate_hand_1_lands'] >= 2) &
                (mulligan_stats_filtered['candidate_hand_1_lands'] <= 5)
            ]
            
            if len(subset_data) >= 3:
                # Filter to only include players with 40-75% win rate
                subset_data = subset_data[
                    (subset_data['user_game_win_rate_bucket'] > 0.40) & 
                    (subset_data['user_game_win_rate_bucket'] < 0.75)
                ]
                
                # Plot scatter points with size based on number of games
                color = '#377eb8' if on_play == 1 else '#ff7f00'
                label_play = 'On Play' if on_play == 1 else 'On Draw'
                # Scale circle size based on number of games (min 20, max 200)
                if len(subset_data) > 0:
                    sizes = 20 + (subset_data['games'] - subset_data['games'].min()) / (subset_data['games'].max() - subset_data['games'].min() + 1e-8) * 180
                    ax.scatter(subset_data['user_game_win_rate_bucket'], 
                              subset_data['mulligan_rate'], 
                              color=color, alpha=0.6, s=sizes, label=f'{label_play} (data)')
                    
                    # Get regression line
                    reg_result = regression_df[
                        (regression_df['on_play'] == on_play) & 
                        (regression_df['lands'] == lands)
                    ]
                    
                    if not reg_result.empty:
                        slope = reg_result['slope'].iloc[0]
                        intercept = reg_result['intercept'].iloc[0]
                        r_squared = reg_result['r_squared'].iloc[0]
                        # Create regression line
                        x_range = np.linspace(subset_data['user_game_win_rate_bucket'].min(), 
                                            subset_data['user_game_win_rate_bucket'].max(), 100)
                        y_pred = slope * x_range + intercept
                        
                        ax.plot(x_range, y_pred, color=color, linewidth=2, 
                               label=f'{label_play} (R²={r_squared:.3f})')
        
        ax.set_xlabel('Player Win Rate')
        ax.set_ylabel('Mulligan Rate')
        ax.set_title(f'{lands} Lands')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_lands, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("four_regressions.png")
    plt.show()
    
    

    return regression_df


# --- Execution ---
if __name__ == '__main__':
    results = player_strength_regression(directory_replays='./trad_replays/', directory_games='./trad_games') 
    plot_ideal_vs_actual_mulligan(results)
    