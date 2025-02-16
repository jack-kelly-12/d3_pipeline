import pandas as pd
import os
import sqlite3
import re


def clean_team_name(name):
    if pd.isna(name):
        return name

    name = str(name)

    for year in range(2021, 2025):
        if str(year) in name:
            name = name.split(' ' + str(year))[0].strip()
    name = re.sub(r'#\d+\s*', '', name)

    return name.strip()


def calculate_park_factors(schedules):
    schedules['home_team'] = schedules['home_team'].apply(clean_team_name)
    schedules['away_team'] = schedules['away_team'].apply(clean_team_name)

    home_away_games = schedules[
        schedules['neutral_site'].isna() &
        schedules['home_team_score'].notna() &
        schedules['away_team_score'].notna()
    ]

    neutral_games = schedules[
        schedules['neutral_site'].notna() &
        schedules['home_team_score'].notna() &
        schedules['away_team_score'].notna()
    ]

    home_stats = home_away_games.groupby(['home_team']).agg({
        'home_team_score': ['sum', 'size']
    }).reset_index()
    home_stats.columns = ['team_name', 'home_runs', 'home_games']

    away_stats_regular = home_away_games.groupby(['away_team']).agg({
        'away_team_score': ['sum', 'size']
    }).reset_index()
    away_stats_regular.columns = ['team_name', 'away_runs', 'away_games']

    # Handle neutral site games - treat both teams as away teams
    neutral_home = neutral_games.groupby(['home_team']).agg({
        'home_team_score': ['sum', 'size']
    }).reset_index()
    neutral_home.columns = ['team_name', 'away_runs', 'away_games']

    neutral_away = neutral_games.groupby(['away_team']).agg({
        'away_team_score': ['sum', 'size']
    }).reset_index()
    neutral_away.columns = ['team_name', 'away_runs', 'away_games']

    away_stats = pd.concat(
        [away_stats_regular, neutral_home, neutral_away], ignore_index=True)
    away_stats = away_stats.groupby(['team_name']).agg({
        'away_runs': 'sum',
        'away_games': 'sum'
    }).reset_index()

    combined_stats = pd.merge(
        home_stats,
        away_stats,
        on=['team_name'],
        how='outer'
    )

    combined_stats = combined_stats.fillna(0)

    combined_stats = combined_stats[
        ((combined_stats['home_games'] + combined_stats['away_games']) >= 30)
    ]

    T = len(combined_stats)

    park_factors = combined_stats.assign(
        H=lambda x: x['home_runs'] / x['home_games'],  # Runs per game at home
        R=lambda x: x['away_runs'] / x['away_games'],  # Runs per game on road
        raw_PF=lambda x: (x['H'] * T) / ((T - 1) *
                                         x['R'] + x['H']),  # Raw park factor
        iPF=lambda x: (x['raw_PF'] + 1) / 2,  # Regressed park factor
        PF=lambda x: 100 * (1 - (1 - x['iPF']) * .6),
        total_home_games=lambda x: x['home_games'],
        total_away_games=lambda x: x['away_games'],
        Years='2021-2024'
    ).sort_values('PF', ascending=False).fillna(100.0)

    final_columns = [
        'team_name', 'Years', 'iPF', 'PF', 'H', 'R',
        'total_home_games', 'total_away_games'
    ]

    return park_factors[final_columns]


def main():
    os.makedirs('data/park_factors', exist_ok=True)

    for division in [1, 2, 3]:
        division_schedules = pd.DataFrame()

        for year in range(2021, 2025):
            file_path = f'C:/Users/kellyjc/Desktop/d3_app_improved/backend/data/schedules/d{division}_{year}_schedules.csv'

            if os.path.exists(file_path):
                year_schedule = pd.read_csv(file_path)
                division_schedules = pd.concat(
                    [division_schedules, year_schedule], ignore_index=True)

        if not division_schedules.empty:
            park_factors = calculate_park_factors(division_schedules)

            output_path = f'C:/Users/kellyjc/Desktop/d3_app_improved/backend/data/park_factors/d{division}_park_factors.csv'
            park_factors.to_csv(output_path, index=False)
            print(f"Saved park factors for Division {division} to CSV")

            db_path = '../ncaa.db'
            table_name = f'd{division}_park_factors'

            with sqlite3.connect(db_path) as conn:
                park_factors.to_sql(
                    table_name, conn, if_exists='replace', index=False)
                print(
                    f"Saved park factors for Division {division} to SQLite table '{table_name}'")
        else:
            print(f"No schedule data found for Division {division}")


if __name__ == '__main__':
    main()
