import os
import sqlite3
import pandas as pd
from pathlib import Path


def connect_db():
    """Create a connection to the SQLite database."""
    db_path = Path('ncaa.db')
    return sqlite3.connect(db_path)


def update_guts_constants():
    conn = connect_db()
    try:
        df = pd.read_csv('data/guts/guts_constants.csv')

        df.to_sql('guts_constants', conn, if_exists='replace', index=False)
        print("Successfully updated guts_constants table")
    except Exception as e:
        print(f"Error updating guts_constants: {e}")
    finally:
        conn.close()


def update_leaderboards():
    conn = connect_db()
    try:
        for lb in ['splits', 'situational', 'baserunning', 'batted_ball']:
            df = pd.read_csv(f'data/leaderboards/{lb}.csv')
            df.to_sql(f'{lb}', conn, if_exists='replace', index=False)
        print("Successfully updated leaderboards")
    except Exception as e:
        print(f"Error updating guts_constants: {e}")
    finally:
        conn.close()


def update_schedules():
    conn = connect_db()
    try:
        delete_query = "DELETE FROM schedules WHERE year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_schedules_2025.csv'
            try:
                df = pd.read_csv(f'data/schedules/{file_name}')
                df.to_sql('schedules', conn, if_exists='append', index=False)
                print(f"Successfully updated schedules with {file_name}")
            except Exception as e:
                print(f"Error updating schedules with {file_name}: {e}")

        conn.commit()
        print("Successfully completed schedules update")
    except Exception as e:
        print(f"Error in schedules update process: {e}")
        conn.rollback()
    finally:
        conn.close()


def update_rosters():
    conn = connect_db()
    try:
        delete_query = "DELETE FROM rosters WHERE Year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_rosters_2025.csv'
            try:
                df = pd.read_csv(f'data/rosters/{file_name}')
                df.to_sql('rosters', conn, if_exists='append', index=False)
                print(f"Successfully updated rosters with {file_name}")
            except Exception as e:
                print(f"Error updating rosters with {file_name}: {e}")

        conn.commit()
        print("Successfully completed rosters update")
    except Exception as e:
        print(f"Error in rosters update process: {e}")
        conn.rollback()
    finally:
        conn.close()


def update_expected_runs():
    conn = connect_db()
    try:
        delete_query = "DELETE FROM expected_runs WHERE Year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_expected_runs_2025.csv'
            try:
                df = pd.read_csv(f'data/miscellaneous/{file_name}')
                df = df.reset_index()
                df.index = ['_ _ _', '1B _ _', '_ 2B _', '1B 2B _',
                            '_ _ 3B', '1B _ 3B', '_ 2B 3B', '1B 2B 3B']
                df = df.reset_index()
                df = df.rename(columns={'index': 'Bases'})

                df['Year'] = 2025
                df['Division'] = int(division[1])

                df.to_sql('expected_runs', conn,
                          if_exists='append', index=False)
                print(f"Successfully updated expected_runs with {file_name}")
            except Exception as e:
                print(f"Error updating expected_runs with {file_name}: {e}")

        conn.commit()
        print("Successfully completed expected_runs update")
    except Exception as e:
        print(f"Error in expected_runs update process: {e}")
        conn.rollback()
    finally:
        conn.close()


def update_pbp():
    conn = connect_db()
    try:
        delete_query = "DELETE FROM pbp WHERE year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_parsed_pbp_new_2025.csv'
            try:
                columns = [
                    'home_team', 'away_team', 'home_score', 'away_score', 'date',
                    'inning', 'top_inning', 'game_id', 'description',
                    'home_win_exp_before', 'wpa', 'run_expectancy_delta', 'woba', 'home_win_exp_after',
                    'player_id', 'pitcher_id', 'batter_id', 'li', 'home_score_after',
                    'away_score_after', 'event_cd', 'times_through_order', 'base_cd_before', 'base_cd_after',
                    'hit_type'
                ]
                df = pd.read_csv(f'data/play_by_play/{file_name}')
                df[columns].to_sql(
                    'pbp', conn, if_exists='append', index=False)
                print(f"Successfully updated rosters with {file_name}")
            except Exception as e:
                print(f"Error updating rosters with {file_name}: {e}")

        conn.commit()
        print("Successfully completed rosters update")
    except Exception as e:
        print(f"Error in rosters update process: {e}")
        conn.rollback()
    finally:
        conn.close()


def update_war():
    conn = connect_db()
    try:
        war_tables = [
            'batting_team_war',
            'batting_war',
            'pitching_team_war',
            'pitching_war'
        ]

        for table in war_tables:
            delete_query = f"DELETE FROM {table} WHERE Season = 2025"
            conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_to_table = {
                f'{division}_batting_team_war_2025.csv': 'batting_team_war',
                f'{division}_batting_war_2025.csv': 'batting_war',
                f'{division}_pitching_team_war_2025.csv': 'pitching_team_war',
                f'{division}_pitching_war_2025.csv': 'pitching_war'
            }

            for file_name, table_name in file_to_table.items():
                try:
                    df = pd.read_csv(f'data/war/{file_name}')
                    df.to_sql(table_name, conn,
                              if_exists='append', index=False)
                    print(
                        f"Successfully updated {table_name} with {file_name}")
                except Exception as e:
                    print(f"Error updating {table_name} with {file_name}: {e}")

        conn.commit()
        print("Successfully completed WAR updates")
    except Exception as e:
        print(f"Error in WAR update process: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    print("Starting database update process...")
    update_guts_constants()
    update_war()
    update_pbp()
    update_leaderboards()
    update_rosters()
    update_expected_runs()
    update_schedules()
    print("Database update process completed")
