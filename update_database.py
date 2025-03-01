import sqlite3
import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Update NCAA baseball statistics database.')
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to the SQLite database file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory containing the data folders (default: data)')
    parser.add_argument('--year', type=str, default='data')
    return parser.parse_args()


def connect_db(db_path):
    """Create a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def update_guts_constants(conn, data_dir, year):
    """Update guts constants table."""
    try:
        df = pd.read_csv(Path(data_dir) / 'guts' / 'guts_constants.csv')
        df.to_sql('guts_constants', conn, if_exists='replace', index=False)
        print("Successfully updated guts_constants table")
    except Exception as e:
        print(f"Error updating guts_constants: {e}")


def update_leaderboards(conn, data_dir, year):
    """Update leaderboard tables."""
    try:
        for lb in ['splits', 'situational', 'baserunning', 'batted_ball']:
            df = pd.read_csv(Path(data_dir) / 'leaderboards' / f'{lb}.csv')
            df.to_sql(f'{lb}', conn, if_exists='replace', index=False)
        print("Successfully updated leaderboards")
    except Exception as e:
        print(f"Error updating leaderboards: {e}")


def update_schedules(conn, data_dir, year):
    """Update schedules table."""
    try:
        delete_query = "DELETE FROM schedules WHERE year = ?"
        conn.execute(delete_query, (year, ))

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_schedules_{year}.csv'
            try:
                df = pd.read_csv(Path(data_dir) / 'schedules' / file_name)
                df.to_sql('schedules', conn, if_exists='append', index=False)
                print(f"Successfully updated schedules with {file_name}")
            except Exception as e:
                print(f"Error updating schedules with {file_name}: {e}")

        conn.commit()
        print("Successfully completed schedules update")
    except Exception as e:
        print(f"Error in schedules update process: {e}")
        conn.rollback()


def update_rosters(conn, data_dir, year):
    """Update rosters table."""
    try:
        delete_query = "DELETE FROM rosters"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            for year in range(2021, 2026):
                file_name = f'{division}_rosters_{year}.csv'
                try:
                    df = pd.read_csv(Path(data_dir) / 'rosters' /
                                     file_name,  dtype={'player_id': str})
                    df.to_sql('rosters', conn, if_exists='append', index=False)
                    print(f"Successfully updated rosters with {file_name}")
                except Exception as e:
                    print(f"Error updating rosters with {file_name}: {e}")

        conn.commit()
        print("Successfully completed rosters update")
    except Exception as e:
        print(f"Error in rosters update process: {e}")
        conn.rollback()


def update_pf(conn, data_dir):
    """Update rosters table."""
    try:
        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_park_factors.csv'
            try:
                df = pd.read_csv(Path(data_dir) / 'park_factors' /
                                 file_name)
                df.to_sql(f'{division}_park_factors', conn,
                          if_exists='replace', index=False)
                print(f"Successfully updated park factors with {file_name}")
            except Exception as e:
                print(f"Error updating park factors with {file_name}: {e}")

        conn.commit()
        print("Successfully completed park factors update")
    except Exception as e:
        print(f"Error in rosters update process: {e}")
        conn.rollback()


def update_expected_runs(conn, data_dir, year):
    """Update expected runs table."""
    try:
        delete_query = "DELETE FROM expected_runs"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            for year in [2021, 2022, 2023, 2024, 2025]:
                file_name = f'{division}_expected_runs_{year}.csv'
                try:
                    df = pd.read_csv(Path(data_dir) /
                                     'miscellaneous' / file_name)
                    df['Bases'] = ['1B 2B 3B', '_ 2 3', '1 _ 3',
                                   '_ _ 3', '1B 2B _', '_ 2 _', '1 _ _', '_ _ _']
                    df['Year'] = year
                    df['Division'] = int(division[1])

                    df_to_upload = df[['Division',
                                       'Year', 'Bases', '0', '1', '2']]

                    df_to_upload.to_sql('expected_runs', conn,
                                        if_exists='append',
                                        index=False)

                    print(
                        f"Successfully updated expected_runs with {file_name}")
                except Exception as e:
                    print(
                        f"Error updating expected_runs with {file_name}: {e}")
        conn.commit()
        print("Successfully completed expected_runs update")
    except Exception as e:
        print(f"Error in expected_runs update process: {e}")
        conn.rollback()


def update_pbp(conn, data_dir, year):
    """Update play-by-play table."""
    try:
        delete_query = "DELETE FROM pbp WHERE year = ?"
        conn.execute(delete_query, (year, ))

        for division in [1, 2, 3]:
            file_name = f'd{division}_parsed_pbp_new_{year}.csv'
            try:
                columns = [
                    'year', 'division', 'play_id', 'home_team', 'away_team', 'home_score', 'away_score', 'date',
                    'inning', 'top_inning', 'game_id', 'description',
                    'home_win_exp_before', 'wpa', 'run_expectancy_delta', 'woba', 'home_win_exp_after',
                    'player_id', 'pitcher_id', 'batter_id', 'li', 'home_score_after',
                    'away_score_after', 'event_cd', 'times_through_order', 'base_cd_before', 'base_cd_after',
                    'hit_type'
                ]
                df = pd.read_csv(Path(data_dir) / 'play_by_play' / file_name,
                                 dtype={'player_id': str, 'pitcher_id': str, 'batter_id': str, })
                df['year'] = year
                df['division'] = division
                df[columns].to_sql(
                    'pbp', conn, if_exists='append', index=False)
                print(f"Successfully updated pbp with {file_name}")
            except Exception as e:
                print(f"Error updating pbp with {file_name}: {e}")

        conn.commit()
        print("Successfully completed pbp update")
    except Exception as e:
        print(f"Error in pbp update process: {e}")
        conn.rollback()


def update_war(conn, data_dir, year):
    """Update WAR-related tables."""
    try:
        war_tables = [
            'batting_team_war',
            'batting_war',
            'pitching_team_war',
            'pitching_war'
        ]

        for table in war_tables:
            delete_query = f"DELETE FROM {table}"
            conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            for year in range(2021, 2026):
                file_to_table = {
                    f'{division}_batting_team_war_{year}.csv': 'batting_team_war',
                    f'{division}_batting_war_{year}.csv': 'batting_war',
                    f'{division}_pitching_team_war_{year}.csv': 'pitching_team_war',
                    f'{division}_pitching_war_{year}.csv': 'pitching_war'
                }

                for file_name, table_name in file_to_table.items():
                    try:
                        if table_name == 'batting_war' or table_name != "pitching_war":
                            df = pd.read_csv(
                                Path(data_dir) / 'war' / file_name, dtype={'player_id': str})
                        else:
                            df = pd.read_csv(
                                Path(data_dir) / 'war' / file_name)
                        if 'IP_float' in df.columns:
                            df = df.drop(columns='IP_float')
                        df.to_sql(table_name, conn,
                                  if_exists='append', index=False)
                        print(
                            f"Successfully updated {table_name} with {file_name}")
                    except Exception as e:
                        print(
                            f"Error updating {table_name} with {file_name}: {e}")

        conn.commit()
        print("Successfully completed WAR updates")
    except Exception as e:
        print(f"Error in WAR update process: {e}")
        conn.rollback()


def main():
    """Main function to run the database update process."""
    args = parse_args()

    print("Starting database update process...")
    print(f"Using database at: {args.db_path}")
    print(f"Using data directory: {args.data_dir}")

    conn = connect_db(args.db_path)
    try:
        update_expected_runs(conn, args.data_dir, args.year)
        update_guts_constants(conn, args.data_dir, args.year)
        update_pf(conn, args.data_dir)
        update_war(conn, args.data_dir, args.year)
        update_pbp(conn, args.data_dir, args.year)
        update_leaderboards(conn, args.data_dir, args.year)
        update_rosters(conn, args.data_dir, args.year)
        update_schedules(conn, args.data_dir, args.year)
        print("Database update process completed")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
