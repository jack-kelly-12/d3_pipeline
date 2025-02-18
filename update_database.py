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
    return parser.parse_args()


def connect_db(db_path):
    """Create a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def update_guts_constants(conn, data_dir):
    """Update guts constants table."""
    try:
        df = pd.read_csv(Path(data_dir) / 'guts' / 'guts_constants.csv')
        df.to_sql('guts_constants', conn, if_exists='replace', index=False)
        print("Successfully updated guts_constants table")
    except Exception as e:
        print(f"Error updating guts_constants: {e}")


def update_leaderboards(conn, data_dir):
    """Update leaderboard tables."""
    try:
        for lb in ['splits', 'situational', 'baserunning', 'batted_ball']:
            df = pd.read_csv(Path(data_dir) / 'leaderboards' / f'{lb}.csv')
            df.to_sql(f'{lb}', conn, if_exists='replace', index=False)
        print("Successfully updated leaderboards")
    except Exception as e:
        print(f"Error updating leaderboards: {e}")


def update_schedules(conn, data_dir):
    """Update schedules table."""
    try:
        delete_query = "DELETE FROM schedules WHERE year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_schedules_2025.csv'
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


def update_rosters(conn, data_dir):
    """Update rosters table."""
    try:
        delete_query = "DELETE FROM rosters WHERE Year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_rosters_2025.csv'
            try:
                df = pd.read_csv(Path(data_dir) / 'rosters' / file_name)
                df.to_sql('rosters', conn, if_exists='append', index=False)
                print(f"Successfully updated rosters with {file_name}")
            except Exception as e:
                print(f"Error updating rosters with {file_name}: {e}")

        conn.commit()
        print("Successfully completed rosters update")
    except Exception as e:
        print(f"Error in rosters update process: {e}")
        conn.rollback()


def update_expected_runs(conn, data_dir):
    """Update expected runs table."""
    try:
        delete_query = "DELETE FROM expected_runs WHERE Year = 2025"
        conn.execute(delete_query)

        for division in ['d1', 'd2', 'd3']:
            file_name = f'{division}_expected_runs_2025.csv'
            try:
                df = pd.read_csv(Path(data_dir) / 'miscellaneous' / file_name)
                df['Bases'] = ['_ _ _', '1B _ _', '_ 2B _', '1B 2B _',
                               '_ _ 3B', '1B _ 3B', '_ 2B 3B', '1B 2B 3B']
                df = df.reset_index()
                df = df.set_index('Bases')
                df['Year'] = 2025
                df['Division'] = int(division[1])

                df[['Division', 'Year', 'Bases', '0', '1', '2']].to_sql('expected_runs', conn,
                                                                        if_exists='append', index=False)
                print(f"Successfully updated expected_runs with {file_name}")
            except Exception as e:
                print(f"Error updating expected_runs with {file_name}: {e}")

        conn.commit()
        print("Successfully completed expected_runs update")
    except Exception as e:
        print(f"Error in expected_runs update process: {e}")
        conn.rollback()


def update_pbp(conn, data_dir):
    """Update play-by-play table."""
    try:
        delete_query = "DELETE FROM pbp WHERE year = 2025"
        conn.execute(delete_query)

        for division in [1, 2, 3]:
            file_name = f'd{division}_parsed_pbp_new_2025.csv'
            try:
                columns = [
                    'year', 'division', 'home_team', 'away_team', 'home_score', 'away_score', 'date',
                    'inning', 'top_inning', 'game_id', 'description',
                    'home_win_exp_before', 'wpa', 'run_expectancy_delta', 'woba', 'home_win_exp_after',
                    'player_id', 'pitcher_id', 'batter_id', 'li', 'home_score_after',
                    'away_score_after', 'event_cd', 'times_through_order', 'base_cd_before', 'base_cd_after',
                    'hit_type'
                ]
                df = pd.read_csv(Path(data_dir) / 'play_by_play' / file_name)
                df['year'] = 2025
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


def update_war(conn, data_dir):
    """Update WAR-related tables."""
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
                    df = pd.read_csv(Path(data_dir) / 'war' / file_name)
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


def main():
    """Main function to run the database update process."""
    args = parse_args()

    print("Starting database update process...")
    print(f"Using database at: {args.db_path}")
    print(f"Using data directory: {args.data_dir}")

    conn = connect_db(args.db_path)
    try:
        update_expected_runs(conn, args.data_dir)
        update_guts_constants(conn, args.data_dir)
        update_war(conn, args.data_dir)
        update_pbp(conn, args.data_dir)
        update_leaderboards(conn, args.data_dir)
        update_rosters(conn, args.data_dir)
        update_schedules(conn, args.data_dir)
        print("Database update process completed")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
