import pandas as pd
import numpy as np
from pathlib import Path
import traceback


def get_re(base_cd, outs, re24_matrix):
    """
    Get run expectancy value from matrix using base state encoding
    """
    try:
        if pd.isna(base_cd) or pd.isna(outs):
            return 0

        # Convert numeric base_cd to the string format used in the matrix
        base_states = {
            0: '_ _ _',
            1: '1B _ _',
            2: '_ 2B _',
            3: '1B 2B _',
            4: '_ _ 3B',
            5: '1B _ 3B',
            6: '_ 2B 3B',
            7: '1B 2B 3B'
        }

        base_state = base_states.get(int(base_cd), '_ _ _')
        outs_idx = str(min(max(int(outs), 0), 2))

        return re24_matrix.loc[base_state, outs_idx]
    except Exception as e:
        print(f"Error in get_re: base_cd={base_cd}, outs={outs}")
        print(f"Error details: {str(e)}")
        raise


def calculate_college_linear_weights(pbp_df, re24_matrix):
    try:
        # Event mapping
        event_map = {
            2: 'out', 3: 'out', 6: 'out', 14: 'walk',
            16: 'hit_by_pitch', 20: 'single', 21: 'double',
            22: 'triple', 23: 'home_run'
        }

        # Convert event_cd to string type before mapping
        pbp_df['event_cd'] = pbp_df['event_cd'].astype(str)
        events = pd.Series(pbp_df['event_cd'].map(
            lambda x: event_map.get(int(x) if x.isdigit() else None, 'other')))

        print("Calculating RE start values...")
        # Ensure numeric types for base_cd and outs
        pbp_df['base_cd_before'] = pd.to_numeric(
            pbp_df['base_cd_before'], errors='coerce')
        pbp_df['outs_before'] = pd.to_numeric(
            pbp_df['outs_before'], errors='coerce')

        re_start = pbp_df.apply(lambda x: get_re(
            x['base_cd_before'], x['outs_before'], re24_matrix), axis=1)

        print("Calculating RE end values...")
        # Create next row values for RE end calculation
        next_base = pd.concat(
            [pbp_df['base_cd_before'].iloc[1:], pd.Series([0])])
        next_outs = pd.concat([pbp_df['outs_before'].iloc[1:], pd.Series([0])])
        re_end = pd.Series([get_re(base, outs, re24_matrix)
                           for base, outs in zip(next_base, next_outs)])

        # Set RE end to 0 for inning endings
        re_end[pbp_df['inn_end'] == 1] = 0

        print("Calculating RE24...")
        re24 = re_end - re_start + pbp_df['runs_on_play']

        # Group and calculate
        print("Grouping and calculating final results...")
        results = pd.DataFrame({
            'events': events,
            're24': re24
        }).groupby('events').agg(
            count=('re24', 'count'),
            total_re24=('re24', 'sum')
        ).reset_index()

        results = results[results['events'] != 'other']
        results['linear_weights_above_average'] = (
            results['total_re24'] / results['count']).round(3)

        out_value = results.loc[results['events'] ==
                                'out', 'linear_weights_above_average'].iloc[0]
        results['linear_weights_above_outs'] = (
            results['linear_weights_above_average'] - out_value).round(3)

        return results.sort_values('linear_weights_above_average', ascending=False)

    except Exception as e:
        print(f"Error in calculate_college_linear_weights:")
        print(traceback.format_exc())
        raise


def calculate_normalized_linear_weights(linear_weights, stats_df):
    try:
        # Calculate total value and PA
        total_value = (linear_weights['linear_weights_above_outs'] *
                       linear_weights['count']).sum()
        total_pa = linear_weights['count'].sum()
        denominator = total_value / total_pa

        # Calculate league OBP
        total_stats = stats_df.sum()
        league_obp = (total_stats['H'] + total_stats['BB'] + total_stats['HBP']) / \
                     (total_stats['AB'] + total_stats['BB'] + total_stats['HBP'] +
                      total_stats['SF'] + total_stats['SH'])

        woba_scale = league_obp / denominator

        # Calculate normalized weights
        result = linear_weights.copy()
        result['normalized_weight'] = (
            result['linear_weights_above_outs'] * woba_scale).round(3)

        # Add wOBA scale row
        woba_scale_row = pd.DataFrame({
            'events': ['wOBA scale'],
            'count': [np.nan],
            'total_re24': [np.nan],
            'linear_weights_above_average': [np.nan],
            'linear_weights_above_outs': [np.nan],
            'normalized_weight': [round(woba_scale, 3)]
        })

        return pd.concat([result, woba_scale_row], ignore_index=True)

    except Exception as e:
        print(f"Error in calculate_normalized_linear_weights:")
        print(traceback.format_exc())
        raise


def main(data_dir):
    data_dir = Path(data_dir)
    year = 2025
    divisions = range(1, 4)

    misc_dir = data_dir / 'miscellaneous'
    misc_dir.mkdir(exist_ok=True)

    for division in divisions:
        try:
            # Create filenames using proper string formatting
            pbp_file = data_dir / 'play_by_play' / \
                f'd{division}_parsed_pbp_{year}.csv'
            re24_file = misc_dir / f'd{division}_expected_runs_{year}.csv'
            stats_file = data_dir / 'stats' / f'd{division}_batting_{year}.csv'

            print(f"\nProcessing Division {division}...")
            print(f"PBP file: {pbp_file}")
            print(f"RE24 file: {re24_file}")
            print(f"Stats file: {stats_file}")

            # Check if all required files exist
            for file, desc in [(pbp_file, "PBP"), (re24_file, "Expected runs matrix"),
                               (stats_file, "Batting stats")]:
                if not file.exists():
                    print(f"{desc} not found: {file}")
                    continue

            # Read all required files
            print("Reading files...")
            pbp_df = pd.read_csv(pbp_file)
            re24_df = pd.read_csv(re24_file)
            stats_df = pd.read_csv(stats_file)

            print("Setting up RE24 matrix...")
            # Set up the matrix with the correct index
            re24_matrix = re24_df.set_index('Bases')[['0', '1', '2']]

            print(f"Loaded {len(pbp_df)} rows for D{division} {year}")

            # Calculate basic linear weights
            print("Calculating linear weights...")
            linear_weights = calculate_college_linear_weights(
                pbp_df, re24_matrix)

            # Calculate normalized weights
            print("Calculating normalized weights...")
            normalized_weights = calculate_normalized_linear_weights(
                linear_weights, stats_df)

            # Save results
            output_file = misc_dir / f'd{division}_linear_weights_{year}.csv'
            normalized_weights.to_csv(output_file, index=False)
            print(
                f"Saved linear weights for D{division} {year} to {output_file}")

        except Exception as e:
            print(f"Error processing D{division} {year}:")
            print(traceback.format_exc())
            continue


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    main(args.data_dir)
