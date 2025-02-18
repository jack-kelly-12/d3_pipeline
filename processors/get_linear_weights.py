import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def get_re(base_cd: int, outs: int, re24_matrix: pd.DataFrame) -> float:
    """
    Get run expectancy value from the matrix based on base state and outs

    Args:
        base_cd: Base state code (0-7)
        outs: Number of outs (0-2)
        re24_matrix: Run expectancy matrix DataFrame

    Returns:
        float: Run expectancy value
    """
    if pd.isna(base_cd) or pd.isna(outs):
        return 0.0

    # Ensure values are within bounds
    base_cd = int(min(max(base_cd, 0), 7))
    # Convert outs to string to match matrix columns
    outs = str(min(max(outs, 0), 2))

    # Map base_cd to base state string
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

    base_state = base_states[base_cd]

    try:
        return float(re24_matrix.loc[base_state, outs])
    except (KeyError, ValueError):
        print(
            f"Warning: Invalid matrix access - base_state: {base_state}, outs: {outs}")
        return 0.0


def calculate_college_linear_weights(pbp_df: pd.DataFrame, re24_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate linear weights from play-by-play data

    Args:
        pbp_df: Play-by-play DataFrame
        re24_matrix: Run expectancy matrix

    Returns:
        pd.DataFrame: Calculated linear weights
    """
    # Event mapping
    event_map = {
        2: 'out', 3: 'out', 6: 'out',
        14: 'walk', 16: 'hit_by_pitch',
        20: 'single', 21: 'double',
        22: 'triple', 23: 'home_run'
    }

    # Convert event codes and map to event types
    pbp_df['event_cd'] = pd.to_numeric(pbp_df['event_cd'], errors='coerce')
    pbp_df['event'] = pbp_df['event_cd'].map(
        lambda x: event_map.get(x, 'other'))

    # Ensure numeric types for base and out states
    pbp_df['base_cd_before'] = pd.to_numeric(
        pbp_df['base_cd_before'], errors='coerce')
    pbp_df['outs_before'] = pd.to_numeric(
        pbp_df['outs_before'], errors='coerce')

    # Calculate RE24 values
    re_start = pbp_df.apply(lambda x: get_re(
        x['base_cd_before'], x['outs_before'], re24_matrix), axis=1)

    next_base = pd.concat(
        [pbp_df['base_cd_before'].iloc[1:].astype(int), pd.Series([0])])
    next_outs = pd.concat(
        [pbp_df['outs_before'].iloc[1:].astype(int), pd.Series([0])])
    re_end = pd.Series([get_re(base, outs, re24_matrix)
                       for base, outs in zip(next_base, next_outs)])

    re_end[pbp_df['inn_end'] == 1] = 0

    # Calculate RE24
    re24 = re_end - re_start + pbp_df['runs_on_play']

    # Group and calculate metrics
    results = (pd.DataFrame({
        'event': pbp_df['event'],
        're24': re24
    })
        .groupby('event')
        .agg(
        count=('re24', 'count'),
        total_re24=('re24', 'sum')
    )
        .reset_index())

    # Filter out 'other' events
    results = results[results['event'] != 'other']

    # Calculate linear weights
    results['linear_weights'] = (
        results['total_re24'] / results['count']).round(3)

    # Calculate weights above average and above outs
    avg_weight = np.average(
        results['linear_weights'], weights=results['count'])
    out_weight = results.loc[results['event']
                             == 'out', 'linear_weights'].iloc[0]

    results['linear_weights_above_average'] = (
        results['linear_weights'] - avg_weight).round(3)
    results['linear_weights_above_outs'] = (
        results['linear_weights'] - out_weight).round(3)

    return results.sort_values('linear_weights', ascending=False)


def calculate_normalized_linear_weights(linear_weights: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized linear weights using league statistics

    Args:
        linear_weights: Linear weights DataFrame
        stats_df: Team/League statistics DataFrame

    Returns:
        pd.DataFrame: Normalized linear weights
    """
    # Calculate total value and PA
    total_value = (
        linear_weights['linear_weights_above_outs'] * linear_weights['count']).sum()
    total_pa = linear_weights['count'].sum()
    denominator = total_value / total_pa

    # Calculate league OBP
    total_stats = stats_df.sum()
    league_obp = (
        (total_stats['H'] + total_stats['BB'] + total_stats['HBP']) /
        (total_stats['AB'] + total_stats['BB'] +
         total_stats['HBP'] + total_stats['SF'] + total_stats['SH'])
    )

    # Calculate wOBA scale
    woba_scale = league_obp / denominator

    # Create normalized weights
    result = linear_weights.copy()
    result['normalized_weight'] = (
        result['linear_weights_above_outs'] * woba_scale).round(3)

    # Add wOBA scale row
    woba_scale_row = pd.DataFrame({
        'event': ['wOBA scale'],
        'count': [np.nan],
        'total_re24': [np.nan],
        'linear_weights': [np.nan],
        'linear_weights_above_average': [np.nan],
        'linear_weights_above_outs': [np.nan],
        'normalized_weight': [round(woba_scale, 3)]
    })

    return pd.concat([result, woba_scale_row], ignore_index=True)


def process_division(
    division: int,
    year: int,
    data_dir: Path,
    save_output: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single division's data

    Args:
        division: Division number (1-3)
        year: Year to process
        data_dir: Root data directory
        save_output: Whether to save results to file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Linear weights and normalized weights
    """
    # Setup paths
    misc_dir = data_dir / 'miscellaneous'
    division_str = str(division)
    year_str = str(year)

    # Use Path objects properly for concatenation
    pbp_file = data_dir / 'play_by_play' / \
        f"d{division_str}_parsed_pbp_{year_str}.csv"
    re24_file = misc_dir / f"d{division_str}_expected_runs_{year_str}.csv"
    stats_file = data_dir / 'stats' / f"d{division_str}_batting_{year_str}.csv"

    print(f"Looking for files:")
    print(f"PBP: {pbp_file}")
    print(f"RE24: {re24_file}")
    print(f"Stats: {stats_file}")

    # Validate files exist
    for file, desc in [
        (pbp_file, "Play-by-play"),
        (re24_file, "Expected runs matrix"),
        (stats_file, "Batting stats")
    ]:
        if not file.exists():
            raise FileNotFoundError(f"{desc} file not found: {file}")

    # Read data
    pbp_df = pd.read_csv(pbp_file)
    re24_df = pd.read_csv(re24_file)
    re24_matrix = re24_df.set_index('Bases')[['0', '1', '2']]
    stats_df = pd.read_csv(stats_file)

    print(f"Successfully loaded data:")
    print(f"PBP rows: {len(pbp_df)}")
    print(f"RE24 matrix shape: {re24_matrix.shape}")
    print(f"RE24 matrix index: {re24_matrix.index.tolist()}")
    print(f"Stats rows: {len(stats_df)}")

    # Calculate weights
    linear_weights = calculate_college_linear_weights(pbp_df, re24_matrix)
    normalized_weights = calculate_normalized_linear_weights(
        linear_weights, stats_df)

    # Save results if requested
    if save_output:
        output_file = misc_dir / \
            f"d{division_str}_linear_weights_{year_str}.csv"
        normalized_weights.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")

    return linear_weights, normalized_weights


def main(data_dir: str):
    """
    Main function to process all divisions

    Args:
        data_dir: Root directory containing the data folders
    """
    data_path = Path(data_dir)
    year = 2025
    divisions = range(1, 4)

    # Create miscellaneous directory if it doesn't exist
    misc_dir = data_path / 'miscellaneous'
    misc_dir.mkdir(exist_ok=True)

    for division in divisions:
        try:
            print(f"\nProcessing Division {division}...")
            linear_weights, normalized_weights = process_division(
                division=division,
                year=year,
                data_dir=data_path
            )
            print(f"Successfully processed Division {division}")
            print("\nLinear Weights:")
            print(linear_weights)
            print("\nNormalized Weights:")
            print(normalized_weights)

        except Exception as e:
            print(f"Error processing Division {division}:")
            print(f"  {str(e)}")
            continue


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate linear weights for college baseball')
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')

    args = parser.parse_args()
    main(args.data_dir)
