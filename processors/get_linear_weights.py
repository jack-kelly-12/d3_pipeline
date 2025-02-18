import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


def get_re(base_cd, outs, re_matrix) -> np.ndarray:
    result = np.zeros_like(base_cd, dtype=float)
    valid_mask = ~(np.isnan(base_cd) | np.isnan(outs))

    if valid_mask.any():
        base_states = np.array(
            ['_ _ _', '1 _ _', '_ 2 _', '1B 2B _', '_ _ 3', '1 _ 3', '_ 2 3', '1B 2B 3B'])
        base_idx = np.clip(base_cd[valid_mask].astype(int), 0, 7)
        base_strings = base_states[base_idx]
        outs_valid = np.clip(outs[valid_mask].astype(int), 0, 2)

        # Look up values using matrix indexing
        for i, (base, out) in enumerate(zip(base_strings, outs_valid)):
            idx = valid_mask.nonzero()[0][i]
            try:
                result[idx] = re_matrix.loc[base, str(out)]
            except KeyError:
                continue

    return result


def calculate_college_linear_weights(df_pbp, df_er):
    event_mapping = {
        '2': 'out', '3': 'out', '6': 'out',
        '14': 'walk', '16': 'hit_by_pitch',
        '20': 'single', '21': 'double',
        '22': 'triple', '23': 'home_run'
    }

    df_pbp['events'] = df_pbp['event_cd'].astype(
        str).map(event_mapping).fillna('other')

    df_pbp.events.unique()

    re_start = get_re(df_pbp['base_cd_before'].values,
                      df_pbp['outs_before'].values, df_er)

    re_end = np.zeros_like(re_start)
    re_end[:-1] = get_re(df_pbp['base_cd_before'].values[1:],
                         df_pbp['outs_before'].values[1:], df_er)

    re_end[df_pbp['inn_end'] == 1] = 0
    re24 = re_end - re_start + df_pbp['runs_on_play'].values
    event_stats = (pd.DataFrame({'events': df_pbp['events'], 're24': re24})
                   .groupby('events')
                   .agg(
        count=('re24', 'count'),
        total_re24=('re24', 'sum')
    )
        .reset_index())

    event_stats['linear_weights_above_average'] = (event_stats['total_re24'] /
                                                   event_stats['count']).round(3)

    final_stats = (event_stats[event_stats['events'] != 'other']
                   .assign(linear_weights_above_outs=lambda x:
                           x['linear_weights_above_average'] -
                           x.loc[x['events'] == 'out', 'linear_weights_above_average'].iloc[0])
                   .sort_values('linear_weights_above_average', ascending=False)
                   .reset_index(drop=True))

    return final_stats


def calculate_normalized_linear_weights(linear_weights: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['events', 'linear_weights_above_outs', 'count']
    if not all(col in linear_weights.columns for col in required_columns):
        raise ValueError(
            f"linear_weights must contain columns: {required_columns}")

    woba_scale_exists = "wOBA scale" in linear_weights['events'].values
    if woba_scale_exists:
        woba_scale_row = linear_weights[linear_weights['events']
                                        == "wOBA scale"]
        linear_weights = linear_weights[linear_weights['events']
                                        != "wOBA scale"]

    total_value = (linear_weights['linear_weights_above_outs'] *
                   linear_weights['count']).sum()

    total_pa = linear_weights['count'].sum()
    denominator = total_value / total_pa

    league_obp = (stats['H'].sum() + stats['BB'].sum() + stats['HBP'].sum()) / \
                 (stats['AB'].sum() + stats['BB'].sum() + stats['HBP'].sum() +
                  stats['SF'].sum() + stats['SH'].sum())

    woba_scale = league_obp / denominator

    normalized_weights = linear_weights.copy()
    normalized_weights['normalized_weight'] = (linear_weights['linear_weights_above_outs'] *
                                               woba_scale).round(3)

    woba_scale_row = pd.DataFrame({
        'events': ['wOBA scale'],
        'linear_weights_above_outs': [np.nan],
        'count': [np.nan],
        'normalized_weight': [round(woba_scale, 3)]
    })

    result = pd.concat([normalized_weights, woba_scale_row], ignore_index=True)

    return result


def process_division(
    division: int,
    year: int,
    data_dir: Path,
    save_output: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    misc_dir = data_dir / 'miscellaneous'
    division_str = str(division)
    year_str = str(year)

    pbp_file = data_dir / 'play_by_play' / \
        f"d{division_str}_parsed_pbp_{year_str}.csv"
    re24_file = misc_dir / f"d{division_str}_expected_runs_{year_str}.csv"
    stats_file = data_dir / 'stats' / f"d{division_str}_batting_{year_str}.csv"

    for file, desc in [
        (pbp_file, "Play-by-play"),
        (re24_file, "Expected runs matrix"),
        (stats_file, "Batting stats")
    ]:
        if not file.exists():
            raise FileNotFoundError(f"{desc} file not found: {file}")

    pbp_df = pd.read_csv(pbp_file)
    re24_matrix = pd.read_csv(re24_file)
    stats_df = pd.read_csv(stats_file)

    linear_weights = calculate_college_linear_weights(pbp_df, re24_matrix)
    normalized_weights = calculate_normalized_linear_weights(
        linear_weights, stats_df)

    if save_output:
        output_file = misc_dir / \
            f"d{division_str}_linear_weights_{year_str}.csv"
        normalized_weights.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")


def main(data_dir: str):
    data_path = Path(data_dir)
    year = 2025
    divisions = range(1, 4)

    misc_dir = data_path / 'miscellaneous'
    misc_dir.mkdir(exist_ok=True)

    for division in divisions:
        try:
            print(f"\nProcessing Division {division}...")
            process_division(
                division=division,
                year=year,
                data_dir=data_path
            )
            print(f"Successfully processed Division {division}")

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
