import pandas as pd
import os


def calculate_woba_constants(lw_df, batting_df):
    woba_scale = lw_df[lw_df['events'] ==
                       'wOBA scale']['normalized_weight'].iloc[0]

    weights = {
        'wBB': lw_df[lw_df['events'] == 'walk']['normalized_weight'].iloc[0],
        'wHBP': lw_df[lw_df['events'] == 'hit_by_pitch']['normalized_weight'].iloc[0],
        'w1B': lw_df[lw_df['events'] == 'single']['normalized_weight'].iloc[0],
        'w2B': lw_df[lw_df['events'] == 'double']['normalized_weight'].iloc[0],
        'w3B': lw_df[lw_df['events'] == 'triple']['normalized_weight'].iloc[0],
        'wHR': lw_df[lw_df['events'] == 'home_run']['normalized_weight'].iloc[0]
    }

    woba_numerator = (
        batting_df['BB'].sum() * weights['wBB'] +
        batting_df['HBP'].sum() * weights['wHBP'] +
        batting_df['1B'].sum() * weights['w1B'] +
        batting_df['2B'].sum() * weights['w2B'] +
        batting_df['3B'].sum() * weights['w3B'] +
        batting_df['HR'].sum() * weights['wHR']
    )

    woba_denominator = batting_df['AB'].sum(
    ) + batting_df['BB'].sum() + batting_df['HBP'].sum() + batting_df['SF'].sum()
    wOBA = woba_numerator / woba_denominator if woba_denominator > 0 else 0

    return {'wOBA': round(wOBA, 3), 'wOBAScale': woba_scale, **weights}


def calculate_baserunning_constants(pbp_df):
    runsOut = pbp_df['runs_on_play'].sum() / pbp_df['outs_on_play'].sum()
    runSB = 0.2
    runCS = -(2 * runsOut + 0.075)

    cs_attempts = len(pbp_df[pbp_df['event_cd'] == 6])
    sb_attempts = len(pbp_df[pbp_df['event_cd'] == 4])
    csRate = cs_attempts / \
        (cs_attempts + sb_attempts) if (cs_attempts + sb_attempts) > 0 else 0

    return {
        'runSB': runSB,
        'runCS': runCS,
        'csRate': round(csRate, 3)
    }


def calculate_run_constants(pbp_df):
    runsPA = pbp_df['runs_on_play'].sum(
    ) / len(pbp_df[~pbp_df['bat_order'].isna()])
    runsOut = pbp_df['runs_on_play'].sum() / pbp_df['outs_on_play'].sum()
    runsWin = pbp_df.groupby('game_id')['runs_on_play'].sum().mean()

    return {
        'runsPA': runsPA,
        'runsOut': runsOut,
        'runsWin': runsWin
    }


def calculate_fip_constant(pitching_df):
    lgERA = (pitching_df['ER'].sum() * 9) / pitching_df['IP'].sum()
    fip_components = (13 * pitching_df['HR-A'].sum() +
                      3 * (pitching_df['BB'].sum() + pitching_df['HB'].sum()) -
                      2 * pitching_df['SO'].sum()) / pitching_df['IP'].sum()
    return lgERA - fip_components


def calculate_guts_constants(division, year, output_path):
    try:
        pbp_df = pd.read_csv(
            output_path / f'play_by_play/d{division}_parsed_pbp_new_{year}.csv')
        lw_df = pd.read_csv(
            output_path / f'miscellaneous/d{division}_linear_weights_{year}.csv')

        pitching_df = pd.read_csv(output_path /
                                  f'stats/d{division}_pitching_{year}.csv')
        batting_df = pd.read_csv(output_path /
                                 f'stats/d{division}_batting_{year}.csv')

        batting_df['1B'] = batting_df['H'] - batting_df['2B'] - \
            batting_df['3B'] - batting_df['HR']

        constants = {
            'Year': year,
            'Division': division,
            **calculate_woba_constants(lw_df, batting_df),
            **calculate_baserunning_constants(pbp_df),
            **calculate_run_constants(pbp_df),
            'cFIP': calculate_fip_constant(pitching_df)
        }

        return constants

    except Exception as e:
        print(f"Error calculating constants for D{division} {year}: {e}")
        return None


def main(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'guts')):
        os.makedirs(os.path.join(data_dir, 'guts'))
    divisions = [1, 2, 3]
    year = 2025

    all_constants = []

    for division in divisions:
        constants = calculate_guts_constants(
            division, year, data_dir)
        if constants:
            all_constants.append(constants)

    guts_df = pd.DataFrame(all_constants).sort_values(
        ['Division', 'Year'], ascending=[True, False])

    guts_df.to_csv(f'{data_dir}/guts/guts_constants.csv', index=False)
    print(f"Saved {len(guts_df)} rows of Guts constants")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    main(args.data_dir)
