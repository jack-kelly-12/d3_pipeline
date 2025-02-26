import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from pathlib import Path


position_adjustments = {
    'SS': 1.85, 'C': 3.09, '2B': 0.62, '3B': 0.62, 'UT': 0.62,
    'CF': 0.62, 'INF': 0.62, 'LF': -1.85, 'RF': -1.85, '1B': -3.09,
    'DH': -3.09, 'OF': 0.25, 'PH': -0.74, 'PR': -0.74, 'P': 0.62,
    'RP': 0.62, 'SP': 0.62, '': 0
}

batting_columns = [
    'Player', 'Pos', 'B/T', 'player_id', 'Team', 'Conference', 'Yr', 'R/PA', 'GP',
    'BB', 'CS', 'GS', 'HBP', 'IBB', 'K', 'RBI', 'SF', 'AB',
    'PA', 'H', '2B', '3B', 'HR', 'R', 'SB', 'OPS+', 'Picked',
    'Sac', 'BA', 'SlgPct', 'OBPct', 'ISO', 'wOBA', 'K%', 'BB%',
    'SB%', 'wRC+', 'wRC', 'Batting', 'Baserunning', 'Adjustment', 'WAR',
    'Clutch', 'WPA', 'REA', 'WPA/LI', 'wSB', 'wGDP', 'wTEB', 'EBT', "Opportunities", 'OutsOB',
    'GDPOpps', 'GDP', 'Division', 'Season'
]

pitching_columns = [
    'Player', 'player_id', 'B/T', 'Team', 'Conference', 'App', 'GS', 'ERA', 'IP', 'H', 'R', 'ER',
    'BB', 'SO', 'HR-A', '2B-A', '3B-A', 'HB', 'BF', 'FO', 'GO', 'Pitches',
    'gmLI', 'K9', 'BB9', 'HR9', 'RA9', 'H9', 'IR-A%', 'K%', 'BB%', 'K-BB%', 'HR/FB', 'FIP',
    'xFIP', 'ERA+', 'WAR', 'Yr', 'Inh Run', 'Inh Run Score',
    'Clutch', 'pWPA', 'pREA', 'pWPA/LI', 'Division', 'Season'
]


def get_data(year, data_dir):
    pitching, batting, pbp, rosters, guts, park_factors = {}, {}, {}, {}, {}, {}

    for division in range(1, 4):
        pitching_df = pd.read_csv(
            data_dir / f'stats/d{division}_pitching_{year}.csv', dtype={'player_id': str})
        pitching[division] = pitching_df

        batting_df = pd.read_csv(
            data_dir / f'stats/d{division}_batting_{year}.csv', dtype={'player_id': str})
        batting[division] = batting_df

        roster_df = pd.read_csv(
            data_dir / f'rosters/d{division}_rosters_{year}.csv', dtype={'player_id': str}).query(
            f'year == {year}').query(f'division == {division}')
        rosters[division] = roster_df

        pbp_df = pd.read_csv(
            data_dir / f'play_by_play/d{division}_parsed_pbp_new_{year}.csv', dtype={'player_id': str, 'pitcher_id': str})
        pbp[division] = pbp_df

        pf_df = pd.read_csv(
            data_dir / f'park_factors/d{division}_park_factors.csv')
        park_factors[division] = pf_df

        guts_df = pd.read_csv(data_dir / 'guts/guts_constants.csv')
        guts_df = guts_df[guts_df['Division'] == division]
        guts_df = guts_df[guts_df['Year'] == int(year)]
        guts[division] = guts_df

    return batting, pitching, pbp, guts, park_factors, rosters


def calculate_war(data_dir, year):
    batting, pitching, pbp, guts, park_factors, rosters = get_data(
        year, data_dir)

    war_dir = Path(data_dir) / 'war'
    war_dir.mkdir(exist_ok=True)

    for division in range(1, 4):
        print(f"Processing Division {division}, Year {year}")

        batting_df = batting[division]
        pitching_df = pitching[division]
        pbp_df = pbp[division]
        guts_df = guts[division]
        park_factors_df = park_factors[division]
        rosters_df = rosters[division]

        batting_war, team_batting_clutch = calculate_batting_war(
            batting_df, guts_df, park_factors_df, pbp_df, rosters_df, division, year)

        pitching_war, team_pitching_clutch = calculate_pitching_war(
            pitching_df, pbp_df, park_factors_df, batting_war.WAR.sum(), year, division
        )

        batting_team_war = calculate_batting_team_war(
            batting_war, guts_df, park_factors_df, team_batting_clutch)
        pitching_team_war = calculate_pitching_team_war(
            pitching_war, park_factors_df, team_pitching_clutch)

        pitch_team_war_file = war_dir / \
            f'd{division}_pitching_team_war_{year}.csv'
        pitching_team_war.to_csv(pitch_team_war_file, index=False)

        batting_team_war_file = war_dir / \
            f'd{division}_batting_team_war_{year}.csv'
        batting_team_war.to_csv(batting_team_war_file, index=False)

        pitch_war_file = war_dir / \
            f'd{division}_pitching_war_{year}.csv'
        pitching_war.to_csv(pitch_war_file, index=False)

        batting_war_file = war_dir / \
            f'd{division}_batting_war_{year}.csv'
        batting_war.to_csv(batting_war_file, index=False)


def calculate_pitching_team_war(pitching_df, park_factors_df, team_clutch):
    df = pitching_df.copy()
    df = df.groupby('Team').agg({'App': 'sum', 'Conference': 'first', 'IP': 'sum', 'H': 'sum',
                                 '2B-A': 'sum', '3B-A': 'sum', 'Inh Run': 'sum', 'Inh Run Score': 'sum',
                                 'HR-A': 'sum', 'R': 'sum', 'ER': 'sum', 'WAR': 'sum',
                                 'FO': 'sum', 'BB': 'sum', 'HB': 'sum', 'SO': 'sum',
                                 'BF': 'sum', 'Pitches': 'sum', 'Season': 'first', 'Division': 'first'}).reset_index()

    def convert_to_baseball_notation(innings):
        parts = str(innings).split('.')
        whole_innings = int(parts[0])

        if len(parts) > 1 and parts[1] != '':
            outs = int(parts[1])
            decimal_part = outs / 3
        else:
            decimal_part = 0

        return whole_innings + decimal_part

    df['IP'] = df['IP'].apply(convert_to_baseball_notation)
    df['ERA'] = (df['ER'] * 9) / df['IP']
    df['IR-A%'] = ((df['Inh Run Score'] / df['Inh Run']) * 100).fillna(0)
    df['RA9'] = (df['R'] / df['IP']) * 9
    df['K9'] = (df['SO'] / df['IP']) * 9
    df['H9'] = (df['H'] / df['IP']) * 9
    df['BB9'] = (df['BB'] / df['IP']) * 9
    df['HR9'] = (df['HR-A'] / df['IP']) * 9
    df['BB%'] = (df['BB'] / df['BF']) * 100
    df['K%'] = (df['SO'] / df['BF']) * 100
    df['K-BB%'] = df['K%'] - df['BB%']
    df['HR/FB'] = (df['HR-A'] / (df['HR-A'] + df['FO'])) * 100

    lg_era = (df['ER'].sum() * 9) / df['IP'].sum()
    fip_components = ((13 * df['HR-A'].sum() + 3 * (df['BB'].sum() + df['HB'].sum()) -
                       2 * df['SO'].sum()) / df['IP'].sum())
    f_constant = lg_era - fip_components
    fip = f_constant + ((13 * df['HR-A'] + 3 * (df['BB'] + df['HB']) -
                        2 * df['SO']) / df['IP'])
    lg_hr_fb_rate = df['HR-A'].sum() / (df['HR-A'].sum() + df['FO'].sum())
    xfip = f_constant + ((13 * ((df['FO'] + df['HR-A']) * lg_hr_fb_rate) +
                          3 * (df['BB'] + df['HB']) - 2 * df['SO']) / df['IP'])

    df['FIP'] = fip
    df['xFIP'] = xfip

    df['PF'] = df['Team'].map(
        park_factors_df.set_index('team_name')['PF'])

    df['ERA+'] = 100 * \
        (2 - (df.ERA / ((df.ER.sum() / df.IP.sum()) * 9)) * (1 / (df.PF / 100)))

    df = df.merge(
        team_clutch,
        left_on='Team',
        right_on='pitch_team',
        how='left'
    )

    return df


def calculate_batting_team_war(batting_df, guts_df, park_factors_df, team_clutch):
    df = batting_df.copy()
    df = df.groupby('Team').agg({'GP': 'max', 'Conference': 'first', 'AB': 'sum', 'BB': 'sum',
                                'IBB': 'sum', 'SF': 'sum', 'HBP': 'sum', 'PA': 'sum', 'H': 'sum',
                                 '2B': 'sum', '3B': 'sum', 'HR': 'sum', 'R': 'sum', 'SB': 'sum',
                                 'Picked': 'sum', 'Sac': 'sum', 'wRC': 'sum', 'Batting': 'sum',
                                 'Baserunning': 'sum', 'Adjustment': 'sum', 'WAR': 'sum',
                                 'K': 'sum', 'CS': 'sum', 'RBI': 'sum', 'GS': 'max', 'Season': 'first',
                                 'Division': 'first'}).reset_index()
    weights = guts_df.iloc[0]

    df['PF'] = df['Team'].map(
        park_factors_df.set_index('team_name')['PF'])
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
    df['PA'] = df['AB'] + df['BB'] + df['IBB'] + df['HBP'] + df['SF']

    df['SB%'] = (df['SB'] / (df['SB'] + df['CS'])) * 100
    df['BB%'] = (df['BB'] / df['PA']) * 100
    df['K%'] = (df['K'] / df['PA']) * 100
    df['BA'] = df['H'] / df['AB']
    df['SlgPct'] = (df['H'] + df['2B'] +
                    2 * df['3B'] + 3 * df['HR']) / df['AB']
    df['OBPct'] = (df['H'] + df['BB'] + df['HBP'] +
                   df['IBB']) / (df['AB'] + df['BB'] +
                                 df['IBB'] + df['HBP'] + df['SF'])
    df['ISO'] = df['SlgPct'] - df['BA']
    lg_obp = (df['H'].sum() + df['BB'].sum() + df['HBP'].sum()) / \
        (df['AB'].sum() + df['BB'].sum() + df['HBP'].sum() + df['SF'].sum())
    lg_slg = (df['1B'].sum() + df['2B'].sum() * 2 +
              df['3B'].sum() * 3 + df['HR'].sum() * 4) / df['AB'].sum()

    df['OPS+'] = 100 * (df['OBPct'] / lg_obp +
                        df['SlgPct'] / lg_slg - 1)
    df['R/PA'] = df['R'] / df['PA']

    numerator = (weights['wBB'] * df['BB'] +
                 weights['wHBP'] * df['HBP'] +
                 weights['w1B'] * df['1B'] +
                 weights['w2B'] * df['2B'] +
                 weights['w3B'] * df['3B'] +
                 weights['wHR'] * df['HR'])
    denominator = df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP']
    df['wOBA'] = numerator / denominator

    df = df.merge(team_clutch,
                  left_on='Team',
                  right_on='bat_team',
                  how='left'
                  )

    return df


def calculate_pitching_war(pitching_df, pbp_df, park_factors_df, bat_war_total, year, division):
    if pitching_df.empty:
        return pitching_df
    df = pitching_df[pitching_df['App'] > 0].copy()

    df = df[df['ERA'].notna()]

    fill_cols = ['HR-A', 'FO', 'IP', 'BB',
                 'SO', 'SV', 'GS', 'HB', 'BF', 'H', 'R']
    df[fill_cols] = df[fill_cols].fillna(0)

    def safe_per_nine(numerator, ip):
        return np.where(ip > 0, (numerator / ip) * 9, 0)

    def safe_percentage(numerator, denominator):
        return np.where(denominator > 0, (numerator / denominator) * 100, 0)

    df['RA9'] = safe_per_nine(df['R'], df['IP'])
    df['K9'] = safe_per_nine(df['SO'], df['IP'])
    df['H9'] = safe_per_nine(df['H'], df['IP'])
    df['BB9'] = safe_per_nine(df['BB'], df['IP'])
    df['HR9'] = safe_per_nine(df['HR-A'], df['IP'])

    df['BB%'] = safe_percentage(df['BB'], df['BF'])
    df['K%'] = safe_percentage(df['SO'], df['BF'])
    df['K-BB%'] = df['K%'] - df['BB%']

    fb_total = df['HR-A'] + df['FO']
    df['HR/FB'] = safe_percentage(df['HR-A'], fb_total)

    df['IR-A%'] = safe_percentage(df['Inh Run Score'], df['Inh Run'])

    valid_ip_mask = df['IP'] > 0
    df.loc[~valid_ip_mask, ['FIP', 'xFIP']] = np.nan

    df['PF'] = df['team_name'].map(
        park_factors_df.set_index('team_name')['PF'])

    lg_era = (df.loc[valid_ip_mask, 'ER'].sum() /
              df.loc[valid_ip_mask, 'IP'].sum()) * 9
    df.loc[valid_ip_mask, 'ERA+'] = 100 * \
        (2 - (df.loc[valid_ip_mask, 'ERA'] / lg_era)
            * (1 / (df.loc[valid_ip_mask, 'PF'] / 100)))
    df.loc[~valid_ip_mask, 'ERA+'] = np.nan

    lg_era = (df['ER'].sum() * 9) / df['IP'].sum()

    fip_components = ((13 * df['HR-A'].sum() + 3 * (df['BB'].sum() + df['HB'].sum()) -
                       2 * df['SO'].sum()) / df['IP'].sum())
    f_constant = lg_era - fip_components

    fip = f_constant + ((13 * df['HR-A'] + 3 * (df['BB'] + df['HB']) -
                        2 * df['SO']) / df['IP'])

    lg_hr_fb_rate = df['HR-A'].sum() / (df['HR-A'].sum() + df['FO'].sum())

    xfip = f_constant + ((13 * ((df['FO'] + df['HR-A']) * lg_hr_fb_rate) +
                          3 * (df['BB'] + df['HB']) - 2 * df['SO']) / df['IP'])

    df['FIP'] = fip
    df['xFIP'] = xfip

    df['player_id'] = df['player_id'].astype(str)
    pbp_df['pitcher_id'] = pbp_df['pitcher_id'].astype(str)

    gmli = (pbp_df[pbp_df['description'].str.contains('to p', na=False)]
            .groupby(['pitcher_id'])
            .agg({'li': 'mean'})
            .reset_index()
            .rename(columns={'li': 'gmLI', 'pitch_team': 'Team'}))
    df.player_id = df.player_id.astype(str)
    df = df.merge(gmli, how='left',
                  left_on=['player_id'],
                  right_on=['pitcher_id'])

    df['gmLI'] = df['gmLI'].fillna(0.0)
    valid_ip_mask = df['IP'] > 0

    def calculate_if_fip_constant(group_df):
        group_df = group_df[group_df['IP'] > 0]
        if len(group_df) == 0:
            return np.nan

        lg_ip = group_df['IP'].sum()
        lg_hr = group_df['HR-A'].sum()
        lg_bb = group_df['BB'].sum()
        lg_hbp = group_df['HB'].sum()
        lg_k = group_df['SO'].sum()
        lg_era = (group_df['ER'].sum() / lg_ip) * 9

        numerator = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_k))
        return lg_era - (numerator / lg_ip)

    def calculate_player_if_fip(row, constant):
        if row['IP'] == 0:
            return np.nan
        numerator = ((13 * row['HR-A']) + (3 *
                                           (row['BB'] + row['HB'])) - (2 * row['SO']))
        return (numerator / row['IP']) + constant

    if_fip_constants = df[valid_ip_mask].groupby('conference').apply(
        calculate_if_fip_constant).reset_index()
    if_fip_constants.columns = ['conference', 'if_fip_constant']
    df = df.merge(if_fip_constants, on='conference', how='left')

    df['ifFIP'] = df.apply(lambda row: calculate_player_if_fip(
        row, row['if_fip_constant']), axis=1)

    valid_df = df[valid_ip_mask]
    if len(valid_df) > 0:
        lgRA9 = (valid_df['R'].sum() / valid_df['IP'].sum()) * 9
        lgERA = (valid_df['ER'].sum() / valid_df['IP'].sum()) * 9
        adjustment = lgRA9 - lgERA
    else:
        adjustment = 0

    df['FIPR9'] = np.where(valid_ip_mask, df['ifFIP'] + adjustment, np.nan)
    df['PF'] = df['PF'].fillna(100)
    df['pFIPR9'] = np.where(
        valid_ip_mask, df['FIPR9'] / (df['PF'] / 100), np.nan)

    def calculate_league_adjustments(group_df):
        valid_group = group_df[group_df['IP'] > 0]
        if len(valid_group) == 0:
            return np.nan

        lg_ip = valid_group['IP'].sum()
        lg_hr = valid_group['HR-A'].sum()
        lg_bb = valid_group['BB'].sum()
        lg_hbp = valid_group['HB'].sum()
        lg_k = valid_group['SO'].sum()

        lg_ifFIP = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) -
                    (2 * lg_k)) / lg_ip + valid_group['if_fip_constant'].iloc[0]

        lgRA9 = (valid_group['R'].sum() / lg_ip) * 9
        lgERA = (valid_group['ER'].sum() / lg_ip) * 9
        adjustment = lgRA9 - lgERA

        return lg_ifFIP + adjustment

    league_adjustments = df[valid_ip_mask].groupby('conference').apply(
        calculate_league_adjustments).reset_index()
    league_adjustments.columns = ['conference', 'conf_fipr9']
    df = df.merge(league_adjustments, on='conference', how='left')

    df['RAAP9'] = np.where(
        valid_ip_mask, df['conf_fipr9'] - df['pFIPR9'], 0)
    df['IP/G'] = np.where(df['App'] > 0, df['IP'] / df['App'], 0)
    df['dRPW'] = np.where(valid_ip_mask,
                          (((18 - df['IP/G']) * df['conf_fipr9'] +
                              df['IP/G'] * df['pFIPR9']) / 18 + 2) * 1.5,
                          0)

    df['WPGAA'] = np.where(valid_ip_mask, df['RAAP9'] / df['dRPW'], 0)
    df['gs/g'] = np.where(df['App'] > 0, df['GS'] / df['App'], 0)
    df['replacement_level'] = (
        0.03 * (1 - df['gs/g'])) + (0.12 * df['gs/g'])
    df['WPGAR'] = np.where(
        valid_ip_mask, df['WPGAA'] + df['replacement_level'], 0)
    df['WAR'] = np.where(valid_ip_mask, df['WPGAR'] * (df['IP'] / 9), 0)

    # Apply relief pitcher adjustment
    relief_mask = df['GS'] < 3
    df.loc[relief_mask & valid_ip_mask,
           'WAR'] *= (1 + df.loc[relief_mask & valid_ip_mask, 'gmLI']) / 2

    total_pitching_war = df['WAR'].sum()
    target_pitching_war = (bat_war_total * 0.43) / 0.57  # 43% of total WAR
    war_adjustment = (target_pitching_war - total_pitching_war) / \
        df.loc[valid_ip_mask, 'IP'].sum()

    df.loc[valid_ip_mask, 'WAR'] += war_adjustment * \
        df.loc[valid_ip_mask, 'IP']

    df = df.rename(columns={
        'player_name': 'Player',
        'conference': 'Conference',
        'team_name': 'Team'
    })

    df['Season'] = year
    df['Division'] = division

    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    df = df.fillna(0)
    df[['Player', 'Team', 'Conference', 'Yr']] = df[
        ['Player', 'Team', 'Conference', 'Yr']].fillna('-')

    pitcher_stats, team_stats = get_pitcher_clutch_stats(
        pbp_df)
    df.player_id = df.player_id.astype(str)
    df = df.merge(
        pitcher_stats,
        left_on=['player_id'],
        right_on=['pitcher_id'],
        how='left'
    )

    df = df[pitching_columns]

    return df.dropna(subset=['WAR']), team_stats


def calculate_wgdp(pbp_df):
    gdp_opps = pbp_df[(pbp_df['r1_name'] != '') & (
        pbp_df['outs_before'].astype(int) < 2)].copy()

    gdp_events = gdp_opps[gdp_opps['description'].str.contains('double play',
                                                               case=False,
                                                               na=False)]

    gdp_stats = pd.DataFrame({
        'GDPOpps': gdp_opps.groupby('batter_standardized').size(),
        'GDP': gdp_events.groupby('batter_standardized').size()
    }).fillna(0)

    lg_gdp_rate = gdp_stats['GDP'].sum() / gdp_stats['GDPOpps'].sum()
    gdp_run_value = -0.5

    gdp_stats['wGDP'] = (
        (gdp_stats['GDPOpps'] * lg_gdp_rate - gdp_stats['GDP']) *
        gdp_run_value
    )

    return gdp_stats


def calculate_extra_bases(df, roster, weights):
    extra_bases = {}
    opportunities = {}
    outs_on_bases = {}

    batting_lookup = {
        team: group.set_index('player_name')['player_id'].to_dict()
        for team, group in roster.groupby('team_name')
    }

    # Handle singles
    singles = df[df['event_cd'] == 20].copy()
    next_plays = singles.shift(-1)

    for idx, play in singles.iterrows():
        next_play = next_plays.loc[idx]
        team_dict = batting_lookup.get(play.bat_team, {})

        # Check first to third on single
        if pd.notna(play.r1_name) and team_dict:
            try:
                matches = process.extractOne(
                    play.r1_name,
                    list(team_dict.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=50
                )
                if matches:
                    standardized_r1 = matches[0]
                    # Count opportunity when runner starts on first
                    opportunities[standardized_r1] = opportunities.get(
                        standardized_r1, 0) + 1

                    if play.outs_on_play > 0 and play.r1_name not in [next_play.r1_name, next_play.r2_name, next_play.r3_name]:
                        # Runner was out on bases
                        outs_on_bases[standardized_r1] = outs_on_bases.get(
                            standardized_r1, 0) + 1
                    elif play.r1_name == next_play.r3_name:
                        # Runner advanced to third
                        extra_bases[standardized_r1] = extra_bases.get(
                            standardized_r1, 0) + 1
            except Exception:
                pass

        # Check second to home on single
        if pd.notna(play.r2_name) and team_dict:
            try:
                matches = process.extractOne(
                    play.r2_name,
                    list(team_dict.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=50
                )
                if matches:
                    standardized_r2 = matches[0]
                    opportunities[standardized_r2] = opportunities.get(
                        standardized_r2, 0) + 1

                    if play.outs_on_play > 0 and play.r2_name not in [next_play.r1_name, next_play.r2_name, next_play.r3_name]:
                        outs_on_bases[standardized_r2] = outs_on_bases.get(
                            standardized_r2, 0) + 1
                    elif (play.r2_name not in [next_play.r1_name, next_play.r2_name, next_play.r3_name] and
                            play.runs_on_play > 0):
                        extra_bases[standardized_r2] = extra_bases.get(
                            standardized_r2, 0) + 1
            except Exception:
                pass

    # Handle doubles
    doubles = df[df['event_cd'] == 21].copy()
    next_plays = doubles.shift(-1)

    for idx, play in doubles.iterrows():
        next_play = next_plays.loc[idx]
        team_dict = batting_lookup.get(play.bat_team, {})

        # Check first to home on double
        if pd.notna(play.r1_name) and team_dict:
            try:
                matches = process.extractOne(
                    play.r1_name,
                    list(team_dict.keys()),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=50
                )
                if matches:
                    standardized_r1 = matches[0]
                    # Count opportunity when runner starts on first
                    opportunities[standardized_r1] = opportunities.get(
                        standardized_r1, 0) + 1

                    if play.outs_on_play > 0 and play.r1_name not in [next_play.r1_name, next_play.r2_name, next_play.r3_name]:
                        # Runner was out on bases
                        outs_on_bases[standardized_r1] = outs_on_bases.get(
                            standardized_r1, 0) + 1
                    elif (play.r1_name not in [next_play.r1_name, next_play.r2_name, next_play.r3_name] and
                            play.runs_on_play > 0):
                        # Runner scored
                        extra_bases[standardized_r1] = extra_bases.get(
                            standardized_r1, 0) + 1
            except Exception:
                pass

    results = pd.DataFrame({
        'EBT': pd.Series(extra_bases),
        'OutsOB': pd.Series(outs_on_bases),
        'Opportunities': pd.Series(opportunities)
    }).fillna(0)

    results['Success_Rate'] = (
        results['EBT'] / results['Opportunities']).round(3)

    lg_teb_rate = results['EBT'].sum(
    ) / results['Opportunities'].sum()
    lg_out_rate = results['OutsOB'].sum() / \
        results['Opportunities'].sum()

    run_extra_base = 0.3
    runs_per_out = weights['runsOut']
    run_out = -1 * (2 * runs_per_out + 0.075)

    results['wTEB'] = (
        (results['EBT'] * run_extra_base) +
        (results['OutsOB'] * run_out) -
        (results['Opportunities'] * (lg_teb_rate *
                                     run_extra_base + lg_out_rate * run_out))
    )

    return results.sort_values('EBT', ascending=False)


def get_clutch_stats(pbp_df):
    pbp_df = pbp_df.copy()
    pbp_df['player_id'] = pbp_df['player_id'].astype(str)

    player_stats = pbp_df.groupby(['player_id']).agg({
        'rea': 'sum',
        'wpa': 'sum',
        'wpa/li': 'sum',
        'li': 'mean'
    }).reset_index().sort_values('wpa/li', ascending=False).reset_index(drop=True)

    player_stats = player_stats.rename(
        columns={'wpa': 'WPA', 'wpa/li': 'WPA/LI', 'rea': 'REA'})

    player_stats['Clutch'] = np.where(
        player_stats['li'] == 0,
        np.nan,
        (player_stats['WPA'] / player_stats['li']) - player_stats['WPA/LI']
    )

    team_stats = pbp_df.groupby('bat_team').agg({
        'rea': 'sum',
        'wpa': 'sum',
        'wpa/li': 'sum',
        'li': 'mean'
    }).reset_index().sort_values('wpa/li', ascending=False).reset_index(drop=True)

    team_stats = team_stats.rename(
        columns={'wpa': 'WPA', 'wpa/li': 'WPA/LI', 'rea': 'REA'})

    team_stats['Clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['WPA'] / team_stats['li']) - team_stats['WPA/LI']
    )

    return player_stats, team_stats


def get_pitcher_clutch_stats(pbp_df):
    pbp_df = pbp_df.copy()
    pbp_df['pitcher_id'] = pbp_df['pitcher_id'].astype(str)

    pbp_df['pREA'] = (-pbp_df['run_expectancy_delta'] -
                      pbp_df['runs_on_play'])
    pbp_df['pWPA'] = np.where(pbp_df['pitch_team'] == pbp_df['home_team'],
                              pbp_df['delta_home_win_exp'],
                              -pbp_df['delta_home_win_exp'])
    pbp_df['pWPA/LI'] = pbp_df['pWPA'].div(
        pbp_df['li'].replace(0, float('nan')))

    pitcher_stats = pbp_df.groupby(['pitcher_id']).agg({
        'pREA': 'sum',
        'pWPA': 'sum',
        'pWPA/LI': 'sum',
        'li': 'mean',
        'pitcher_standardized': 'first',
    }).reset_index().sort_values('pWPA/LI', ascending=False).reset_index(drop=True)

    pitcher_stats = pitcher_stats[pitcher_stats['pitcher_standardized'] != "Starter"]

    pitcher_stats['Clutch'] = np.where(
        pitcher_stats['li'] == 0,
        np.nan,
        (pitcher_stats['pWPA'] / pitcher_stats['li']) -
        pitcher_stats['pWPA/LI']
    )

    team_stats = pbp_df.groupby(['pitch_team']).agg({
        'pREA': 'sum',
        'pWPA': 'sum',
        'pWPA/LI': 'sum',
        'li': 'mean'
    }).reset_index().sort_values('pWPA/LI', ascending=False).reset_index(drop=True)

    team_stats['Clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['pWPA'] / team_stats['li']) -
        team_stats['pWPA/LI']
    )

    return pitcher_stats, team_stats


def calculate_batting_war(batting_df, guts_df, park_factors_df, pbp_df, rosters_df, division, year):
    if batting_df.empty:
        return batting_df

    weights = guts_df.iloc[0]

    df = batting_df.copy()
    df['player_id'] = df['player_id'].astype(str)
    df['Pos'] = df['Pos'].apply(lambda x: '' if pd.isna(
        x) else str(x).split('/')[0].upper())

    gdp_stats = calculate_wgdp(pbp_df)
    teb_stats = calculate_extra_bases(pbp_df, rosters_df, weights)

    df = df.merge(
        gdp_stats,
        left_on='player_name',
        right_index=True,
        how='left'
    )

    df = df.merge(
        teb_stats,
        left_on='player_name',
        right_index=True,
        how='left'
    )

    df[['wGDP', 'GDPOpps', 'GDP']] = df[[
        'wGDP', 'GDPOpps', 'GDP']].fillna(0)

    fill_cols = ['HR', 'R', 'GP', 'GS', '2B', '3B', 'H', 'CS', 'BB', 'K',
                 'SB', 'IBB', 'RBI', 'Picked', 'SH', 'AB', 'HBP', 'SF']
    df[fill_cols] = df[fill_cols].fillna(0)

    df['GP'] = pd.to_numeric(
        df['GP'], errors='coerce').fillna(0).astype(int)
    df['GS'] = pd.to_numeric(
        df['GS'], errors='coerce').fillna(0).astype(int)
    df = df[df['AB'] > 0]

    df['PF'] = df['team_name'].map(
        park_factors_df.set_index('team_name')['PF'])
    df['1B'] = df['H'] - df['HR'] - df['3B'] - df['2B']
    df['PA'] = df['AB'] + df['BB'] + df['IBB'] + df['HBP'] + df['SF']

    df['SB%'] = (df['SB'] / (df['SB'] + df['CS'])) * 100
    df['BB%'] = (df['BB'] / df['PA']) * 100
    df['K%'] = (df['K'] / df['PA']) * 100
    df['ISO'] = df['SlgPct'] - df['BA']
    lg_obp = (df['H'].sum() + df['BB'].sum() + df['HBP'].sum()) / \
        (df['AB'].sum() + df['BB'].sum() + df['HBP'].sum() + df['SF'].sum())
    lg_slg = (df['1B'].sum() + df['2B'].sum() * 2 +
              df['3B'].sum() * 3 + df['HR'].sum() * 4) / df['AB'].sum()

    df['OPS+'] = 100 * (df['OBPct'] / lg_obp +
                        df['SlgPct'] / lg_slg - 1)
    df['R/PA'] = df['R'] / df['PA']

    numerator = (weights['wBB'] * df['BB'] +
                 weights['wHBP'] * df['HBP'] +
                 weights['w1B'] * df['1B'] +
                 weights['w2B'] * df['2B'] +
                 weights['w3B'] * df['3B'] +
                 weights['wHR'] * df['HR'])
    denominator = df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP']
    df['wOBA'] = numerator / denominator

    df['PF'].fillna(100, inplace=True)
    pf = df['PF'] / 100

    league_woba = weights['wOBA']
    league_rpa = (df['R'].sum() / df['PA'].sum())
    pa = df['PA']
    woba = df['wOBA']
    woba_scale = weights['wOBAScale']
    rpw = weights['runsWin']
    runs_per_out = weights['runsOut']
    runCS = -1 * (2 * runs_per_out + 0.075)
    runSB = 0.2

    conf_wrc = {}
    for conf in df['conference'].unique():
        conf_df = df[df['conference'] == conf]
        if len(conf_df) > 0:
            conf_wrc[conf] = conf_df['R'].sum() / conf_df['PA'].sum()

    def calculate_batting_runs(row):
        pf = row['PF'] / 100
        conf_rpa = conf_wrc.get(row['conference'], league_rpa)

        return (row['wRAA'] +
                (league_rpa - (pf * league_rpa)) * row['PA'] +
                (league_rpa - conf_rpa) * row['PA'])

    df['wRC'] = (((woba - league_woba) / woba_scale) + league_rpa) * pa
    league_wrcpa = (df['wRC'].sum() / df['PA'].sum())
    df['wRAA'] = ((woba - league_woba) / woba_scale) * pa
    df['batting_runs'] = df.apply(calculate_batting_runs, axis=1)

    lgwSB = ((df['SB'].sum() * runSB + df['CS'].sum() * runCS) /
             (df['1B'].sum() + df['BB'].sum() + df['HBP'].sum() - df['IBB'].sum()))

    df['wSB'] = (df['SB'] * runSB + df['CS'] * runCS -
                 lgwSB * (df['1B'] + df['BB'] + df['HBP'] - df['IBB']))

    team_count = len(df['team_name'].unique())
    games_played = (df['GS'].sum() / 9) / team_count
    replacement_constant = ((team_count / 2) * games_played -
                            (team_count * games_played * 0.294))
    df['replacement_level_runs'] = replacement_constant * \
        (rpw / pa.sum()) * rpw
    df['baserunning'] = df['wSB'] + df['wGDP'] + df['wTEB']

    base_adjustments = df['Pos'].map(position_adjustments).fillna(0)
    df['Adjustment'] = base_adjustments * \
        (df['GP'] / (np.where(division == 3, 40, 50)))

    conf_adjustments = {}
    for conf in df['conference'].unique():
        conf_df = df[df['conference'] == conf]
        if len(conf_df) > 0:
            lg_batting = conf_df['batting_runs'].sum()
            lg_baserunning = conf_df['wSB'].sum()
            lg_positional = conf_df['Adjustment'].sum()
            lg_pa = conf_df['PA'].sum()
            if lg_pa > 0:
                conf_adjustments[conf] = (-1 * (lg_batting + lg_baserunning +
                                                lg_positional) / lg_pa)

    df['league_adjustment'] = df.apply(
        lambda x: conf_adjustments.get(x['conference'], 0) * x['PA'], axis=1
    )

    df['WAR'] = ((df['batting_runs'] + df['replacement_level_runs'] +
                  df['baserunning'] + df['Adjustment'] +
                  df['league_adjustment']) / rpw)

    wraa_pa = df['wRAA'] / df['PA']
    league_wrcpa = (df['wRC'].sum() / df['PA'].sum())
    df['wRC+'] = (((wraa_pa + league_rpa) +
                   (league_rpa - pf * league_rpa)) / league_wrcpa) * 100

    df = df.rename(columns={
        'player_name': 'Player',
        'team_name': 'Team',
        'SH': 'Sac',
        'batting_runs': 'Batting',
        'baserunning': 'Baserunning',
        'conference': 'Conference'
    })

    player_stats, team_stats = get_clutch_stats(pbp_df)
    df.player_id = df.player_id.astype(str)
    df = df.merge(
        player_stats,
        left_on=['player_id'],
        right_on=['player_id'],
        how='left'
    )

    df['Season'] = year
    df['Division'] = division

    df[['Player', 'Team', 'Conference', 'Yr']] = df[
        ['Player', 'Team', 'Conference', 'Yr']].fillna('-')
    df = df.fillna(0)

    df = df[batting_columns]

    return df.dropna(subset=['WAR']), team_stats


def main(data_dir, year):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    try:
        calculate_war(data_dir, year)
        print("Successfully processed all statistics!")
    except Exception as e:
        print(f"Error processing statistics: {str(e)}")
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    parser.add_argument('--year', required=True)
    args = parser.parse_args()

    main(args.data_dir, args.year)
