import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score


def make_heatmap(df, cols, rows, title):
    corr_df = df.loc[:, cols + rows].corr(method='pearson', min_periods=1)[cols][len(cols):]
    return sns.heatmap(data=corr_df, annot=True, fmt='.4f').set(title=title)


def make_row_df(df, cols_to_include, cols_to_merge, main_col, main_col_new_name=''):
    if not main_col_new_name:
        main_col_new_name = main_col

    temp_df = init_row_df(cols_to_include, main_col_new_name)
    for col in cols_to_merge:
        temp_df = add_to_row_df(temp_df, df[[main_col, col]], col)
    return temp_df


def init_row_df(col_names, diff_col_name):
    df = pd.DataFrame(columns=col_names + [diff_col_name])
    return df


def add_to_row_df(row_df, df_to_add, diff_record):
    col_names = row_df.columns
    temp = df_to_add.set_axis(col_names[0:-1], axis='columns')
    temp.insert(len(col_names) - 1, col_names[-1], diff_record)

    df = row_df.append(temp)

    return df


def rename_cols(df):
    return df.rename(columns={
        'PAR_2017_VOL_SEZNAM': 'reg_voters_17',
        'PAR_2017_ODEVZ_OBAL': 'votes_17',
        'PrumerzVOLICI': 'reg_voters_13',
        'PrumerzHLASYCEL': 'votes_13'
    })


def make_attendance(df: pd.DataFrame):
    adf = df.assign(
        attendance_13=lambda row: 100 * row['votes_13'] / row['reg_voters_13'],
        attendance_17=lambda row: 100 * row['votes_17'] / row['reg_voters_17']
    )
    return adf


def test_add_feature(df, model_type, y, known_features, new_feature=None):
    kf = known_features
    if new_feature:
        kf.append(new_feature)
    new_x = df[kf]
    new_model = model_type().fit(new_x, y)
    n_score = cross_val_score(new_model, new_x, y, cv=4).mean()
    return n_score


def optimise_features(df, model_type, targets, initial_features, features_to_try, debug_print=False):
    features = initial_features
    y = df[targets]
    old_score = test_add_feature(df, model_type, y, features)
    print(f'Initial score with only 2013 attendance: {old_score}')
    for current_feature in features_to_try:
        new_score = test_add_feature(df, model_type, y, features, current_feature)
        if new_score > old_score:
            if debug_print:
                print(f'Adding feature {current_feature}, new score is {new_score}')
            features.append(current_feature)
            old_score = new_score
        elif debug_print:
            print(f'Ignoring feature {current_feature}')

    print(f'Obtained score {old_score}')
    if debug_print:
        print(f'Selected features: {features}')
    return old_score, features
