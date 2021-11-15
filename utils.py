import pandas as pd
import seaborn as sns


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
    adf = df.dropna(subset=['reg_voters_17', 'votes_17', 'reg_voters_13', 'votes_13'])
    adf = adf.assign(
        attendance_13=lambda row: 100 * row['votes_13'] / row['reg_voters_13'],
        attendance_17=lambda row: 100 * row['votes_17'] / row['reg_voters_17']
    )
    return adf

