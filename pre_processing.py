import pandas as pd
import datetime
from utils import Preprocessing as pp
import warnings
warnings.filterwarnings("ignore")


class Preparing:
    def data_prepared() -> None:
        """
        Create a new dataframe as df to manipulate, and keep a raw dataframe as df_raw
        """
        df = pp.load_raw()
        df_raw = pp.load_raw()

        """
        Set columns as category data and transform into continous data

        super_hero_group
        ['A', 'B', 'C', 'D', 'G', 'W', 'Y', '₢']

        crystal_supergroup
        ['0', '1', '1ª']

        Cycle
        ['131', '1ª', '2ª', '33', '3ª']

        """
        df['super_hero_group'] = df['super_hero_group'].astype('category')
        df['Cycle'] = df['Cycle'].astype('category')
        df['crystal_supergroup'] = df['crystal_supergroup'].astype('category')

        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        """
        Transform categorical group data into continuous data

        crystal_type
        ['group 0', 'group 1', 'group 10', 'group 100', 'group 101', ..., 'group 99']

        """
        df['crystal_type'] = df['crystal_type'].dropna().apply(lambda x: int(x.replace('group ', '')))
        """
        Transform date into days counting until the present date
        """
        today = datetime.datetime.now().date()
        df['when'] = df['when'].dropna().apply(lambda x: (today-datetime.datetime.strptime(x, '%d/%m/%Y').date()).days)
        """
        This data is droped due its bad date appointing to 1969-12-31 21:00:44.021674, for example"
        """
        df['opened'] = df['opened'].dropna().apply(lambda x: float("NaN") if '.' in str(x) else x)
        df['opened'] = df['opened'].dropna().apply(lambda x: (today-datetime.datetime.strptime(x, '%d/%m/%Y %H:%M').date()).days)
        """
        Transform two date columns into one using its seconds difference
        For, example

        subprocess1_end starts with '09/07/2020 13:27'
        and reported_on_tower starts with '09/07/2020 13:37'
        then, diff_on_tower is 600 seconds
        """
        df = pp.make_column(df, 'predicted_process_end', 'process_end', 'diff_process_end')
        df = pp.make_column(df, 'start_critical_subprocess1', 'start_subprocess1', 'diff_subprocess1')
        df = pp.make_column(df, 'expected_start', 'start_process', 'diff_process')
        df = pp.make_column(df, 'reported_on_tower', 'subprocess1_end', 'diff_on_tower')

        # Remove target column from dataframe before pre-processing (df) and after pre-processing (df_raw)
        df_target = df['target']
        df = df.drop(columns=['target'])
        df_raw = df_raw.drop(columns=['target'])
        return df, df_raw, df_target
