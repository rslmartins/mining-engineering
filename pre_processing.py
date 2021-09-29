import pandas as pd
import datetime

import warnings
warnings.filterwarnings("ignore")


"""
Load data and merge
"""
def load_raw():
	df_raw = pd.read_csv('md_raw_dataset.csv', sep = ";")
	df_raw.rename({'Unnamed: 0': 'index'}, axis=1, inplace=True)
	
	df_target = pd.read_csv('md_target_dataset.csv', sep = ";")
	df_target['groups'] = df_target['groups'].map(lambda x: float(x))
	
	df_raw = pd.merge(df_target, df_raw, on=['index', 'groups'], how='left')
	return df_raw

"""
Create a new dataframe as df to manipulate, and keep a raw dataframe as df_raw
"""
df = load_raw()
df_raw = load_raw()

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
df['crystal_type'] = df['crystal_type'].dropna().apply(lambda x: int(x.replace('group ','')))


"""
Transform date into days counting until the present date
"""
today = datetime.datetime.now().date()
df['when'] = df['when'].dropna().apply(lambda x: (today-datetime.datetime.strptime(x,'%d/%m/%Y').date()).days )


"""
This data is droped due its bad date appointing to 1969-12-31 21:00:44.021674, for example"
"""
df['opened'] = df['opened'].dropna().apply(lambda x: float("NaN") if '.' in str(x) else x )
df['opened'] = df['opened'].dropna().apply(lambda x: (today-datetime.datetime.strptime(x,'%d/%m/%Y %H:%M').date()).days )

"""
Transform two date columns into one using its seconds difference
For, example

subprocess1_end starts with '09/07/2020 13:27'
and reported_on_tower starts with '09/07/2020 13:37'
then, diff_on_tower is 600 seconds


"""
def make_column(df,column1,column2,new_column):
    try:
        df[column1] = df[column1].dropna().apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%M'))
    except:
        pass
    try: df[column2] = df[column2].dropna().apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H:%M'))
    except:
        pass

    df[new_column] = None
        
    for id in df.index:
        if df[column1][id] >= df[column2][id]: df[new_column][id] = (df[column1][id]-df[column2][id]).seconds/60
        if df[column1][id] < df[column2][id]: df[new_column][id] = (df[column2][id]-df[column1][id]).seconds/60
                
    df = df.drop(columns=[column1, column2])
        
    return df


df = make_column(df,'predicted_process_end','process_end','diff_process_end')
df = make_column(df,'start_critical_subprocess1','start_subprocess1','diff_subprocess1')
df = make_column(df,'expected_start','start_process','diff_process')
df = make_column(df,'reported_on_tower','subprocess1_end','diff_on_tower')
