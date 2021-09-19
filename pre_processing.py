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
Transform target data to perform classification

target
[1.044776119, 1.079136691, 1.096774194, 1.125827815, 1.137724551, 1.140939597, 1.156462585, 1.19047619, ..., 268.1879195, 268.7022901, 270.0, 270.8015267, 272.9007634, 275.0]
"""
df['target'] = df['target'].map(lambda x: 1 if x>= 0 and x <100 else x)
df['target'] = df['target'].map(lambda x: 2 if x>= 100 and x < 200 else x)
df['target'] = df['target'].map(lambda x: 3 if x>= 150 and x < 200 else x)
df['target'] = df['target'].map(lambda x: 4 if x>= 200 and x < 250 else x)
df['target'] = df['target'].map(lambda x: 5 if x>= 250 and x <= 275 else x)
df['target'].unique()

"""
Transform date into days counting until the present date
"""
today = datetime.datetime.now().date()
df['when'] = df['when'].dropna().apply(lambda x: (today-datetime.datetime.strptime(x,'%d/%m/%Y').date()).days )

"""
This data is droped due its bad date appointing to 1969-12-31 21:00:44.021674, for example"
"""
df['opened'].dropna().apply(lambda x: datetime.datetime.fromtimestamp(float(x)/ 1e3) if '.' in x and type(x)==str else x )
df = df.drop(columns=['opened'])

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

