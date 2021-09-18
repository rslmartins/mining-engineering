import pandas as pd
import datetime

import warnings
warnings.filterwarnings("ignore")


"""
Load data and merge
"""
df_raw = pd.read_csv('md_raw_dataset.csv', sep = ";")
df_raw.rename({'Unnamed: 0': 'index'}, axis=1, inplace=True)

df_target = pd.read_csv('md_target_dataset.csv', sep = ";")
df_target['groups'] = df_target['groups'].map(lambda x: float(x))


"""
Keep a Raw Dataframe
"""
df_raw = pd.merge(df_target, df_raw, on=['index', 'groups'], how='left')

"""
Create a new Dataframe to manipulate
"""
df = df_raw

"""
Set columns as category data and transform into continous data
"""
df['super_hero_group'] = df['super_hero_group'].astype('category')
df['Cycle'] = df['Cycle'].astype('category')
df['crystal_supergroup'] = df['crystal_supergroup'].astype('category')

cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

"""
Transform categorical group data into continuous data
"""
df['crystal_type'] = df['crystal_type'].dropna().apply(lambda x: int(x.replace('group ','')))

"""
Transform target data to perform classification
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
This data is droped due its bad date appointing to 1969"
"""
df['opened'].dropna().apply(lambda x: datetime.datetime.fromtimestamp(float(x)/ 1e3) if '.' in x and type(x)==str else x )
df = df.drop(columns=['opened'])

"""
Transform two date columns into one using its seconds difference
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
        df[new_column][id] = (df[column1][id]-df[column2][id]).seconds
        
    df = df.drop(columns=[column1, column2])
        
    return df


df = make_column(df,'predicted_process_end','process_end','diff_process_end')
df = make_column(df,'start_critical_subprocess1','start_subprocess1','diff_subprocess1')
df = make_column(df,'expected_start','start_process','diff_process')
df = make_column(df,'reported_on_tower','subprocess1_end','diff_on_tower')

