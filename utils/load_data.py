import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path)

    # drop useless cols/rows
    df = df.drop(columns=['Unnamed: 0','id','url','locality','roomCount']) # use postal code for location; roomCount: seems that people fill it wrongly
    df = df[df['type'].isin(['APARTMENT', 'HOUSE'])] # drop apartment_group and house_group
    df = df.dropna(subset=['price']) # drop price rows of NaN

    # drop duplicated rows
    df = df.drop_duplicates()

    # drop completely empty rows/cols
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')

    X = df.drop(columns=['price'])
    y = df['price']

    print('end of load_and_clean_data')
    return X, y