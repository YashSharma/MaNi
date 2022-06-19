import random
import pandas as pd

def load_data(source='kric', target='tnbc'):

    if source == 'kric':
        SOURCE_DOMAIN = '/data/kric_split.csv'
        SOURCE_BASE = #PATH to KIRC Image
    elif source == 'tnbc':
        SOURCE_DOMAIN = '/data/tnbc_split.csv'
        SOURCE_BASE = #PATH to TNBC Image
    elif source == 'tcia':
        SOURCE_DOMAIN = '/data/tcia_split.csv'
        SOURCE_BASE = #PATH to TCIA Image
        
    if target == 'kric':
        TARGET_DOMAIN = '/data/kric_split.csv'
        DOMAIN_KEY = 'KRIC'
        TARGET_BASE = #PATH to KIRC Image
    elif target == 'tnbc':
        TARGET_DOMAIN = '/data/tnbc_split.csv'
        DOMAIN_KEY = 'TNBC'        
        TARGET_BASE = #PATH to TNBC Image
    elif target == 'tcia':
        TARGET_DOMAIN = '/data/tcia_split.csv'
        DOMAIN_KEY = 'TCIA'        
        TARGET_BASE = #PATH to TCIA Image
    
    df_source = pd.read_csv(SOURCE_DOMAIN)
    file_list = df_source['fname'].apply(lambda x: SOURCE_BASE+x).tolist()

    df_split = pd.read_csv(TARGET_DOMAIN)

    train_val_list = df_split['split'].tolist()
    random.shuffle(train_val_list)
    df_split['split'] = train_val_list

    valid_file_list = df_split.loc[df_split['split']=='valid', 'fname'].apply(lambda x: TARGET_BASE+x).tolist()
    domain_file_list = df_split.loc[df_split['split']=='train', 'fname'].apply(lambda x: TARGET_BASE+x).tolist()
    
    test_file_list = df_split.loc[df_split['split']=='test', 'fname'].apply(lambda x: TARGET_BASE+x).tolist()
    
    return file_list, valid_file_list, domain_file_list, DOMAIN_KEY, test_file_list    