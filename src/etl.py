import pandas as pd
from funcs import set_env, get_project_root

set_env()

def get_data():
    """Load the dataset and optionally preprocess it."""
    path = get_project_root() + r"\data\raw\spider_dataset_v2_6_20251027_022215.csv"
    df = pd.read_csv(path, dtype={'postal_code': str})
    return preprocess_records(df)

def preprocess_records(df: pd.DataFrame):
    """Clean and normalize string columns for blocking keys and hashing."""
    df_clean = df.copy()
    df_clean.set_index('record_id', inplace=True, drop=True)
    
    # Ensure postal code and phone are zero-padded strings
    df_clean['postal_code'] = df_clean['postal_code'].astype(str).str.zfill(5)
    df_clean['phone'] = df_clean['phone'].astype(str).str.replace(r'\D', '', regex=True)  # Remove non-digit characters
    df_clean['phone'] = df_clean['phone'].str.zfill(10)
    
    # Create standardized lower-case combined representations
    df_clean['first_name'] = df_clean['first_name'].astype(str).str.lower().str.strip()
    df_clean['last_name'] = df_clean['last_name'].astype(str).str.lower().str.strip()
    df_clean['city'] = df_clean['city'].astype(str).str.lower().str.strip().str.replace(' ','')
    
    # DOB normalization to YYYY-MM-DD format
    df_clean['dob'] = pd.to_datetime(df_clean['dob'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_clean['dob'] = df_clean['dob'].astype(str).str.replace(r'\D', '', regex=True) 
    
    # Split adress into components for more granular blocking
    df_clean[['street_number', 'street_name']] = df_clean['street'].str.split(' ', expand=True, n=1)
    df_clean['street_number'] = df_clean['street_number'].astype(str).str.zfill(5)
    df_clean['apt_number'] = df_clean['street_name'].str.replace(r'\D', '', regex=True).str.zfill(3)
    df_clean['street_name'] = df_clean['street_name'].str.replace(' ','').str.lower().str.strip().str.replace('apt.','')
    df_clean['street_name'] = df_clean['street_name'].str.replace(r'\d', '', regex=True) # remove digits
    df_clean.drop(columns=['street'], inplace=True)
    
    # State cleaning
    df_clean['state'] = df_clean['state'].astype(str).str.lower().str.strip()
    
    # Email cleaning
    df_clean[['email_user', 'email_domain']] = df_clean['email'].str.split('@', expand=True, n=1).astype(str)
    df_clean['email_domain'] = df_clean['email_domain'].str.split(".").str[0]
    df_clean['email_user'] = df_clean['email_user'].str.lower().str.strip().str.replace('.','')
    df_clean['email_user'] = df_clean['email_user'].apply(lambda x: x[:x.find('+')] if '+' in x else x)
    df_clean.drop(columns=['email'], inplace=True)
    
    cluster_columns = ['is_duplicate', 'cluster_id', 'is_duplicate_of', 'rule_id', 'rule_category']
    cluster_data = df_clean[cluster_columns]
    df_clean.drop(columns=cluster_columns, inplace=True)
    
    return df_clean, cluster_data
