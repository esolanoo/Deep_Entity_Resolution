"""
### Data Pipelines & Candidate Generation
* Parse SPIDER v2 into a unified schema per entity domain (e.g., handling missing fields, text normalization).
* Implement a fast blocking strategy to restrict candidate pairs from O(N^2) to a manageable subset while preserving high recall (>95%).
* Plan:
                            Raw Records (N = 50,000)
                                         |
               +-------------------------+-------------------------+
               |                                                   |
    [ Multi-Pass Rule Blocking ]                        [ MinHash LSH Blocking ]
    • Pass 1: Soundex(Last) + Zip                       • Tokenize: Char 3-grams (shingles)
    • Pass 2: Email Domain + City                       • Signatures: K=128
    • Pass N: other rules                               • Bands: b=32, r=4 (t≈0.42)
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                            Union Set: C_total = C_rule ∪ C_lsh
                                         |
                            Evaluated Candidate Pairs (C)
                         (Reduces ~1.25B pairs to ~500k pairs)
"""
import pandas as pd
from funcs import set_env
from etl import get_data

set_env()

def get_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Generate blocking keys for each record based on multiple strategies."""
    df_blocking = df.copy()
    
    # Soundex for last name
    df_blocking['soundex_last'] = df_blocking['last_name'].apply(lambda x: soundex(x))
    
    # Combine Soundex of last name with postal code
    df_blocking['block_key_1'] = df_blocking['soundex_last'] + '_' + df_blocking['postal_code']
    
    # Email domain and city combination
    df_blocking['block_key_2'] = df_blocking['email_domain'] + '_' + df_blocking['city']
    
    # Additional blocking keys can be added here as needed
    
    return df_blocking[['block_key_1', 'block_key_2']]


data = get_data(processed=True)