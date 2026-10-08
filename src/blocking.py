"""
### Data Pipelines & Candidate Generation
* Parse FEBRL-2 into a unified schema per entity domain 
* Implement a fast blocking strategy to restrict candidate pairs from O(N^2) to a manageable subset while preserving high recall (>95%).
* Plan:
                            Raw Records (N = 5,000)
                                         |
                              train-val-test split
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               |                                        [ MinHash LSH Blocking ]
    [ Multi-Pass Rule Blocking ]                        • Tokenize: Char n-grams (shingles)
    • Pass N rules                                      • Signatures: K=128
               |                                        • Bands: b=32, rows=4
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                            Union Set: C_total = C_rule ∪ C_lsh
                                         |
                            Evaluated Candidate Pairs (C)
                         (Reduces 12,497,500 pairs to ~29K pairs)
                         
Results:
LSH
Candidate pairs : 28,144
True links      : 1,934
Captured links  : 1,703
Recall          : 88.06%
Reduction ratio : 99.77%

Rule
Candidate pairs : 1,862
True links      : 1,934
Captured links  : 1,638
Recall          : 84.69%
Reduction ratio : 99.99%

Union
Candidate pairs : 28,548
True links      : 1,934
Captured links  : 1,889
Recall          : 97.67%
Reduction ratio : 99.77%
"""

import pandas as pd
import numpy as np
from itertools import combinations
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split
import os
from funcs import set_env, get_project_root
from etl import get_data


set_env()

def soundex(name: str) -> str:
    """
    Compute the Soundex code for a given name.
    This is based on american Soundex algorithm (given the dataset is regional-us), which encodes homophones to the same representation so that they can be matched despite minor differences in spelling.
    """
    
    name = name.upper()
    soundex_mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
        'A': '0', 'E': '0', 'I': '0', 'O': '0', 'U': '0', 'H': '0', 'W': '0', 'Y': '0'
    }
    
    first_letter = name[0]
    tail = name[1:]
    
    digits = [soundex_mapping.get(char, '') for char in tail]
    
    # Remove consecutive duplicates
    filtered_digits = []
    previous_digit = ''
    for digit in digits:
        if digit != previous_digit and digit != '':
            filtered_digits.append(digit)
            previous_digit = digit
    
    soundex_code = first_letter + ''.join(filtered_digits)
    soundex_code = (soundex_code + "000")[:4] # Pad with zeros or truncate to ensure length is 4
    
    return soundex_code


def multipass_rule_blocking(df: pd.DataFrame, key_nums: int = 3) -> set:
    """
    Performs multipass rule-based blocking.
    """
    
    pairs = set()
    df_with_keys = df.copy()
    
    # Combine Soundex of surname with postcode
    df_with_keys['soundex_surname'] = df_with_keys['surname'].apply(lambda x: soundex(x))
    df_with_keys['block_key_1'] = df_with_keys['soundex_surname'] + '_' + df_with_keys['postcode']
    
    # Given name and dob combination
    df_with_keys['block_key_2'] = df_with_keys['given_name'].str[:3] + '_' + df_with_keys['date_of_birth']
        
    # Surname and suburb
    df_with_keys['block_key_3'] = df_with_keys['surname'].str[:3] + '_' + df_with_keys['suburb']
    
    for i in range(1, key_nums+1):
        block_key = f"block_key_{i}"
        grouped = df_with_keys.groupby(block_key)['rec_id'].apply(list) # Group by blocking key and collect record IDs
        for group in grouped:
            if 1 < len(group) <= 100: # Cap max number of records in group to avoid combinatorial explosion
                for pair in combinations(sorted(group),2):
                    pairs.add(pair)
    
    return pairs


def lsh_blocking(df: pd.DataFrame, num_perm: int = 128, bands: int = 32, rows: int = 4, shingle_size: int = 3) -> set:
    """
    Performs MinHash LSH blocking
    """
    
    minhashes = {}
    threshold = round((1/bands)**(1/rows),4)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    for _, row in df.iterrows():
        m = MinHash(num_perm=num_perm)
        rec_id = row['rec_id']
        text = f"{row['given_name'][:3]}_{row['address_1']}_{row['address_2']}_{row['suburb']}"
        
        shingles = [text[i:i+shingle_size] for i in range(len(text)-shingle_size+1)] # Create shingles
        
        for shingle in shingles:
            m.update(shingle.encode('utf8'))
            
        lsh.insert(rec_id, m)
        minhashes[rec_id] = m
        
    pairs = set()
    for rec_id, m in minhashes.items():
        neighbors = lsh.query(m)
        for n in neighbors:
            if rec_id != n:
                pair = tuple(sorted((rec_id, n))) # type: ignore
                pairs.add(pair)
        
    return pairs
    

def generate_candidates(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Generates candidate pairs using both rule-based and LSH blocking.
    """
    data.reset_index(inplace=True)
    path = get_project_root() + rf"\data\pairs\blocking_candidates_{name}.txt"
    
    if not os.path.exists(path): # Check if file exists
        print(f"Generating Candidate Pairs using Rule-based and LSH Blocking for {name}...")  
        C_rule = multipass_rule_blocking(data)
        C_lsh = lsh_blocking(data)
        C_total = C_rule.union(C_lsh)
        C_total = list(C_total)
        with open(path, 'w') as file:
            for pair in C_total:
                file.write(f"{pair[0]},{pair[1]}\n")
                
    else: # Load file
        print(f"Loading {name} Candidate Pairs from file...")
        C_total = list()
        with open(path, 'r') as file:
            for line in file:
                rec1, rec2 = line.strip().split(',')
                C_total.append((rec1, rec2))

    C_total = pd.DataFrame(data=C_total, columns=["rec_id_1", "rec_id_2"])
    return C_total


def add_labels(pairs: pd.DataFrame, entity_map: pd.Series) -> pd.DataFrame:
    
    pairs = pairs.copy()
    pairs["entity_1"] = pairs["rec_id_1"].map(entity_map)
    pairs["entity_2"] = pairs["rec_id_2"].map(entity_map)
    if pairs["entity_1"].isna().any():
        raise ValueError("Some rec_id_1 values are missing from entity_map.")
    if pairs["entity_2"].isna().any():
        raise ValueError("Some rec_id_2 values are missing from entity_map.")
    pairs["label"] = ( pairs["entity_1"] == pairs["entity_2"]).astype("int64")
    pairs.drop(columns=["entity_1", "entity_2"], inplace=True)

    return pairs.sample(frac=1).reset_index(drop=True)

def evaluate_blocking(candidate_pairs: set, true_pairs: pd.MultiIndex, data: pd.DataFrame):

    candidate_pairs = {tuple(sorted(pair)) for pair in candidate_pairs}
    true_pair_set = {tuple(sorted(pair)) for pair in true_pairs}
    captured = (candidate_pairs & true_pair_set)

    recall = 100*len(captured)/len(true_pair_set) 
    total_possible = (len(data)*(len(data) - 1) // 2)
    reduction_ratio =100* (1-len(candidate_pairs)/total_possible)

    print(f"Candidate pairs : {len(candidate_pairs):,}")
    print(f"True links      : {len(true_pair_set):,}")
    print(f"Captured links  : {len(captured):,}")
    print(f"Recall          : {recall:.2f}%")
    print(f"Reduction ratio : {reduction_ratio:.2f}%")
    

def train_test_sets(eval=False):
    data, true_pairs, entity_map = get_data()
    entity_ids = entity_map.unique()
    
    train_entities, temp_entities = train_test_split(entity_ids, train_size=0.8,)
    val_entities, test_entities = train_test_split(temp_entities, train_size=0.5)

    train_records = data.loc[entity_map.isin(train_entities)].copy()
    val_records = data.loc[entity_map.isin(val_entities)].copy()
    test_records = data.loc[entity_map.isin(test_entities)].copy()

    train_pairs = generate_candidates(train_records, name="train") # type: ignore
    test_pairs = generate_candidates(test_records, name="test") # type: ignore
    val_pairs = generate_candidates(val_records, name="val") # type: ignore
    
    train_pairs = add_labels(train_pairs, entity_map)
    test_pairs = add_labels(test_pairs, entity_map)
    val_pairs = add_labels(val_pairs, entity_map)
    
    if eval:
        data.reset_index(inplace=True)
        C_lsh = lsh_blocking(data) # type: ignore
        C_rule = multipass_rule_blocking(data) # type: ignore
        C_total = C_lsh | C_rule

        print("LSH")
        evaluate_blocking(C_lsh, true_pairs, data) # type: ignore
        print("\nRule")
        evaluate_blocking(C_rule, true_pairs, data) # type: ignore
        print("\nUnion")
        evaluate_blocking(C_total, true_pairs, data) # type: ignore
    
    return data, train_pairs, test_pairs, val_pairs
