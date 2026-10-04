"""
### Data Pipelines & Candidate Generation
* Parse SPIDER v2 into a unified schema per entity domain (e.g., handling missing fields, text normalization).
* Implement a fast blocking strategy to restrict candidate pairs from O(N^2) to a manageable subset while preserving high recall (>95%).
* Plan:
                            Raw Records (N = 50,000)
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
                         (Reduces ~1.25B pairs to ~400k pairs)
                         
Results:
Pairs: lsh_blocker, Candidate Pairs: 371997, Recall: 97.88, Reduction Ratio: 99.9702
Pairs: block_key_1, Candidate Pairs: 9463, Recall: 86.69, Reduction Ratio: 99.9992
Pairs: block_key_2, Candidate Pairs: 8111, Recall: 77.13, Reduction Ratio: 99.9994
Pairs: block_key_3, Candidate Pairs: 13993, Recall: 78.41, Reduction Ratio: 99.9989
Pairs: C_total, Candidate Pairs: 378655, Recall: 100.00, Reduction Ratio: 99.9697
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


def get_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate blocking keys for each record based on multiple strategies.
    """
    
    df_blocking_keys = df.copy()
    
    # Combine Soundex of last name with postal code
    df_blocking_keys['soundex_last'] = df_blocking_keys['last_name'].apply(lambda x: soundex(x))
    df_blocking_keys['block_key_1'] = df_blocking_keys['soundex_last'] + '_' + df_blocking_keys['city']
    
    # First name and dob combination
    df_blocking_keys['block_key_2'] = df_blocking_keys['first_name'].str[:3] + '_' + df_blocking_keys['dob']
        
    # Last name and email combination
    df_blocking_keys['block_key_3'] = df_blocking_keys['last_name'].str[:3] + '_' + df_blocking_keys['email_user']
    
    return df_blocking_keys


def multipass_rule_blocking(df: pd.DataFrame, key_nums: int = 3) -> set:
    """
    Performs multipass rule-based blocking.
    """
    df = get_blocking_keys(df)
    pairs = set()
    df_with_keys = get_blocking_keys(df)
    
    for i in range(1, key_nums+1):
        block_key = f"block_key_{i}"
        grouped = df_with_keys.groupby(block_key)['record_id'].apply(list) # Group by blocking key and collect record IDs
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
        rec_id = row['record_id']
        text = f"{row['first_name']}_{row['last_name']}_{row['email_user']}_{row['street_name']}"
        
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
     
    
def evaluate_candidates(df: pd.DataFrame, C_total:  list):
    """
    Evaluates both rule-based and LSH blocking candidates.
    """
    
    total_true_pairs = 0
    for _, group in df.groupby('cluster_id'):
        if len(group) > 1:
            total_true_pairs += len(list(combinations(group['record_id'], 2)))
    total_possible_pairs = len(df) * (len(df) - 1) / 2
    
    
    for i in range(0, 3+2):
        pairs = set()
        if i==0:
            key_column = "lsh_blocker"
            pairs = lsh_blocking(df) 
        elif i==4:
            key_column = "C_total"
            pairs = set(C_total)
        else: 
            key_column = f"block_key_{i}"
            df_with_keys = get_blocking_keys(df)
            grouped = df_with_keys.groupby(key_column)['record_id'].apply(list) # Group by blocking key and collect record IDs
            for group in grouped:
                if 1 < len(group) <= 100:
                    for pair in combinations(sorted(group),2):
                        pairs.add(pair)
            
        cluster_map = df.set_index('record_id')['cluster_id'].to_dict()
        captured_true_pairs = sum(1 for a, b in pairs if cluster_map[a] == cluster_map[b])
        
        reduction_ratio = 1.0 - (len(pairs) / total_possible_pairs)
        recall = captured_true_pairs / total_true_pairs if total_true_pairs > 0 else 0
        
        print (f"Pairs: {key_column}, Candidate Pairs: {len(pairs)}, Recall: {100.00*recall:.2f}, Reduction Ratio: {100.00*reduction_ratio:.4f}")
    
    
def generate_candidates(eval=False) -> pd.DataFrame:
    """
    Generates candidate pairs using both rule-based and LSH blocking.
    """
    
    path = get_project_root() + r"\data\pairs\blocking_candidates.txt"
    
    if not os.path.exists(path): # Check if file exists
        print("Generating Candidate Pairs using Rule-based and LSH Blocking...")
        data, _ = get_data()
        C_rule = multipass_rule_blocking(data)
        C_lsh = lsh_blocking(data)
        C_total = C_rule.union(C_lsh)
        C_total = list(C_total)
        with open(path, 'w') as file:
            for pair in C_total:
                file.write(f"{pair[0]},{pair[1]}\n")
                
    else: # Load file
        print("Loading Candidate Pairs from file...")
        C_total = list()
        with open(path, 'r') as file:
            for line in file:
                rec1, rec2 = line.strip().split(',')
                C_total.append((rec1, rec2))
                
    print(f"Total Candidate Pairs from Union of Rule-based and LSH Blocking: {len(C_total)}")
    
    if eval:
        print("Performing evaluation...")
        data, _= get_data()
        evaluate_candidates(data, C_total)

    C_total = pd.DataFrame(data=C_total, columns=["record_id_1", "record_id_2"])
    return C_total


def train_test_sets():
    data, cluster_data = get_data()
    candidate_pairs = generate_candidates() 

    pairs = candidate_pairs.copy()
    records = cluster_data.copy()
    cluster_of = records["cluster_id"]
        
    pairs["cluster_1"] = pairs["record_id_1"].map(cluster_of)
    pairs["cluster_2"] = pairs["record_id_2"].map(cluster_of)

    pairs["label"] = (
        pairs["cluster_1"] == pairs["cluster_2"]
    ).astype(np.int64)
    
    pairs.drop(columns=['cluster_1', 'cluster_2'], inplace=True)

    true_pairs = pairs[pairs['label']==1]
    false_pairs = pairs[pairs['label']==0]

    true_train, true_test = train_test_split(true_pairs, train_size=0.8)
    false_train, false_test = train_test_split(false_pairs, train_size=0.8)

    train_pairs = pd.concat([true_train, false_train])
    test_pairs = pd.concat([true_test, false_test])
    
    return data, train_pairs, test_pairs
