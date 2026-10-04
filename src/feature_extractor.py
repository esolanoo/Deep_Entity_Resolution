"""


Candidate Pair (Record A, Record B)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 [ Record A Text ]              [ Record B Text ]
       │                               │
       ▼                               ▼
 ┌───────────┐                   ┌───────────┐
 │ Module A  │                   │ Module A  │   <-- 1D Char CNN or TF-IDF
 └─────┬─────┘                   └─────┬─────┘       (Shared Weights / Siamese)
       │ Embedding Vector eA           │ Embedding Vector eB
       └───────────────┬───────────────┘
                       │
                       ▼
    [ Pair Feature Vector Formulation ]
       V_pair = [ eA , eB , |eA - eB| , eA ⊙ eB ]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from funcs import set_env, get_project_root, get_device
from etl import get_data
from blocking import generate_candidates

set_env()

data, cluster_data = get_data()
candidate_pairs = generate_candidates() 

pairs = candidate_pairs.copy()
records = cluster_data.copy()
cluster_of = records["cluster_id"]
duplicate_of = records["is_duplicate"]
    
pairs["cluster_1"] = pairs["record_id_1"].map(cluster_of)
pairs["cluster_2"] = pairs["record_id_2"].map(cluster_of)

pairs["label"] = (
    pairs["cluster_1"] == pairs["cluster_2"]
).astype(np.int64)

true_pairs = pairs[pairs['label']==1]
false_pairs = pairs[pairs['label']==0]

true_train, true_test = train_test_split(true_pairs, train_size=0.8)
false_train, false_test = train_test_split(false_pairs, train_size=0.8)

train_pairs_idx = pd.concat([true_train, false_train]).index
test_pairs_idx = pd.concat([true_test, false_test]).index

train_pairs = candidate_pairs.filter(items=train_pairs_idx, axis=0)
test_pairs = candidate_pairs.filter(items=test_pairs_idx, axis=0)

print(train_pairs.sample(5))
print(train_pairs.size)



