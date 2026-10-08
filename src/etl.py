import recordlinkage.datasets
import pandas as pd
from funcs import set_env

set_env()


def get_entity_map(records: pd.DataFrame, true_pairs: pd.MultiIndex):
    data = records.copy()
    parent = {rec_id: rec_id for rec_id in data.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for rec_a, rec_b in true_pairs:
        union(rec_a, rec_b)

    roots = {rec_id: find(rec_id) for rec_id in data.index} # Compress paths
    # Convert arbitrary roots into clean integer entity IDs
    root_to_entity = {root: entity_idx for entity_idx, root in enumerate(sorted(set(roots.values())))}
    entity_map = pd.Series({rec_id: root_to_entity[root] for rec_id, root in roots.items()}, name="entity_id")

    return entity_map


def get_data():    
    data = recordlinkage.datasets.load_febrl2(return_links=True)
    records = data[0]
    true_pairs = data[1]
    records = records.drop(columns='soc_sec_id') # 'soc_sec_id' causes almost always an exact-match for true pairs (89 variations out of 4000 entities)
    num_fields = ['street_number', 'postcode', 'date_of_birth']

    for field in records.columns:
        records[field] = records[field].astype(str) # str casting for blockers
        records[field] = records[field].str.lower().str.strip()
        if field in num_fields:
            records[field] = records[field].str.zfill(int(max(records[field].str.len()))) # Homogenize lengths
    records.fillna("_MISSING_", inplace=True) # Several missing values but I don't want to drop them, rather make them explicitly show they are missing
    
    entity_map = get_entity_map(records, true_pairs) # type: ignore
    
    return records, true_pairs, entity_map