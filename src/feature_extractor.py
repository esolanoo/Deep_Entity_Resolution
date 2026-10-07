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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import time
import string
from sklearn.metrics import precision_score, recall_score, f1_score
from funcs import get_device
from blocking import train_test_sets


PAD_IDX = 0
UNK_IDX = 1
chars = string.ascii_lowercase + string.digits + " .,/-'"
char_to_idx = { char: idx + 2 for idx, char in enumerate(chars) }
vocab_size = len(char_to_idx) + 2


def char_tokenizer(text, max_len=26):
      text = str(text).lower()
      tokens = []
      for char in text:
            tokens.append(char_to_idx.get(char, UNK_IDX))
      tokens = tokens[:max_len] # truncate
      tokens += [PAD_IDX] * (max_len - len(tokens)) # padding

      return torch.tensor(tokens, dtype=torch.long)


class EntityPairDataset(Dataset):
      def __init__(self, data, pairs, tokenizer, fields, max_len=50):
            self.data = data
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.fields = fields
            self.max_len = max_len

      def __len__(self):
            return len(self.pairs)

      def __getitem__(self, idx):
            row = self.pairs.iloc[idx]

            record_1 = self.data.loc[row["record_id_1"]]
            record_2 = self.data.loc[row["record_id_2"]]

            features_A = {}
            features_B = {}

            for field in self.fields:
                  if field!='label':
                        text_A = str(record_1[field])
                        text_B = str(record_2[field])

                        features_A[field] = self.tokenizer(text_A, self.max_len)
                        features_B[field] = self.tokenizer(text_B, self.max_len)

            label = torch.tensor(row["label"], dtype=torch.float32)

            return features_A, features_B, label


def data():
      data, train_pairs, test_pairs, val_pairs = train_test_sets()
      data = data.set_index('record_id', drop=True)
      fields = data.columns.to_list()
      train_dataset = EntityPairDataset(
            data=data,
            pairs=train_pairs,
            tokenizer=char_tokenizer,
            fields=fields,
            max_len=23
      )

      test_dataset = EntityPairDataset(
            data=data,
            pairs=test_pairs,
            tokenizer=char_tokenizer,
            fields=fields,
            max_len=23
      )

      val_dataset = EntityPairDataset(
            data=data,
            pairs=val_pairs,
            tokenizer=char_tokenizer,
            fields=fields,
            max_len=23
      )
      
      train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True
      )

      test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False
      )

      val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False
      )
      
      return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader

class CharCNN(nn.Module):
      def __init__(self, vocab_size=vocab_size, embed_dim=32, num_filters=64, output_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
            # Parallel 1D Convolutions capturing 3-gram, 5-gram, and 7-gram char features
            self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
            self.conv7 = nn.Conv1d(embed_dim, num_filters, kernel_size=7, padding=3)
            
            self.fc = nn.Sequential(
                  nn.Linear(num_filters * 3, output_dim),
                  nn.BatchNorm1d(output_dim),
                  nn.ReLU()
            )
            
      def forward(self, x):
            # x shape: (batch_size, max_seq_len)
            x_embed = self.embedding(x).transpose(1, 2)  # (batch_size, embed_dim, max_seq_len)
            
            feat3 = torch.relu(self.conv3(x_embed)).max(dim=2)[0]  # Global Max Pooling
            feat5 = torch.relu(self.conv5(x_embed)).max(dim=2)[0]
            feat7 = torch.relu(self.conv7(x_embed)).max(dim=2)[0]
            
            concat_feats = torch.cat([feat3, feat5, feat7], dim=1)
            out_embedding = self.fc(concat_feats)
            return out_embedding  # Vector shape: (batch_size, 128)


class MLPMatcher(nn.Module):
      def __init__(self, embed_dim=128, num_fields=12, hidden_dim=128):
            super().__init__()
            # Input size is 4 * embed_dim due to concat(eA, eB, |eA - eB|, eA * eB)
            input_dim = embed_dim * 4 * num_fields
            
            self.classifier = nn.Sequential(
                  nn.Linear(input_dim, hidden_dim),
                  nn.BatchNorm1d(hidden_dim),
                  nn.ReLU(),
                  nn.Dropout(0.3),
                  nn.Linear(hidden_dim, hidden_dim // 2),
                  nn.ReLU(),
                  # nn.Linear(hidden_dim // 2, 1),
                  # nn.ReLU(),
                  nn.Linear(hidden_dim // 2, 1)
                  # nn.Sigmoid()
            )
            
      def forward(self, embeddings_A, embeddings_B):
            field_comparisons = []
            for field in embeddings_A.keys():
                  embedding_A = embeddings_A[field]
                  embedding_B = embeddings_B[field]
                  diff = torch.abs(embedding_A - embedding_B)
                  prod = embedding_A * embedding_B
                  field_vector = torch.cat(
                        [embedding_A, embedding_B, diff, prod ],
                        dim=1
                  )
                  field_comparisons.append(field_vector)

            pair_vector = torch.cat(field_comparisons, dim=1)
            logits = self.classifier(pair_vector)
            return logits.squeeze(-1)


class EntityMatcher(nn.Module):
      def __init__(self, vocab_size=vocab_size, fields=[], embedding_dim=128):
            super().__init__()
            self.fields = fields

            self.encoders = nn.ModuleDict({
                  field: CharCNN(
                        vocab_size = vocab_size,
                        embed_dim = 32,
                        num_filters = 64,
                        output_dim = embedding_dim
                  ) for field in fields
            })

            self.matcher = MLPMatcher(
                  embed_dim = embedding_dim,
                  num_fields = len(fields),
                  hidden_dim = 28
            )

      def forward(self, features_A, features_B, return_embeddings=False):
            embeddings_A = {}
            embeddings_B = {}

            for field in self.fields:
                  embeddings_A[field] = self.encoders[field](features_A[field])
                  embeddings_B[field] = self.encoders[field](features_B[field])
            match_prob = self.matcher(embeddings_A,embeddings_B)
            if return_embeddings:
                  return (match_prob,embeddings_A,embeddings_B)
            
            return match_prob
      

def evaluatembeddings_Baseline_performance(model_a, model_b, dataloader):
      device=get_device()
      model_a.eval()
      model_b.eval()
      
      y_true, y_pred = [], []
      
      start_time = time.perf_counter()
      total_pairs = 0
      
      with torch.no_grad():
            for batch_a, batch_b, labels in dataloader:
                  batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                  
                  # Module A: Forward pass
                  emb_a = model_a(batch_a)
                  emb_b = model_a(batch_b)
                  
                  # Module B: Pairwise match prediction
                  probs = model_b(emb_a, emb_b)
                  preds = (probs > 0.5).long().cpu().numpy()
                  
                  y_true.extend(labels.numpy())
                  y_pred.extend(preds)
                  total_pairs += len(labels)
                  
      total_time_ms = (time.perf_counter() - start_time) * 1000
      ms_per_pair = total_time_ms / total_pairs
      
      precision = precision_score(y_true, y_pred)
      recall = recall_score(y_true, y_pred)
      f1 = f1_score(y_true, y_pred)
      
      print("=== BASELINE MODULE A+B PERFORMANCE ===")
      print(f"Precision          : {precision * 100:.2f}%")
      print(f"Recall             : {recall * 100:.2f}%")
      print(f"F1-Score           : {f1 * 100:.2f}%")
      print(f"Inference Latency  : {ms_per_pair:.4f} ms/pair ({ms_per_pair * 1000:.2f} ms / 1k pairs)")
      
      return {"precision": precision, "recall": recall, "f1": f1, "ms_per_pair": ms_per_pair}

