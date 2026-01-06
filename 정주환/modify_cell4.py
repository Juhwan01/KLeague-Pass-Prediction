import json

# 노트북 읽기
with open(r'i:\KLeague-Pass-Prediction\정주환\[V12]N-gram(3,5),+teamID copy.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4 (dataset) 가져오기
cell4 = nb['cells'][4]
source_lines = cell4['source']

# __getitem__ 메서드를 찾아서 수정
source = ''.join(source_lines)

# __getitem__ 수정
old_getitem = """    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.episodes[idx])
        ng3 = torch.LongTensor(self.ngram3_indices[idx])
        ng5 = torch.LongTensor(self.ngram5_indices[idx])
        team_id = self.team_ids[idx]

        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
            ng3 = ng3[-MAX_SEQ_LEN:]
            ng5 = ng5[-MAX_SEQ_LEN:]

        if self.mode == 'train':
            return seq, ng3, ng5, team_id, torch.FloatTensor(self.targets[idx])
        return seq, ng3, ng5, team_id, self.episode_ids[idx]"""

new_getitem = """    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.episodes[idx])
        ng3 = torch.LongTensor(self.ngram3_indices[idx])
        ng5 = torch.LongTensor(self.ngram5_indices[idx])
        team_id = self.team_ids[idx]
        episode_id = self.episode_ids[idx]

        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
            ng3 = ng3[-MAX_SEQ_LEN:]
            ng5 = ng5[-MAX_SEQ_LEN:]

        if self.mode == 'train' or self.mode == 'val':
            return seq, ng3, ng5, team_id, torch.FloatTensor(self.targets[idx]), episode_id
        return seq, ng3, ng5, team_id, episode_id  # test mode"""

source = source.replace(old_getitem, new_getitem)

# collate_fn 수정
old_collate = """def collate_fn(batch):
    \"\"\"
    Collate function for N-gram + Team ID
    \"\"\"
    seqs = [b[0] for b in batch]
    ng3s = [b[1] for b in batch]
    ng5s = [b[2] for b in batch]
    team_ids = [b[3] for b in batch]

    lens = torch.LongTensor([len(s) for s in seqs])

    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    ng3_padded = pad_sequence(ng3s, batch_first=True, padding_value=0)
    ng5_padded = pad_sequence(ng5s, batch_first=True, padding_value=0)
    team_ids_tensor = torch.LongTensor(team_ids)

    mask = (seqs.sum(dim=-1) != 0).float()

    if len(batch[0]) == 5:  # Train: (seq, ng3, ng5, team_id, target)
        targets = torch.stack([b[4] for b in batch])
        return (seqs, ng3_padded, ng5_padded, team_ids_tensor, targets, mask, lens)
    else:  # Test: (seq, ng3, ng5, team_id, episode_id)
        episode_ids = [b[4] for b in batch]
        return (seqs, ng3_padded, ng5_padded, team_ids_tensor, mask, lens, episode_ids)"""

new_collate = """def collate_fn(batch):
    \"\"\"
    Collate function for N-gram + Team ID
    \"\"\"
    seqs = [b[0] for b in batch]
    ng3s = [b[1] for b in batch]
    ng5s = [b[2] for b in batch]
    team_ids = [b[3] for b in batch]

    lens = torch.LongTensor([len(s) for s in seqs])

    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    ng3_padded = pad_sequence(ng3s, batch_first=True, padding_value=0)
    ng5_padded = pad_sequence(ng5s, batch_first=True, padding_value=0)
    team_ids_tensor = torch.LongTensor(team_ids)

    mask = (seqs.sum(dim=-1) != 0).float()

    if len(batch[0]) == 6:  # Train/Val: (seq, ng3, ng5, team_id, target, episode_id)
        targets = torch.stack([b[4] for b in batch])
        episode_ids = [b[5] for b in batch]
        return (seqs, ng3_padded, ng5_padded, team_ids_tensor, targets, mask, lens, episode_ids)
    else:  # Test: (seq, ng3, ng5, team_id, episode_id) - 5개
        episode_ids = [b[4] for b in batch]
        return (seqs, ng3_padded, ng5_padded, team_ids_tensor, mask, lens, episode_ids)"""

source = source.replace(old_collate, new_collate)

# 수정된 내용 출력
print(source)
