import torch
from torch.utils.data import Dataset

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Get unique labels
        self.label_list = sorted(list(set(label for sublist in labels for label in sublist)))
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(self.label_list)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.labels[idx]

        encodings = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        label_ids = [-100] * len(encodings['input_ids'][0])
        offset_mapping = encodings.pop('offset_mapping')[0]

        word_ids = encodings.word_ids(batch_index=0)
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            label_ids[i] = self.label2id[labels[word_idx]]
            previous_word_idx = word_idx

        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(label_ids)

        return item


def convert_goldparse(data):
    """
    Converts a labeled dataset from JSON format into token-label pairs.
    Expected format:
    [
        {"tokens": ["My", "name", "is", "John"], "labels": ["O", "O", "O", "B-NAME"]},
        ...
    ]
    """
    texts = []
    labels = []

    for entry in data:
        tokens = entry["tokens"]
        ents = entry["labels"]
        texts.append(tokens)
        labels.append(ents)

    return texts, labels
