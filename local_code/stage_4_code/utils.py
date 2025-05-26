import os
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import re

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()

def load_data_from_folders(base_dir):
    texts = []
    labels = []
    for label, sentiment in enumerate(["neg", "pos"]):
        folder = os.path.join(base_dir, sentiment)
        for filename in os.listdir(folder):
            if not filename.endswith(".txt"):
                continue  # Skip hidden or unexpected files like .DS_Store
            file_path = os.path.join(folder, filename)
            with open(file_path, encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

def tokenize_joke(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.strip().split()

def build_joke_vocab(joke_list):
    from torchtext.vocab import build_vocab_from_iterator
    tokenized = [tokenize_joke(j) for j in joke_list]
    vocab = build_vocab_from_iterator(tokenized, specials=["<pad>", "<sos>", "<eos>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def joke_collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # vocab['<pad>'] = 0
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=500):
        self.pad_idx = vocab['<pad>']
        self.texts = [torch.tensor(vocab(tokenize(text))[:max_len]) for text in texts]
        self.labels = torch.tensor(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) < self.max_len:
            padding = torch.full((self.max_len - len(text),), self.pad_idx, dtype=torch.long)
            text = torch.cat([text, padding])
        return text, self.labels[idx]

def build_vocab(texts):
    token_gen = (tokenize(text) for text in texts)
    vocab = build_vocab_from_iterator(token_gen, specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])
    return vocab

class JokeDataset(Dataset):
    def __init__(self, jokes, vocab, max_len=50):
        self.vocab = vocab
        self.jokes = jokes
        self.max_len = max_len
        self.encoded = []
        for joke in jokes:
            tokens = [vocab["<sos>"]] + vocab(tokenize_joke(joke)) + [vocab["<eos>"]]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            self.encoded.append(torch.tensor(tokens))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = self.encoded[idx][:-1]
        y = self.encoded[idx][1:]
        return x, y

def generate_joke(model, vocab, start_words="what did the", max_len=50):
    model.eval()
    words = tokenize_joke(start_words)
    tokens = [vocab["<sos>"]] + [vocab[w] for w in words]
    input_seq = torch.tensor(tokens).unsqueeze(0)

    generated = words[:]
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_seq, hidden)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            next_word = vocab.lookup_token(next_token)
            if next_word == "<eos>":
                break
            generated.append(next_word)
            input_seq = torch.tensor([[next_token]])

    return " ".join(generated)
