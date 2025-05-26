import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from model import JokeGeneratorRNN
from utils import tokenize_joke, build_joke_vocab, JokeDataset, joke_collate_fn
from train import train_generator
import matplotlib.pyplot as plt



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


if __name__ == "__main__":
    print("=== RNN Text Generation from Jokes Dataset ===")

    # Load jokes dataset
    print("[INFO] Loading joke dataset...")
    csv_path = os.path.expanduser("/mnt/d/ECS暑课/189G/stage_4_data/text_generation/jokes.csv")
    df = pd.read_csv(csv_path)
    jokes = df["Joke"].dropna().tolist()

    print(f"[INFO] Loaded {len(jokes)} jokes.")

    # Build vocab
    print("[INFO] Building vocabulary...")
    vocab = build_joke_vocab(jokes)
    print(f"[INFO] Vocab size: {len(vocab)}")

    # Prepare dataset and dataloader
    print("[INFO] Preparing dataset and dataloader...")
    joke_dataset = JokeDataset(jokes, vocab, max_len=50)
    joke_loader = DataLoader(joke_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=joke_collate_fn)

    # Define model
    print("[INFO] Initializing JokeGeneratorRNN model...")
    model = JokeGeneratorRNN(len(vocab), embedding_dim=128, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # Train model
    print("[INFO] Starting training...")
    train_generator(model, joke_loader, optimizer, criterion, n_epochs=20)

    # Generate a joke
    print("[INFO] Generating a joke with the prompt 'what did the'...")
    joke = generate_joke(model, vocab, start_words="what did the")
    print("[RESULT] Generated Joke:")
    print(joke)

    # Optionally save model
    torch.save(model.state_dict(), "joke_generator.pt")
