import pandas as pd
import torch
from model import JokeGeneratorRNN
from utils import tokenize_joke, build_joke_vocab
import os

from Run_Generation import generate_joke

# Load jokes to rebuild vocab
csv_path = os.path.expanduser("/mnt/d/ECS暑课/189G/stage_4_data/text_generation/jokes.csv")
df = pd.read_csv(csv_path)
jokes = df["Joke"].dropna().tolist()
vocab = build_joke_vocab(jokes)

# Recreate model
model = JokeGeneratorRNN(len(vocab), embedding_dim=128, hidden_dim=256)
model.load_state_dict(torch.load("joke_generator.pt", map_location="cpu"))
model.eval()

# Test with custom prompts
prompts = ["how many programmers", "my dog walked", "I love deeplearning"]
for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print("Generated:", generate_joke(model, vocab, start_words=prompt))

