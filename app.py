import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re

# Define the NextWordPredictor model
class NextWordPredictor(nn.Module):
    def _init_(self, vocab_size, emb_dim, hidden_size, context_size):
        super()._init_()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(context_size * emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)  # Embedding lookup
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer (vocab size)
        return x

# Load the trained model
def load_model(model_path, vocab_size, context_size, embedding_dim, hidden_dim):
    model = NextWordPredictor(vocab_size, embedding_dim, hidden_dim, context_size)
    # Set weights_only=True to avoid future warnings and improve security
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess user input
def preprocess_input(input_text):
    return re.sub('[^a-zA-Z0-9 \.]', '', input_text).lower()


# Generate next k words
def generate_next_k_words(model, input_text, word_to_idx, idx_to_word, k, context_size):
    input_text = preprocess_input(input_text)  # Ensure the text is preprocessed
    words = input_text.split()
    
    # Check if the input text is too short
    if len(words) < context_size:
        st.warning("Input text is too short. Please provide more context.")
        return ""

    # Prepare the input tensor
    input_words = words[-context_size:]  # Get the last context_size words
    
    # Map words to indices, using a default index (0) if not found
    input_indices = []
    for word in input_words:
        if word in word_to_idx:
            input_indices.append(word_to_idx[word])
        else:
            input_indices.append(0)  # Default index for unknown words

    input_indices = torch.tensor([input_indices], dtype=torch.long)  # Convert to tensor and add batch dimension

    generated_words = []
    for _ in range(k):
        with torch.no_grad():
            output = model(input_indices)
            next_word_idx = torch.argmax(output, dim=1).item()  # Get index of the predicted word
            next_word = idx_to_word.get(next_word_idx, "<UNK>")  # Use a default token for unknown indices
            generated_words.append(next_word)

            # Update input_indices with the new word for the next prediction
            input_indices = torch.cat((input_indices[:, 1:], torch.tensor([[next_word_idx]], dtype=torch.long)), dim=1)

    return ' '.join(generated_words)
# Main Streamlit app
def main():
    st.title("Next Word Prediction")

    # Load model parameters - adjust these to match your trained model
    vocab_size = 3473  # Use the actual vocab size from your training
    context_size = 5
    embedding_dim = 32  # Adjust to the embedding size used during training
    hidden_dim = 1024   # Keep the same as used during training
    model_path = 'next_word_predictor.pth'

    # Load the trained model
    model = load_model(model_path, vocab_size, context_size, embedding_dim, hidden_dim)

    # Create vocabulary mappings (update with your actual vocabulary)
    vocab = list(set("sample text dataset for vocabulary".split()))  # Replace with your actual vocabulary
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # User input
    input_text = st.text_area("Enter text:")
    k = st.number_input("Number of words to predict:", min_value=1, max_value=50, value=5)

    if st.button("Generate"):
        if input_text:
            generated_text = generate_next_k_words(model, input_text, word_to_idx, idx_to_word, k, context_size)
            st.success(f"Generated next {k} words: {generated_text}")
        else:
            st.warning("Please enter some text.")

if __name__ == "_main_":
    main()