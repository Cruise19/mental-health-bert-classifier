"""Run the fine-tuned model on new text.

Same loading and inference code as the "Loading The Model For Testing Again"
section of notebooks/mental_health_bert.ipynb, wrapped as a command-line script.

Usage:
    python src/predict.py --model-dir path/to/saved_mental_status_bert "some text"
    python src/predict.py --model-dir path/to/saved_mental_status_bert --probs "some text"
"""
import argparse
import pickle
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def clean_statement(text):
    # Must match the cleaning used at training time
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # strip URLs
    text = re.sub(r'\s+', ' ', text)                 # collapse whitespace
    return text.strip()


def load(model_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    model.eval()
    return model, tokenizer, label_encoder, device


def predict_probs(text, model, tokenizer, label_encoder, device):
    inputs = tokenizer(clean_statement(text), return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ('input_ids', 'attention_mask')}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=1)[0]
    return sorted(zip(label_encoder.classes_, probs.tolist()), key=lambda x: -x[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('text', help='Statement to classify')
    parser.add_argument('--model-dir', required=True, help='Folder with model.safetensors, tokenizer files and label_encoder.pkl')
    parser.add_argument('--probs', action='store_true', help='Print the probability of every class')
    args = parser.parse_args()

    model, tokenizer, label_encoder, device = load(args.model_dir)
    ranked = predict_probs(args.text, model, tokenizer, label_encoder, device)

    print(f"Predicted: {ranked[0][0]}")
    if args.probs:
        for name, p in ranked:
            print(f"  {name:22s} {p:.3f}")


if __name__ == '__main__':
    main()
