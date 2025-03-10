import os
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration

def download_models():
    # Create model directories
    os.makedirs("models/distilbert_model", exist_ok=True)
    os.makedirs("models/t5_model", exist_ok=True)

    # Download and save DistilBERT
    print("Downloading DistilBERT model...")
    BertTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained("models/distilbert_model")
    BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).save_pretrained("models/distilbert_model")

    # Download and save T5-small
    print("Downloading T5 model...")
    T5Tokenizer.from_pretrained("t5-small").save_pretrained("models/t5_model")
    T5ForConditionalGeneration.from_pretrained("t5-small").save_pretrained("models/t5_model")

if __name__ == "__main__":
    try:
        download_models()
        print("Models downloaded successfully!")
    except Exception as e:
        print(f"Error: {str(e)}\n\nRequired dependencies: pip install sentencepiece transformers torch")