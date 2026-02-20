# Text Summarization using BART Transformer

This project explores abstractive text summarization using the BART (Bidirectional and Auto-Regressive Transformer) model from Hugging Face Transformers.

The objective is to compare:

✅ Pre-trained BART model (without fine-tuning)

✅ Fine-tuned BART model on summarization dataset


⚠️ Note: If the notebook preview does not load on GitHub, download the file and open it locally in Jupyter Notebook
OR upload it directly to Google Colab to view and run it.

📥 Notebook file:
Text_Summarizer_Using_BART_Transformer.ipynb

## Project Overview

BART is a sequence-to-sequence transformer model designed for natural language generation tasks like:

- Text summarization
- Question answering
- Text generation
- Translation

In this project, we:

- Load a pre-trained BART model
- Generate summaries without fine-tuning
- Fine-tune the model on a summarization dataset
- Compare output quality

## Model Used
- `facebook/bart-large-cnn` (Pre-trained summarization model)

## Tech Stack

- Python
- Hugging Face Transformers
- PyTorch
- Google Colab

## Results

The fine-tuned BART model generates:

- More context-aware summaries
- Better domain adaptation (if trained on specific dataset)
- Improved coherence compared to base model

## Key Learnings

- Understanding encoder-decoder transformer architecture
- Working with Hugging Face Trainer API
- Fine-tuning large language models
- Text preprocessing and tokenization

## Future Improvements

- Add ROUGE score evaluation
- Deploy as API using FastAPI
- Add Streamlit demo app
- Experiment with other models (T5, PEGASUS)

