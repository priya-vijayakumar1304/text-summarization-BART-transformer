# Text Summarization using BART Transformer

This project explores abstractive text summarization using the BART (Bidirectional and Auto-Regressive Transformer) model from Hugging Face Transformers.

The objective is to compare:

✅ Pre-trained BART model (without fine-tuning)

✅ Fine-tuned BART model on summarization dataset


⚠️ Note: If the notebook preview does not load on GitHub, download the file and open it locally in Jupyter Notebook
OR upload it directly to Google Colab to view and run it.

📥 Notebook file:
`Text_Summarizer_Using_BART_Transformer.ipynb`

Colab notebook link: https://colab.research.google.com/drive/1m_Q9BN-cSY718xt2G9AF9yeNM-qAmeqR

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

## Model & Dataset

### Pre-trained Model

**BART Large CNN**  
- Link: https://huggingface.co/facebook/bart-large-cnn  
- Optimized for abstractive summarization tasks  

### Dataset Used for Fine-Tuning

**DialogSum Dataset**  
- Link: https://huggingface.co/datasets/knkarthick/dialogsum  
- Contains human-annotated dialogue-summary pairs  
- Used for training and evaluation 

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

