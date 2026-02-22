# 📝 Dialogue Summarization using BART (Supervised Fine-Tuning)

This project implements **abstractive dialogue summarization** using a pre-trained BART transformer model and applies **Supervised Fine-Tuning (SFT)** on the DialogSum dataset.

The objective is to compare baseline summarization performance with a task-adapted fine-tuned model.

📓 Notebook: `Text_Summarizer_Using_BART_Transformer.ipynb`

> ⚠️ If the notebook preview does not load on GitHub, download the file and open it locally in Jupyter Notebook, or use the Google Colab link below.

**Colab Link:**  
https://colab.research.google.com/drive/1m_Q9BN-cSY718xt2G9AF9yeNM-qAmeqR

---

## 🚀 Project Objective

- Evaluate baseline performance of a pre-trained BART summarization model  
- Apply supervised fine-tuning (SFT) on dialogue–summary pairs  
- Measure performance using ROUGE metrics  
- Analyze qualitative differences between baseline and fine-tuned outputs  

---

## 🤖 Model Used

**BART (Bidirectional and Auto-Regressive Transformer)**  
Model: `facebook/bart-large-cnn`  
https://huggingface.co/facebook/bart-large-cnn  

BART is a sequence-to-sequence transformer combining:
- A bidirectional encoder (BERT-style)
- An autoregressive decoder (GPT-style)

It is well-suited for abstractive summarization tasks.

---

## 📊 Dataset

**DialogSum Dataset**  
https://huggingface.co/datasets/knkarthick/dialogsum  

- Contains human-annotated dialogue–summary pairs  
- Designed specifically for dialogue summarization  
- Used for training and evaluation  

---

## 🔧 Fine-Tuning Approach

This project uses **Supervised Fine-Tuning (SFT)**.

- Input: Dialogue text  
- Target: Human-written summary  
- Loss Function: Cross-entropy loss  
- Training Method: End-to-end parameter updates using Hugging Face Trainer  

All model parameters are updated to adapt the pre-trained BART model to dialogue summarization.

---

## 📈 Evaluation

Model performance is evaluated using **ROUGE metrics**:

- **ROUGE-1** → Word overlap  
- **ROUGE-2** → Bigram overlap  
- **ROUGE-L** → Longest common subsequence similarity  

### ⚠ Resource Constraint Note

During evaluation, full test-set ROUGE computation with `bart-large` exceeded Google Colab memory limits due to:

- Large model size (~400M parameters)
- Sequence generation memory overhead
- Dataset loading and prediction storage in RAM

To ensure stable experimentation:

- Evaluation was performed on a representative subset of the test split
- Memory-aware generation settings were used
- Model checkpoints were saved separately for reproducibility

This reflects practical constraints when working with large transformer models in limited-resource environments.

---

## 📌 Key Results

The fine-tuned model demonstrates:

- Improved contextual understanding of dialogue flow  
- Better abstraction compared to the baseline model  
- More coherent and structured summaries  

---

## 🛠 Tech Stack

- Python  
- Hugging Face Transformers  
- PyTorch  
- Hugging Face Datasets  
- Google Colab  

---

## 📚 Key Learnings

- Sequence-to-sequence transformer architecture  
- Supervised fine-tuning workflow  
- ROUGE-based evaluation for summarization  
- Memory-aware training and evaluation  
- Practical constraints of large-model experimentation in limited-resource environments  

---

## 🔮 Future Improvements

- Deploy as a FastAPI service  
- Add inference latency benchmarking  
- Perform larger-scale evaluation  
- Compare BART-base vs BART-large  
- Experiment with parameter-efficient fine-tuning (LoRA / PEFT)  

---

## 📜 License

This project is for educational and experimentation purposes.

