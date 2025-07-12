# ConvBERT GLUE Benchmark: Paper Reproduction Study

This repository provides a comprehensive reproduction of the ConvBERT model performance as presented in the original paper, specifically focusing on fine-tuning and evaluating ConvBERT on the GLUE benchmark, a suite of natural language understanding tasks.

## 📌 Project Overview

The goal of this notebook is to validate and reproduce the findings of the original ConvBERT paper by fine-tuning and evaluating the ConvBERT model across multiple GLUE tasks. Advanced optimization techniques like gradient accumulation and mixed precision training have been utilized to manage computational complexity efficiently.

## 📚 GLUE Tasks Covered

* **SST-2**: Sentiment analysis
* **CoLA**: Linguistic acceptability
* **MRPC**: Paraphrase detection
* **STS-B**: Semantic textual similarity
* **QQP**: Question paraphrase detection
* **MNLI**: Multi-genre natural language inference
* **QNLI**: Question-answer inference
* **RTE**: Recognizing textual entailment
* **WNLI**: Winograd natural language inference

## ⚙️ Implementation Details

* **Model:** [ConvBERT (YituTech/conv-bert-base)](https://huggingface.co/YituTech/conv-bert-base)
* **Framework:** PyTorch and Hugging Face Transformers
* **Optimization Techniques:**

  * Gradient Accumulation
  * Mixed Precision Training (AMP)

## 🛠️ Setup

Install dependencies:

```bash
pip install torch transformers datasets evaluate
```

## 🚀 Running the Notebook

You can run this notebook in:

* Google Colab (Recommended)
* Local Jupyter Lab/Notebook

## 💻 Notebook Structure

* Data loading and preprocessing with tokenization
* Custom padding function to handle variable-length sequences
* Training with gradient accumulation and AMP
* Evaluation of each GLUE task with appropriate metrics (Accuracy, Pearson Correlation, MSE)
* Final computation of the overall GLUE score

## 📈 Results

The notebook computes and outputs:

* Task-specific performance metrics
* Aggregated GLUE score

## 📜 Citation

If referencing this reproduction, please cite the original ConvBERT paper:

```bibtex
@article{jiang2020convbert,
  title={ConvBERT: Improving BERT with Span-based Dynamic Convolution},
  author={Jiang, Heng and He, Wei and Chen, Weizhu and Liu, Xinyan},
  journal={arXiv preprint arXiv:2008.02496},
  year={2020}
}
```

## 🙌 Acknowledgements

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [GLUE Benchmark](https://gluebenchmark.com/)
* [YituTech ConvBERT](https://huggingface.co/YituTech/conv-bert-base)
