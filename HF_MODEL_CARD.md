---
language:
- en
license: apache-2.0
tags:
- financial-analysis
- transformer
- classification
- finbert
- financial-statements
base_model: yiyanghkust/finbert-pretrain
model-index:
- name: tiger-transformer
  results: []
---

# Tiger Transformer (Standardizing Financial Statements)

This model is a fine-tuned version of [yiyanghkust/finbert-pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain) designed to standardize financial statement line items from Balance Sheets and Income Statements into a unified schema.

**Full Source Code & Training Data**: [GitHub - Ruinius/tiger-transformer](https://github.com/Ruinius/tiger-transformer)

## Model Description

The **Tiger Transformer** serves as a specialized classification engine for financial analysis AI agents. It addresses the inconsistency found in broad-purpose LLMs when mapping diverse, raw line items (e.g., "Cash & Equivalents", "Cash and due from banks") to standardized accounting categories.

### Key Features:
- **Context-Aware Classification**: Unlike simple keyword matching, this model uses a context window of 2 lines before and 2 lines after the target line to refine predictions.
- **Architecture**: Fine-tuned `BertForSequenceClassification` using the FinBERT base.
- **Quantization Support**: A quantized version (`pytorch_model_quantized.pt`) is available for low-latency CPU inference.

## Intended Uses & Limitations

### Intended Use
Standardizing raw line items extracted from 10-K, 10-Q, and other financial reports into a consistent format for downstream financial modeling (DCF, ROIC analysis, etc.).

### Training Data Strategy
The model was trained on a painstakingly curated dataset of manually cleaned financial statement labels. To maximize performance on a niche dataset, the model utilizes all available high-quality labels for training, with validation performed iteratively against new unseen batches.

### Performance
- **Accuracy**: 90-95% on modern financial reports.
- **Robustness**: High accuracy on critical fields (Subtotals and Totals), which are essential for structural validation.
- **Limitations**: Accuracy may decrease for companies in highly specialized industries or niche regions with non-standard terminology not present in the training set.

## Training Procedure

### Input Format
The model expects input strings formatted with surrounding context:
`[PREV_2] [PREV_1] [SECTION] [RAW_NAME] [NEXT_1] [NEXT_2]`

*   `[SECTION]`: Balance Sheet or Income Statement.
*   `[RAW_NAME]`: The line item name to be classified.
*   `[PREV/NEXT]`: Surrounding line items providing structural context.

### Hyperparameters
- **Base Model**: FinBERT
- **Quantization**: Dynamic quantization (int8) applied to Linear layers for optimized CPU performance.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Ruinius/tiger-transformer")
model = AutoModelForSequenceClassification.from_pretrained("Ruinius/tiger-transformer")

# Example input with context
text = "Cash and Short-term Investments [SEP] Cash and Equivalents [SEP] Balance Sheet [SEP] Accounts Receivable [SEP] Inventory [SEP] Prepaid Expenses"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()

# Map ID back to label using model.config.id2label
```

## Acknowledgments & Licensing
This project is a fine-tuned version of the FinBERT-Pretrain model developed by Yang et al. (HKUST).
Licensed under the **Apache License 2.0**. Same as the base FinBERT model.
