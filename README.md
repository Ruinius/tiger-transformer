# Tiger Transformer

A fine-tuned transformer model designed to standardize financial statement line items (Balance Sheet and Income Statement).

## Overview

This project supports a classification step for a financial analysis AI agent. The primary goals were:
- **Model Input**: Uses a context window of 2 lines before and after the target line to improve prediction accuracy.
- **Data Strategy**: **All** available labeled data is used for training to maximize model performance. There is no traditional train/val split; instead, validation is done manually by labeling and testing against new, unseen data batches.
- **Post-Processing**: Deterministic attributes are handled separately from the probabilistic model predictions to ensure high accuracy for rule-based flags.

Since commercial LLMs gave inconsistent responses and no off-the-shelf solution matched these requirements, a custom model was fine-tuned on a curated dataset.

## Results

Accuracy for new sets of 3 companies is 90-95% with even the mistakes being acceptable/useable guesses.
Accuracy falls for companies in new industries or regions with unseen line items, but the mistakes are still reasonable.
The model has excellent accuracy on the important subtotal and total fields, which enables downstream validation in the full financial analysis AI agent.

**Note**: This repository is **fork-only**.

## Dataset

The repository includes a painstakingly labeled dataset of:
- **Balance Sheets**
- **Income Statements**

These raw labels from commercial LLMs were cleaned manually to generate clean labels for the transformer.

## Workflow & Implementation

The project follows a structured implementation plan, executed through a series of Jupyter notebooks:

1.  **Data Cleaning & Exploration** (`notebooks/01_data_cleaning.ipynb`):
    - Visualizes, searches, and support manual label cleaning
2.  **Training Data Generation** (`notebooks/02_generate_training_data.ipynb`):
    - Creates context windows for the model (including surrounding line items).
    - Format: `[PREV_2] [PREV_1] [SECTION] [RAW_NAME] [NEXT_1] [NEXT_2]`.
3.  **Model Training** (`notebooks/03_train_transformer.ipynb`):
    - Fine-tunes a `FinBERT` model.
    - Uses all available labeled data for training (iterative approach). Validation is performed manually by labeling new batches of data.
4.  **Evaluation** (`notebooks/04_evaluate.ipynb`):
    - Compares model predictions against ground truth.
5.  **Additional Mapping** (`notebooks/05_additional_mapping.ipynb`):
    - Helper for creating deterministic attributes: `is_calculated`, `is_operating`, and `is_expense`
    - Actual mapping is in `notebooks/bs_calculated_operating_mapping.csv` and `notebes/is_calculated_operating_expense_mapping.csv`

## Project Structure

```
tiger-transformer/
├── balance_sheet_raw_label/          # Original labeled data (Balance Sheets)
├── income_statement_raw_label/       # Original labeled data (Income Statements)
├── balance_sheet_clean_label/        # Processed data output from Notebook 1
├── income_statement_clean_label/     # Processed data output from Notebook 1
├── data/
│   └── training_data.jsonl           # Generated training examples
├── models/
│   └── checkpoint-best/              # Fine-tuned model artifacts
└── notebooks/
    ├── 01_data_cleaning.ipynb
    ├── 02_generate_training_data.ipynb
    ├── 03_train_transformer.ipynb
    ├── 04_evaluate.ipynb
    └── 05_additional_mapping.ipynb
```

## Mapping History & Schema Evolution

The files `notebooks/bs_mapping_history.csv` and `notebooks/is_mapping_history.csv` are critical for maintaining schema consistency. They serve as specific lookup tables to:
-   **Track Schema Changes**: Record how standardized names have evolved over time.
-   **Consolidate Labels**: Map older, potentially more granular or divergent labels to the current unified schema (e.g., mapping `accounts_payable_accrued_liabilities` → `accounts_payable`).
-   **Enforce Consistency**: Ensure that all data, regardless of when it was labeled, aligns with the latest `standardized_name` definitions.

## Acknowledgments & Licensing
This project is a fine-tuned version of the FinBERT-Pretrain model developed by Yang et al. (HKUST).

The base model and this fine-tuned version are licensed under the Apache License 2.0.