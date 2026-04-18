# Beyond Prompting: Label Engineering for Support Ticket Classification

This repository contains the implementation, dataset analysis, and comparative study of three Natural Language Processing (NLP) strategies for classifying 20,000 customer support tickets. 

## Project Overview
The objective of this project is to evaluate the transition from a baseline Zero-Shot model to a production-ready Fine-Tuned model. The study highlights the significant impact of **Label Engineering** as a cost-effective alternative to immediate model training.

## Key Result
By refining the semantic descriptions of the labels (Label Engineering), the model's performance increased by **22%** (from 0.67 to 0.89 Macro F1-score) without any model retraining or additional computational overhead.

## Project Structure
* **data/**: Contains the `customer_support_tickets.csv` dataset.
* **notebooks/**: 
    * `1_Exploratory_Data_Analysis.ipynb`: Statistical analysis and visualization of class imbalance.
    * `2_Zero_Shot_Baseline.ipynb`: Implementation of Strategy I (Vanilla labels).
    * `3_Label_Engineering.ipynb`: Implementation of Strategy II (Elaborated labels).
    * `4_Fine_Tuning_BART.ipynb`: Implementation of Strategy III (Fine-tuning BART-base).
* **images/**: Confusion matrices and performance plots generated for the study.

## Strategies Evaluated

### Strategy I: Vanilla Zero-Shot
* **Model**: facebook/bart-large-mnli
* **Methodology**: Classification using raw, one-word category names.
* **Result**: 0.67 Macro F1-Score.

### Strategy II: Label Engineering
* **Model**: facebook/bart-large-mnli
* **Methodology**: Expanding labels into descriptive semantic anchors (e.g., "Fraud" transformed into "Fraud, unauthorized access, or security breach").
* **Result**: 0.89 Macro F1-Score.

### Strategy III: Domain Fine-Tuning
* **Model**: facebook/bart-base
* **Methodology**: Supervised fine-tuning for 3 epochs on 16,000 labeled examples.
* **Result**: 1.00 Accuracy/F1-Score.

## Installation and Requirements

To run the notebooks, ensure the following dependencies are installed:

```bash
pip install transformers datasets pandas matplotlib seaborn scikit-learn
```

## Related Article
A detailed analysis of the business logic, methodology, and ROI of these strategies can be found in the full article published on **Towards Data Science**.

## About the Author
**Ohad Jhirad** is a Senior Data Scientist and ML Engineer specializing in NLP and Anomaly Detection. He holds a BSC in Industrial Engineering.

Connect on [LinkedIn](https://www.linkedin.com/in/ohad-jhirad-8a842a173/)
