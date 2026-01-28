# Master Thesis Emanuele Salonico
emanuele.salonico@tum.de

## Title

**Reducing Workload for Title and Abstract Screening in Medical Systematic Reviews: A Comparison of ML and LLM Approaches**

## Abstract

Systematic reviews play a crucial role in evidence-based medicine, but the process of selecting relevant studies is highly time-consuming. In particular, **title and abstract screening (TIAB)** requires researchers to manually evaluate thousands of publications, often taking several months and significant resources in terms of time and money.

With the rapid growth of (biomedical) literature, there is an increasing need for automated solutions that can support researchers with this task.

This master thesis explores how **natural language processing (NLP)**, **machine learning (ML)**, and **large language models (LLMs)** can reduce the workload of TIAB screening.

## Approaches Compared

Three approaches are compared:

1. **Embedding-based classification with traditional ML models**
Text embeddings (e.g., from transformer models) are used as feature representations, which are then fed into classical machine learning classifiers such as Support Vector Machines, XGBOOST and Logistic Regression.

2. **Direct LLM classification with prompt engineering**
Large language models are used directly for include/exclude decisions via zero-shot and few-shot prompting strategies, also enhanced with semantic retreival to provide relevant context.

3. **Hybrid approach: LLM-based feature extraction with ML classification**
An LLM is used to extract structured binary features from titles and abstracts (via a data labeling pipeline), which are then used as input for a Random Forest classifier.

A labeled dataset from a medical domain is used as the basis for evaluation, focusing on **accuracy**, **efficiency**, and **practical applicability**. The results of this benchmark provide insights into the strengths and limitations of different methods and highlight how combining automation with human expertise can make systematic reviews faster, more reliable, and less resource-intensive.

The most important metrics for this study are **recall** and **specificity**.

## Research Questions

- **RQ1:** What is the current state of the art in existing literature for using ML and NLP techniques in TIAB screening?
- **RQ2:** Can embedding-based representations combined with traditional ML classifiers achieve reliable include/exclude decisions in systematic reviews?
- **RQ3:**  How do LLM-based classification approaches (zero-shot, few-shot, and semantic retrieval-enhanced) perform in terms of recall and specificity compared to traditional methods?
- **RQ4:**  Does a hybrid approach (using LLM-extracted features with a random forest classifier) offer a favorable trade-off between performance and computational cost?

## Getting Started

1. Create a virtual environment in the project root:

   ```bash
   python -m venv .venv
   ```

2. Install the project in editable mode:

   ```bash
   .venv/bin/pip install -e .
   ```

3. Create a `.env` file based on the template:

   ```bash
   cp .env-template .env
   ```

   Then fill in your API keys and configuration values.
