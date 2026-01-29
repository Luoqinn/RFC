# RFC: Reciprocal Fusion with Calibration for Reranking


* This repository implements the **RFC (Reciprocal Fusion with Calibration)** framework, an openset rerank for recommendation pipeline
* Paper: Reciprocal Collaborative and Semantic Fusion with Calibration via Large Language Models for Recommendation Reranking


## 📝 Closed-set Rerank ---> Open-set Rerank
We propose a novel framework termed **R**eciprocal **F**usion with **C**alibration (**RFC**) to transform existing LLM reranking from a closed-set permutation into an open-set enhancement process, as illustrated in the following Figure. Here are the two main differences:

| LLM-based Rerank Framework | Items in the original list and reranked list | How to use collaborative and semantic signals  |
|-----------------------------------------|------------|------------|
| Existing Closed-set permutation rerank | Exactly the same | sequential post-processing pipeline | 
| Proposed Open-set enhancement rerank | Maybe deffirent (**it can recall new candidates not present in the original list**) | deeply fused both collaborative and semantic signals via **neighbor-based reciprocal infusion mechanism** | 


## 🛠️ Requirements

* Python 3.8+
* PyTorch, Transformers, OpenAI, Pandas, NumPy

## 📂 Project Structure

* `main.py`: Entry point.
* `GASS.py`: Global Anchor based Semantic Scoring.
* `LLMUserEmbGen.py`: Generate user semantic embedding.
* `ReciprocalRescorer.py`: Reciprocal Rescoring Fusion.
* `Calibration.py`: Calibration for Reranking.
* `Evaluate.py`: Metrics calculation.

## 🚀 Pipeline Overview

1.  **Collaborative Scoring**: Retrieves initial candidate items and collaborative scores from base CFM.
2.  **Semantic Scoring (GASS)**: Uses LLM for pairwise comparisons (Item A vs. Item B).
3.  **Reciprocal Rescoring Fusion**: Fuses scores using reciprocal neighbor-based infusion mechanism (fusing Collaborative & Semantic signals).
4.  **Calibration**: Use LLM for Credibility Estimation and derive the final recommendations.



## ⚙️ Configuration (main.py)

`Model Configuration`
  
model_name = "../llama3"       # Path to your local LLM (e.g., Llama 3)

api_key = "sk-..."             # Your OpenAI API Key

`Algorithm Hyperparameters`
  
alpha = 0.5        # Weighting factor for fusion (0.0 - 1.0)

sim_users = 50     # Number of reciprocal neighbors to consider



## 🖥️ Usage
``` python
python main.py

