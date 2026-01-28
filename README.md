# RFC
Reciprocal Collaborative and Semantic Fusion with Calibration via Large Language Models for Recommendation Reranking


# RFC: Reciprocal Fusion & Calibration for Reranking

This repository implements the **RFC (Reciprocal Fusion & Calibration)** framework, a openset rerank for recommendation pipeline that integrates Collaborative Filtering (CF) signals with semantic signals of Large Language Models (LLMs).

## 🚀 Pipeline Overview

1.  **Collaborative Scoring**: Retrieves initial candidate items.
2.  **Semantic Scoring (GASS)**: Uses LLM for pairwise comparisons (Movie A vs. Movie B).
3.  **Reciprocal Rescoring Fusion**: Fuses scores using reciprocal neighbor-based approach (Collaborative & Semantic spaces).
4.  **Calibration**: Refines ranking by estimating prediction credibility (Yes/No probability).

## 📂 Project Structure

* `main.py`: Entry point.
* `GASS.py`: Generative Augmented Semantic Scoring (Pairwise).
* `Calibration.py`: Calibration for Reranking (Pointwise Yes/No).
* `ReciprocalRescorer.py`: Hybrid score fusion logic.
* `LLMUserEmbGen.py`: User profile embedding generation.
* `Evaluate.py`: Metrics calculation.

## 🛠️ Requirements

* Python 3.8+
* PyTorch, Transformers, OpenAI, Pandas, NumPy

## ⚙️ Configuration (main.py)

```python
model_name = "../llama3"       # Local LLM path
alpha = 0.5                    # Fusion weight
sim_users = 50                 # Neighbor count

## Run
python main.py

