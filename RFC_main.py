import json
import torch
import numpy as np
import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from GASS import GASS
from LLMUserEmbGen import LLMUserEmbeddingGenerator
from ReciprocalRescorer import ReciprocalRescorer
from Calibration import CalibrationForReranking
from Evaluate import evaluate_k_torch

interest_num = 6
topk = 20
topl = 50


def main():
    ################### Configurations ################################
    alpha = 0.5  # Weighting factor
    sim_users = 50  # Number of similar users to consider

    model_name = "../llama3"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    pos_dict_path = 'data/pos_dict.json'
    interest_path = 'data/interest.json'
    item_info_path = 'data/item_info.json'
    user_info_path = 'data/user_info.json'
    train_csv_path = 'data/train.csv'
    rating_mat_path = 'data/rating_mat.pt'
    col_user_emb_path = 'data/llm_user_emb.pt'
    label_mat_path = 'data/label_mat.npy'
    api_key = "your_openai_api_key_here"  # Replace with your actual OpenAI API key
    client = OpenAI(api_key=api_key)

    print("Loading external data...")
    with open(pos_dict_path, 'r', encoding='utf-8') as file:
        pos_dict = json.load(file)
    with open(interest_path, 'r', encoding='utf-8') as file:
        interest = json.load(file)
    with open(item_info_path, 'r', encoding='utf-8') as file:
        item_info = json.load(file)
    with open(user_info_path, 'r', encoding='utf-8') as file:
        user_info = json.load(file)
    label = torch.FloatTensor(np.load(label_mat_path)).to(device)



    ########################## Prompt Design ##########################
    pairwise_comparison_prompt = """Considering user profile and preferences, identify which of the following two movies is more preferred by user, and ONLY output A or B.
### User profile: %s.
### User preferences: %s.
### Movie A: %s.
### Movie B: %s.
"""
    credibility_estimation_prompt = """Based on the user profile and movie watching history (including movie titles, genres, and user ratings from 1 to 5), predict whether the user will like to watch the given movie. Respond ONLY with "Yes" or "No", without any additional explanation or text.
### User profile: %s.
### Watching history: %s.
### Given movie: %s.
"""

    ########################### Load Model and Tokenizer ###########################
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure padding
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


    ##############################################################################
    ################################# START RFC ##################################
    ##############################################################################

    ################# 3.2.1 Collaborative Scoring #########################
    cfm_rating_mat = torch.tensor(torch.load(rating_mat_path, weights_only=False)).to(device)
    cfm_score, cfm_recidx = torch.topk(cfm_rating_mat, topk, dim=1)


    ################# 3.2.2 Semantic Scoring #########################
    gass = GASS(model=model,
                tokenizer=tokenizer,
                interest=interest,
                item_info=item_info,
                user_info=user_info,
                cfm_topk_list=cfm_recidx,
                interest_num=interest_num,
                pairwise_comparison_prompt=pairwise_comparison_prompt)
    sem_score = gass.run_gass()



    ################## 3.3 Reciprocal Rescoring Fusion #########################
    # get llm user embeddings
    llm_emb_gen = LLMUserEmbeddingGenerator(
        client=client,
        pos_dict=pos_dict,
        movie_info=item_info,
        user_info=user_info
    )
    llm_user_emb = llm_emb_gen.generate_embeddings()
    # get collaborative user embeddings
    col_user_emb = torch.tensor(torch.load(col_user_emb_path, weights_only=False)).to(device)

    # start reciprocal rescoring process
    reciprocal_rescorer = ReciprocalRescorer(device=device)

    fused_score, fused_idx = reciprocal_rescorer.run_reciprocal_rescoring(
        col_emb=col_user_emb,
        cfm_rating_mat=cfm_rating_mat,
        sem_score=sem_score,
        sem_emb=llm_user_emb, 
        alpha=alpha,
        sim_users=sim_users,
        topk=topk,
        topl=topl
    )


    ###################### 3.4 Calibration for Reranking #########################
    # Read CSV and create rating dictionary
    df = pd.read_csv(train_csv_path)
    rating_dict = {(row['UserID'], row['MovieID']): row['Rating'] for _, row in df.iterrows()}

    calibrator = CalibrationForReranking(
        model=model,
        tokenizer=tokenizer,
        pos_dict=pos_dict,
        interest=interest,
        item_info=item_info,
        user_info=user_info,
        rating_dict=rating_dict,
        credibility_estimation_prompt=credibility_estimation_prompt
    )
    reranked_list = calibrator.run_calibration(
        fused_idx=fused_idx,
        fused_score=fused_score
    )


    ######################## Evaluation #########################
    for k in [1, 10, 20]:
         recall, ndcg, map = evaluate_k_torch(
            reranked_list[:, :k], label, k
         )
         print(f"Recall@{k}: {recall:.4f}\tNDCG@{k}: {ndcg:.4f}\tMAP@{k}: {map:.4f}")


if __name__ == "__main__":

    main()
