import torch
from tqdm import tqdm


class CalibrationForReranking:
    def __init__(self, model, tokenizer, pos_dict, interest, item_info, user_info, rating_dict, credibility_estimation_prompt):
        """
        Initialize the Calibration module with external data and models.

        Args:
            model: Pre-trained Language Model (e.g., Llama3).
            tokenizer: Tokenizer corresponding to the model.
            pos_dict: Dictionary containing positive interaction history for users.
            interest: User interests.
            movie_info: Dictionary containing metadata (Title, Genres) for movies.
            user_info: Dictionary containing metadata (Age, Gender, Occupation) for users.
            rating_dict: Dictionary mapping (UserID, MovieID) to specific ratings.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pos_dict = pos_dict
        self.interest = interest
        self.item_info = item_info
        self.user_info = user_info
        self.rating_dict = rating_dict

        # Define the pointwise prompt template
        self.credibility_estimation_prompt = credibility_estimation_prompt
        # Pre-calculate token IDs for "Yes" and "No"
        # add_special_tokens=False ensures we get the ID of the word itself without start tokens
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]

    def _get_probs(self, prompt):
        """
        Internal method to calculate probabilities for 'Yes' and 'No' tokens based on the prompt.
        """
        messages = [
            {"role": "system",
             "content": "You are a movie recommendation engine that can accurately predict whether a user would like to watch a given movie based on their profile and watching history."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Safe encoding with padding and truncation handling
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,  # Activate automatic padding
            truncation=True,  # Prevent input sequence overflow
            max_length=10000  # Adapt to context window (e.g., for 8B models)
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(inputs.input_ids, attention_mask=inputs.attention_mask).logits

        # Extract logits for the next token (last time step)
        next_token_logits = logits[0, -1, :]

        # Convert logits to probabilities via softmax
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Extract specific probabilities for Yes and No
        probs_yes = probs[self.yes_token_id].item()

        return probs_yes

    def _prompt_construction(self, user_id, item_id):
        """
        Constructs the string prompt using user profile, history, and target movie info.
        """
        str_uid = str(user_id)
        str_iid = str(item_id)

        # Construct User Profile String
        user_profile = "age: " + str(self.user_info[str_uid]["Age"]) + \
                       ", gender: " + self.user_info[str_uid]["Gender"] + \
                       ", occupation: " + self.user_info[str_uid]["Occupation"]

        # Construct History String (Last 300 items)
        history_items = []
        for his in self.pos_dict[str_uid][:300]:
            str_his = str(his)
            # Use .get() to handle potential missing keys safely
            meta = self.item_info.get(str_his, {"Title": "", "Genres": ""})
            rating = self.rating_dict.get((user_id, his), None)

            item_str = "《" + meta["Title"] + "》（genre: " + meta["Genres"] + "; rating:" + str(rating) + "）"
            history_items.append(item_str)

        history_str = ', '.join(history_items)

        # Construct Target Movie String
        target_meta = self.item_info.get(str_iid, {"Title": "", "Genres": ""})
        target_str = "《" + target_meta["Title"] + "》（" + target_meta["Genres"] + "）"

        return self.credibility_estimation_prompt % (user_profile, history_str, target_str)

    def run_calibration(self, fused_idx, fused_score):
        """
        Main execution loop: Iterates through recalled items, calculates probabilities, and saves results.

        Args:
            recall_idx_tensor: Tensor containing recalled movie IDs for each user.
            output_csv_path: Path to save the detailed CSV report.
            output_pt_path: Path to save the probability tensor.
        """
        credibility_score = []

        # Iterate over users in the recall tensor
        for user in tqdm(range(fused_idx.shape[0]), desc='Credibility Estimation via LLM'):
            # Iterate over recalled movies for the specific user
            for movie in fused_idx[user]:
                movie_item = movie.item()

                # Construct prompt and get probabilities
                prompt = self._prompt_construction(user, movie_item)
                probs_yes = self._get_probs(prompt)

                credibility_score.append(probs_yes)


        # Extract and reshape 'Yes' probabilities to Tensor
        # Assuming fixed recall size per user (e.g., 50) based on input tensor shape
        # Reranking Calibration
        credibility_score_tensor = torch.tensor(credibility_score, dtype=torch.float32).reshape(fused_idx.shape)
        calibrated_score = fused_score * credibility_score_tensor

        # Final Ranking
        calibrated_score_top20, positions_top20 = torch.topk(calibrated_score, 20, dim=1)  # positions_top20 是在 0..49 中的位置
        reranked_list = torch.gather(fused_idx, 1, positions_top20)  # 映射回原始项索引

        return reranked_list
