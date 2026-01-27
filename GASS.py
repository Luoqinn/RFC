import torch
from tqdm import tqdm


class GASS:
    def __init__(self, model, tokenizer, interest, item_info, user_info, cfm_topk_list, interest_num, pairwise_comparison_prompt):
        """
        Initialize the Calibration module with external data and models.

        Args:
            model: Pre-trained Language Model (e.g., Llama3).
            tokenizer: Tokenizer corresponding to the model.
            interest: User interests.
            item_info: Dictionary containing metadata (Title, Genres) for items.
            user_info: Dictionary containing metadata (Age, Gender, Occupation) for users.
            cfm_topk_list: List of candidate movie IDs for each user.
            interest_num: Number of interests to consider for each user.
            pairwise_comparison_prompt: Template for pairwise comparison prompts.
        """
        self.model = model
        self.tokenizer = tokenizer

        self.interest = interest
        self.item_info = item_info
        self.user_info = user_info
        self.cfm_topk_list = cfm_topk_list
        self.interest_num = interest_num
        self.global_anchor = self.get_global_anchor(cfm_topk_list)


        # Define the pointwise prompt template
        self.pairwise_comparison_prompt = pairwise_comparison_prompt

        # get A/B's token ids
        self.A_token_id = self.tokenizer.encode("A", add_special_tokens=False)
        self.B_token_id = self.tokenizer.encode("B", add_special_tokens=False)

    def get_global_anchor(self, cfm_reclist):
        '''
        :param cfm_reclist:
        :return: global_anchor_idx
        '''
        values, counts = torch.unique(cfm_reclist, return_counts=True)
        most_frequent_idx = torch.argmax(counts)
        global_anchor_idx = values[most_frequent_idx]

        return global_anchor_idx.item()

    def _get_probs(self, prompt):
        """
        Internal method to calculate probabilities for 'A' and 'B' tokens based on the prompt.
        """
        messages = [{"role": "user", "content": prompt}]

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

        next_token_logits = logits[0, -1, :]

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        probs_a = probs[self.A_token_id].item()
        probs_b = probs[self.B_token_id].item()

        return probs_a, probs_b


    def _prompt_construction(self, user_id, item_a, item_b):
        """
        Constructs the string prompt using user profile, history, and target movie info.
        """
        str_uid, str_iid_a, str_iid_b = str(user_id), str(item_a), str(item_b)

        return self.pairwise_comparison_prompt % (
            "age: " + str(self.user_info[str_uid]["Age"]) + ", gender: " + self.user_info[str_uid]["Gender"] + ", occupation: " +
            self.user_info[str_uid]["Occupation"],
            ' > '.join(self.interest[str_uid][:self.interest_num]),
            f"《{self.item_info[str_iid_a]['Title'][:-7]}》（{self.item_info[str(str_iid_a)]['Genres']}）",
            f"《{self.item_info[str_iid_b]['Title'][:-7]}》（{self.item_info[str(str_iid_b)]['Genres']}）",
        )


    def run_gass(self):

        semantic_scores = []

        for user in tqdm(range(self.cfm_topk_list.shape[0]), desc='Get semantic scores via GASS'):
            for item in self.cfm_topk_list[user]:
                candidate_item = item.item()

                # Construct prompt and get probabilities
                # global anchor first, [anchor, candidate]
                prompt1 = self._prompt_construction(user, self.global_anchor, candidate_item)
                probs_a, probs_b = self._get_probs(prompt1)

                # candidate first, [candidate, anchor]
                prompt2 = self._prompt_construction(user, candidate_item,  self.global_anchor)
                probs_a2, probs_b2 = self._get_probs(prompt2)

                item_score = 0.5 * ((probs_b - probs_a) + (probs_a2 - probs_b2))
                semantic_scores.append(item_score)

        semantic_scores = torch.tensor(semantic_scores, dtype=torch.float32).reshape(self.cfm_topk_list.shape)

        return semantic_scores


