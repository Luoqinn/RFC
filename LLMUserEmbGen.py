import torch
from tqdm import tqdm


class LLMUserEmbeddingGenerator:
    def __init__(self, client, pos_dict, movie_info, user_info):
        """
        Initialize the UserEmbeddingGenerator with external data and API client.

        Args:
            client: Initialized OpenAI client object.
            pos_dict: Dictionary containing positive interaction history for users.
            movie_info: Dictionary containing metadata (Title, Genres) for movies.
            user_info: Dictionary containing metadata (Age, Gender, Occupation) for users.
        """
        self.client = client
        self.pos_dict = pos_dict
        self.movie_info = movie_info
        self.user_info = user_info

    def _prompt_construction(self, user_id):
        """
        Constructs a structured string containing user profile and watching history.
        """
        str_uid = str(user_id)

        # specific handling for user_info keys
        if str_uid not in self.user_info:
            return None

        # Construct User Profile String
        profile = "age: " + str(self.user_info[str_uid]["Age"]) + \
                  ", gender: " + self.user_info[str_uid]["Gender"] + \
                  ", occupation: " + self.user_info[str_uid]["Occupation"]

        # Construct History String (Last 300 items)
        # Using list comprehension to build movie details string
        history_list = []
        # Ensure user exists in pos_dict, default to empty list if not
        user_history = self.pos_dict.get(str_uid, [])[:300]

        for his in user_history:
            str_his = str(his)
            if str_his in self.movie_info:
                item_str = "《" + self.movie_info[str_his]["Title"] + \
                           "》（genre: " + self.movie_info[str_his]["Genres"] + "）"
                history_list.append(item_str)

        history = ', '.join(history_list)

        # Return formatted dictionary-like string
        return r"{'User profile': '%s.', 'Watching history': '%s'}" % (profile, history)

    def _get_gpt_emb(self, prompt):
        """
        Calls the OpenAI API to get the embedding for the given prompt.
        Uses model: text-embedding-ada-002
        """
        try:
            response = self.client.embeddings.create(
                input=prompt,
                model="text-embedding-ada-002"
            )
            # Extract embedding from response
            emb = response.data[0].embedding
            return emb
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            return None

    def generate_embeddings(self):
        """
        Main loop to generate embeddings for all users and save to a tensor file.

        """
        all_user_emb = []

        # Iterate through all users based on user_info length
        # Assuming user_ids map linearly to range(len(user_info))
        for u in tqdm(range(len(self.user_info)), desc='embedding users'):
            prompt = self._prompt_construction(u)

            if prompt:
                emb = self._get_gpt_emb(prompt)
                if emb:
                    all_user_emb.append(emb)
                else:
                    # Handle failure case (e.g., append zero vector or skip)
                    # Here we append a zero vector of size 1536 (ada-002 size) to keep shape consistent
                    all_user_emb.append([0.0] * 1536)
            else:
                all_user_emb.append([0.0] * 1536)

        # Convert to Tensor
        all_user_emb_tensor = torch.tensor(all_user_emb)
        # print(f"Embeddings shape: {all_user_emb_tensor.shape}")

        # Save Tensor
        return all_user_emb_tensor