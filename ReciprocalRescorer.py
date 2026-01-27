
import torch
import numpy as np



class ReciprocalRescorer:
    def __init__(self, device):
        """
        Initialize the Reranker module.

        Args:
            device: torch.device object (CPU or CUDA).
        """
        self.device = device

    def _reconstruct_matrix(self, A, B):
        """
        Reconstruct matrix C from index matrix A and value matrix B.
        C[i, A[i, j]] = B[i, j], other positions are 0.

        Args:
            A: 2D tensor of indices (same shape as B)
            B: 2D tensor of values (same shape as A)

        Returns:
            C: 2D tensor with C[i, A[i,j]] = B[i,j], others 0
        """
        rows, cols = A.shape
        max_col = A.max().item() + 1
        C = torch.zeros((rows, max_col), dtype=B.dtype, device=B.device)

        row_indices = torch.arange(rows, device=A.device).unsqueeze(1).expand_as(A)
        C[row_indices, A] = B

        return C

    def _pairwise_cosine_similarity(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise cosine similarity between rows of the input matrix.
        """
        # L2 Norm
        row_norms = torch.norm(matrix, p=2, dim=1, keepdim=True)
        row_norms = torch.clamp(row_norms, min=1e-8)

        # Normalize
        normalized_matrix = matrix / row_norms

        # Matrix Multiplication
        cosine_sim = torch.mm(normalized_matrix, normalized_matrix.t())

        return cosine_sim

    def _normalize_scores(self, scores):
        """Min-Max normalization supporting both Numpy and Tensor."""
        if isinstance(scores, np.ndarray):
            min_val = np.min(scores, axis=1, keepdims=True)
            max_val = np.max(scores, axis=1, keepdims=True)
            return (scores - min_val) / (max_val - min_val + 1e-8) + 1e-8
        elif isinstance(scores, torch.Tensor):
            min_val = scores.min(dim=1, keepdim=True)[0]
            max_val = scores.max(dim=1, keepdim=True)[0]
            return (scores - min_val) / (max_val - min_val + 1e-8) + 1e-8
        else:
            raise TypeError("Unsupported type for normalization")

    def _batched_weighted_topk_sum(self, A, B, k, batch_size=100):
        """
        Batched processing for weighted top-k sum to reduce memory usage.

        Args:
            A: Input value matrix [n_users, n_items]
            B: Similarity matrix [n_users, n_users]
            k: Number of neighbors to consider (sim_users)
            batch_size: Batch size for user processing
        """
        n_users, n_items = A.shape
        result = torch.zeros((n_users, n_items), device=A.device, dtype=A.dtype)

        for i in range(0, n_users, batch_size):
            end_idx = min(i + batch_size, n_users)
            batch_indices = slice(i, end_idx)

            # Get similarity top-k for current batch
            batch_B = B[batch_indices]
            topk_values, topk_indices = torch.topk(batch_B, k=k, dim=1)

            topk_values = self._normalize_scores(topk_values)

            # Select rows from A corresponding to similar users
            selected = A[topk_indices]  # [batch_size, k, n_items]

            # Weighted sum calculation
            # result = sum(similarity * value)
            weighted_sum = (topk_values.unsqueeze(-1) * selected).sum(dim=1)

            result[batch_indices] = weighted_sum

            # Memory cleanup
            del batch_B, topk_values, topk_indices, selected, weighted_sum
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

    def run_reciprocal_rescoring(self, col_emb, cfm_rating_mat, sem_score, sem_emb, alpha, sim_users, topk=20, topl=50):
        """
        Executes the hybrid reranking logic.

        Args:
            rating_mat: Original rating tensor [n_users, n_movies].
            global_score_np: Global scores from pairwise comparison (Numpy array).
            llm_emb: LLM embeddings tensor for user similarity.
            alpha: Weighting factor (0 to 1).
            sim_users: Number of similar users to consider.
            topk: Top K items to consider for initial indices.

        Returns:
            idx3: Reranked item indices tensor.
        """
        # Similarity Calculation
        similarity_col = self._pairwise_cosine_similarity(col_emb)
        similarity_sem = self._pairwise_cosine_similarity(sem_emb)

        # Extract Top-K from original ratings to get indices
        cfm_score, cfm_idx = torch.topk(cfm_rating_mat, topk, dim=1)

        # Normalize Scores
        normalize_sem_score = torch.tensor(
            self._normalize_scores(sem_score), device=self.device
        )
        normalize_cfm_score = torch.tensor(
            self._normalize_scores(cfm_score), device=self.device
        )

        # Reconstruct Full Matrices based on indices
        normalize_cfm_score_full = self._reconstruct_matrix(cfm_idx, normalize_cfm_score)
        normalize_sem_score_full = self._reconstruct_matrix(cfm_idx, normalize_sem_score)


        # Ch3.3.1 Semantic-infused Collaborative Rescoring
        cfm_score_infused = self._batched_weighted_topk_sum(
            normalize_cfm_score_full, similarity_sem, sim_users
        )

        # Ch3.3.2 Collaborative-infused Semantic Rescoring
        sem_score_infused = self._batched_weighted_topk_sum(
            normalize_sem_score_full, similarity_col, sim_users
        )

        # Ch3.3.3 Reciprocal Rescoring Fusion
        fused_score = alpha * sem_score_infused + (1 - alpha) * cfm_score_infused

        # Top-l Extraction for Calibration
        fused_score, fused_idx = torch.topk(fused_score, topl, dim=1)

        return fused_score, fused_idx