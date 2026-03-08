"""
Representation extraction utilities for model analysis
"""

import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


class RepresentationExtractor:
    """Extract representations from transformer models for analysis"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.last_layer_repr = None
        self.hook = None

    def hook_fn(self, module, input, output):
        """Hook function to capture last layer representations"""
        self.last_layer_repr = output.detach().cpu()

    def register_hook(self):
        """Register hook on the final layer norm"""
        self.hook = self.model.transformer.ln_f.register_forward_hook(self.hook_fn)

    def remove_hook(self):
        """Remove the hook"""
        if self.hook:
            self.hook.remove()

    def extract_representation(self, input_batch, seq_info):
        """Extract representation for a single batch"""
        self.model.eval()
        all_representations = []
        with torch.no_grad():
            for _ in range(seq_info["new_len"]):
                logits = self.model(input_batch)
                logits = logits[:, -1, :]
                inp_next = torch.argmax(logits, -1, keepdims=True)
                input_batch = torch.cat((input_batch, inp_next), dim=1)
                final_representation = self.last_layer_repr[:, -1, :]
                # create a new dimension in between 1 and 0
                final_representation = final_representation.unsqueeze(1)
                all_representations.append(final_representation)
        all_representations = torch.cat(all_representations, dim=1)
        all_representations = all_representations.reshape(
            all_representations.shape[0], -1
        )

        return all_representations, input_batch[:, seq_info["prompt_pos_end"] :]


def create_unique_function_permutations(all_function_lists):
    """Create unique permutations from function lists and assign colors"""
    # Remove identity functions and create tuples
    simplified_permutations = []
    for fn_list in all_function_lists:
        simplified = tuple(fn for fn in fn_list if fn != "identity")
        simplified_permutations.append(simplified)

    # Get unique permutations
    unique_permutations = list(set(simplified_permutations))

    # Create color mapping - convert tuples to strings for LabelEncoder
    simplified_perms_strings = [str(perm) for perm in simplified_permutations]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(simplified_perms_strings)

    return simplified_permutations, unique_permutations, labels, label_encoder


def perform_tsne_and_visualize(
    train_representations,
    test_representations,
    train_function_lists,
    test_function_lists,
    save_path=None,
    n_components=2,
    perplexity=30,
    random_state=42,
):
    """Perform t-SNE and create visualization"""

    # Combine all representations
    all_representations = np.vstack([train_representations, test_representations])
    all_function_lists = train_function_lists + test_function_lists

    # Create dataset labels (train vs test)
    dataset_labels = ["train"] * len(train_representations) + ["test"] * len(
        test_representations
    )

    # Create unique function permutations and labels
    simplified_perms, unique_perms, perm_labels, label_encoder = (
        create_unique_function_permutations(all_function_lists)
    )

    print(f"Found {len(unique_perms)} unique function permutations")
    print(f"Total representations: {len(all_representations)}")

    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1,
    )
    embeddings = tsne.fit_transform(all_representations)

    # Create visualization
    plt.figure(figsize=(15, 12))

    # Plot 1: Color by function permutations
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=perm_labels, cmap="tab20", alpha=0.7, s=20
    )
    plt.title("t-SNE colored by Function Permutations")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 2: Color by dataset (train vs test)
    plt.subplot(2, 2, 2)
    colors = ["blue" if label == "train" else "red" for label in dataset_labels]
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.7, s=20)
    plt.title("t-SNE colored by Dataset (Blue=Train, Red=Test)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 3: Train data only, colored by function permutations
    plt.subplot(2, 2, 3)
    train_embeddings = embeddings[: len(train_representations)]
    train_perm_labels = perm_labels[: len(train_representations)]
    plt.scatter(
        train_embeddings[:, 0],
        train_embeddings[:, 1],
        c=train_perm_labels,
        cmap="tab20",
        alpha=0.7,
        s=20,
    )
    plt.title("Train Data Only - Colored by Function Permutations")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 4: Test data only, colored by function permutations
    plt.subplot(2, 2, 4)
    test_embeddings = embeddings[len(train_representations) :]
    test_perm_labels = perm_labels[len(train_representations) :]
    plt.scatter(
        test_embeddings[:, 0],
        test_embeddings[:, 1],
        c=test_perm_labels,
        cmap="tab20",
        alpha=0.7,
        s=20,
    )
    plt.title("Test Data Only - Colored by Function Permutations")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.savefig("tsne_visualization.png", dpi=300, bbox_inches="tight")
        print("Visualization saved to tsne_visualization.png")

    plt.show()

    # Print some statistics
    print("\nFunction Permutation Statistics:")
    perm_counts = defaultdict(int)
    for perm in simplified_perms:
        perm_counts[perm] += 1

    # Show top 10 most common permutations
    sorted_perms = sorted(perm_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 most common function permutations:")
    for i, (perm, count) in enumerate(sorted_perms[:10]):
        print(f"{i+1}. {perm}: {count} occurrences")

    return embeddings, perm_labels, dataset_labels, unique_perms, simplified_perms


def save_representation_results(results, save_path):
    """Save representation analysis results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")


def load_representation_results(load_path):
    """Load representation analysis results"""
    with open(load_path, "rb") as f:
        results = pickle.load(f)
    print(f"Results loaded from {load_path}")
    return results
