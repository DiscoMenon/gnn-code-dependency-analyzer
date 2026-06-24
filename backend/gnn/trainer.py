import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from backend.gnn.model import CodeGNN
from backend.gnn.dataset import RepoGraphDataset
from pathlib import Path

def train_multi_repo(
    dataset: RepoGraphDataset,
    epochs: int = 300,
    lr: float = 0.01,
    save_path: str = "data/model.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Repos in dataset: {len(dataset)}")

    # Infer input dim from first graph
    input_dim = dataset[0][0].x.shape[1]
    model = CodeGNN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Train on every repo graph each epoch
        for data, _, repo_name in dataset:
            data = data.to(device)

            if data.edge_index.size(1) == 0:
                continue  # skip graphs with no edges

            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)

            # Positive edges
            pos_src = data.edge_index[0]
            pos_dst = data.edge_index[1]
            pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

            # Negative edges
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_index.size(1)
            )
            neg_src = neg_edge_index[0]
            neg_dst = neg_edge_index[1]
            neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, save_path)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to {save_path}")
    return model


def save_model(model: CodeGNN, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, input_dim: int = 7) -> CodeGNN:
    model = CodeGNN(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model