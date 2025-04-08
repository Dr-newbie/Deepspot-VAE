import os
import torch
import numpy as np
import anndata as ad
from torch.utils.data import Dataset, DataLoader
from VAE import VAE
from torch import nn, optim
from tqdm import tqdm
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_path, h5ad_dir):
        self.embeddings = np.load(embedding_path)
        self.h5ad_files = sorted([
            os.path.join(h5ad_dir, f) for f in os.listdir(h5ad_dir) if f.endswith(".h5ad")
        ])

        # 각 tile 별 임베딩을 쪼개서 저장
        self.tile_to_embeddings = {}
        num_tiles = len(self.embeddings) // len(self.h5ad_files)
        for i, f in enumerate(self.h5ad_files):
            tile_id = os.path.basename(f).split('.')[0]  # 예: combined_cluster_core_1
            start = i * num_tiles
            end = start + num_tiles
            self.tile_to_embeddings[tile_id] = self.embeddings[start:end]

        self.paired_data = []
        for f in tqdm(self.h5ad_files, desc="🔗 Matching embeddings and expression by tile_id"):
            adata = ad.read_h5ad(f)

            if 'tile_id' not in adata.obs.columns:
                print(f"❌ Missing tile_id in {f}, skipping.")
                continue

            unique_tile_ids = adata.obs['tile_id'].unique()
            for tile_id in unique_tile_ids:
                if tile_id not in self.tile_to_embeddings:
                    print(f"⚠️ tile_id {tile_id} not in embeddings, skipping.")
                    continue

                expr_subset = adata[adata.obs['tile_id'] == tile_id].X
                if hasattr(expr_subset, "toarray"):
                    expr_subset = expr_subset.toarray()

                embs = self.tile_to_embeddings[tile_id]

                if len(embs) != expr_subset.shape[0]:
                    print(f"❗ Mismatch in count for tile_id {tile_id}, skipping.")
                    continue

                for i in range(len(embs)):
                    self.paired_data.append((embs[i], expr_subset[i]))

        if len(self.paired_data) == 0:
            raise ValueError("❌ No matched embedding-expression pairs found.")

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        emb, expr = self.paired_data[idx]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(expr, dtype=torch.float32)



# ------------------- 설정 -----------------------
embedding_path = 'crop_phi_embeddings.npy'
h5ad_dir = '/data/project/banana9903/DeepSpot/0050586/0050586_h5ad'

batch_size = 4
latent_dim = 128
epochs = 30
lr = 3e-4
output_dim = 5000

# ------------------- 데이터 로딩 -----------------------
dataset = EmbeddingDataset(embedding_path, h5ad_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------- 모델 정의 -----------------------
input_dim = dataset[0][0].shape[0]
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)
loss_fn = nn.MSELoss()

train_log = []

# ------------------- 학습 루프 -----------------------
vae.train()
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    recon_loss_total = 0
    kld_loss_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=100, bar_format='{l_bar}{bar} | {elapsed}<{remaining} [{rate_fmt}]')

    for emb, expr in pbar:
        emb, expr = emb.to(device), expr.to(device)
        recon, mu, logvar = vae(emb)

        recon_loss = loss_fn(recon, expr)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kld_loss_total += kld_loss.item()

        pbar.set_postfix(loss=loss.item())

    elapsed = time.time() - start_time

    avg_total_loss = total_loss / len(dataloader)
    avg_recon_loss = recon_loss_total / len(dataloader)
    avg_kld_loss = kld_loss_total / len(dataloader)

    train_log.append({
        "epoch": epoch + 1,
        "total_loss": avg_total_loss,
        "recon_loss": avg_recon_loss,
        "kld_loss": avg_kld_loss,
        "time_sec": elapsed
    })

    print(f"✅ Epoch {epoch+1}: Total Loss = {avg_total_loss:.4f} | Recon = {avg_recon_loss:.4f} | KLD = {avg_kld_loss:.4f} | Time: {elapsed:.2f}s")

# ------------------- 저장 -----------------------
torch.save(vae.state_dict(), "vae_trained.pth")
print("✅ 모델 저장 완료: vae_trained.pth")

log_df = pd.DataFrame(train_log)
log_df.to_csv("vae_training_log.csv", index=False)
print("📄 로그 저장 완료: vae_training_log.csv")
