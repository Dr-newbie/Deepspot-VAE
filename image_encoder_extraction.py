import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deepspot.utils.utils_image import get_morphology_model_and_preprocess
from deepspot import DeepSpot
import yaml

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load filtered crop info
    coord_df = pd.read_csv("coord_with_white.csv")
    coord_df = coord_df[coord_df['is_white'] < 230].reset_index(drop=True)

    # -----------------------------
    # ✅ Load DeepSpot configuration and weights
    model_path = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    model_config_path = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    gene_path = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv'

    genes = pd.read_csv(gene_path)
    output_size = genes[genes["isPredicted"]].shape[0]

    model = DeepSpot(
        n_ensemble_phi=10,
        p_phi=0.3,
        input_size=1024,
        phi2rho_size=1536,
        output_size=output_size
    )

    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Use one phi_spot
    phi = model.phi_spot[0]
    phi.eval()

    # -----------------------------
    # ✅ Load ViT encoder
    morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
        model_name='uni',
        device=device,
        model_path='/data/project/banana9903/huggingface/hub/models--MahmoodLab--UNI/blobs/56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40/pytorch_model.bin'
    )
    morphology_model = morphology_model.to(device)
    morphology_model.eval()

    # -----------------------------
    # ✅ Embedding 추출
    embeddings = []
    for idx, row in tqdm(coord_df.iterrows(), total=len(coord_df)):
        crop_path = row["crop_path"]
        image = cv2.imread(crop_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # ensure input size
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = morphology_model(image_tensor)
            embedding = phi(feat).cpu().numpy().squeeze()
            embeddings.append(embedding)

    embeddings = np.stack(embeddings)
    np.save("crop_phi_embeddings.npy", embeddings)
    print(f"✅ 추출 완료: {embeddings.shape} saved to crop_phi_embeddings.npy")


if __name__ == "__main__":
    main()
