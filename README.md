# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

# Image Encoder
The image encoder is UNI(https://github.com/mahmoodlab/UNI, the H&E foundation model (SSL settings)) based.
![Image](https://github.com/user-attachments/assets/2416be0d-2e08-40f5-b549-279aef2541da)
* not using the "Rho", predictor for gene expression.

# VAE
Using VAE for reconstruction & imputation for Xenium gene expression matrix
![Image](https://github.com/user-attachments/assets/16221348-0797-4fdb-b1e3-ee721320efa9)

# Code
    1. python imagefeature_extraction.py
    2. python image_encoder_extration.py
    3. python vae_train.py

# Code Flow
```mermaid
flowchart TD
    A[Slide Image] --> B[Crop Generation]
    B --> C[White Crop Filtering]
    C --> D[Embedding Extraction]
    D --> E[Tile Inference]
    E --> F[Expression Matching]
    F --> G[Dataset Construction]
    G --> H[VAE Training]
    H --> I[Output: Weights & Logs]
```

# Alignment final & Image tiling
Have to scaling our images & GT st spots! (!!!!Need Polygon of ST!!!!)


now we using 30x30 tiles. but have to use 224x224 -> if overlapped, just think it as "biological context" (30x30 tiles also could contains 2 or more spots)





