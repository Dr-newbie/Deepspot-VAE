# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

# Image Encoder
The image encoder is UNI(https://github.com/mahmoodlab/UNI, the H&E foundation model (SSL settings)) based.
![Image](https://github.com/user-attachments/assets/2416be0d-2e08-40f5-b549-279aef2541da)
* not using the "Rho", predictor for gene expression.

# VAE
Using VAE for reconstruction & imputation for Xenium gene expression matrix



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
