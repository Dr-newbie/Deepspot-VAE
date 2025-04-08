# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

```mermaid
flowchart TD
    A[Slide Image (.png)] --> B[Crop generation (imagefeature_extraction.py)]
    B --> C[White Crop filtering (coord_with_white.csv)]
    C --> D[Embedding extraction (image_encoder_extraction.py)]
    D --> E[crop_coordinates.csv based tile inference]
    E --> F[h5ad Expression alignment and matching]
    F --> G[Dataset generation (Embedding + Expression)]
    G --> H[VAE trainig (vae_train.py)]
    H --> I[Output: vae_trained.pth, vae_training_log.csv]
```

