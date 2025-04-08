# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

```mermaid
flowchart TD
    A[Slide Image (.png)] --> B[Crop 생성<br>imagefeature_extraction.py]
    B --> C[White Crop 필터링<br>coord_with_white.csv]
    C --> D[Embedding 추출<br>image_encoder_extraction.py]
    D --> E[crop_coordinates.csv 기반 tile 추정]
    E --> F[h5ad Expression 정렬 및 매칭]
    F --> G[Dataset 생성<br>(Embedding, Expression Pair)]
    G --> H[VAE 학습<br>vae_train.py]
    H --> I[출력: vae_trained.pth, vae_training_log.csv]
```
