# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

<pre> ```mermaid flowchart TD A[📂 Slide Image (.png)] --> B[🧩 Crop 생성<br>imagefeature_extraction.py] B --> C[📉 White Crop 필터링<br>coord_with_white.csv 생성] C --> D[📦 임베딩 추출<br>ViT encoder + phi_spot<br>image_encoder_extraction.py] D --> E[🧮 crop_coordinates.csv 기반 tile 추정] E --> F[🧬 h5ad Expression 정렬 및 매칭<br>tile_id 기반 정렬] F --> G[📊 Dataset 생성<br>(Embedding, Expression Pair)] G --> H[🔧 VAE 학습<br>vae_train.py] H --> I[💾 출력: vae_trained.pth, vae_training_log.csv] ``` </pre>
