# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 

<pre> ```text 📦 Pipeline: Slide Image → Crop → Embedding → Gene Expression Matching → VAE Training [ Slide 이미지 (.png) ] │ ▼ [ Crop 생성 ] (imagefeature_extraction.py) │ ▼ [ 유효 Crop 필터링 (밝기 기준) ] coord_with_white.csv 저장 │ ▼ [ 임베딩 추출 (ViT encoder + phi_spot) ] image_encoder_extraction.py └─> 출력: crop_phi_embeddings.npy │ ▼ [ crop_coordinates.csv 로부터 tile 기준 crop 개수 추정 ] │ ▼ [ h5ad gene expression 정렬 + 매칭 ] (각 타일 별 expression matrix) vae_train.py └─> tile_to_embeddings 및 expression matching │ ▼ [ (embedding, expression) pair dataset 생성 ] │ ▼ [ VAE 학습 ] vae_train.py └─> 출력: vae_trained.pth, vae_training_log.csv ``` </pre>
