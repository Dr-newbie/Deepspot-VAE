# Deepspot-VAE
Inference Xenium matrix using H&amp;E foundation model &amp; imputation matrix with VAE 


[ Slide 이미지 (.png) ]
        ↓
[ Crop 생성 (imagefeature_extraction.py) ]
        ↓
[ 유효 Crop 필터링 + 저장 (coord_with_white.csv) ]
        ↓
[ 임베딩 추출 (ViT + phi_spot) → crop_phi_embeddings.npy ]
        ↓
[ crop_coordinates.csv 로부터 타일 단위 crop 수 추정 ]
        ↓
[ h5ad 표현행렬 타일 단위로 정렬 → 임베딩과 매칭 ]
        ↓
[ (embedding, expression) pair dataset 생성 ]
        ↓
[ VAE 학습 → gene expression 복원 ]
