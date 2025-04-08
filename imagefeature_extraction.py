
import os
import cv2
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def process_tile(row, image_dir, spot_diameter):
    try:
        img_path = os.path.join(image_dir, row["source_image"])
        image = cv2.imread(img_path)

        if image is None:
            logging.warning(f"이미지 로드 실패: {img_path}")
            return None

        x = int(row.x_pixel - spot_diameter // 2)
        y = int(row.y_pixel - spot_diameter // 2)
        crop = image[x:x+spot_diameter, y:y+spot_diameter]

        if crop.shape[:2] != (spot_diameter, spot_diameter):
            return None

        white_score = np.mean(crop[:, :, :3])
        return white_score

    except Exception as e:
        logging.error(f"에러 발생: {e}")
        return None

def main():
    from deepspot.utils.utils_image import predict_spatial_transcriptomics_from_image_path
    from deepspot.utils.utils_image import get_morphology_model_and_preprocess
    from deepspot.utils.utils_image import crop_tile

    from deepspot import DeepSpot

    import matplotlib.image as mpimg
    from openslide import open_slide
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import squidpy as sq
    import anndata as ad
    import pandas as pd
    import numpy as np
    import pyvips
    import torch
    import math
    import yaml
    import PIL

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device

    out_folder = "feature_extraction_images"
    white_cutoff = 200  # recommended, but feel free to explore
    downsample_factor = 10 # downsampling the image used for visualisation in squidpy
    model_weights = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    model_hparam = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    gene_path = '/data/project/banana9903/DeepSpot/pretrained_model_weights/DeepSpot_pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv'
    image_path = '/data/project/banana9903/DeepSpot/0050586/0050586_images'


    with open(model_hparam, "r") as stream:
        config = yaml.safe_load(stream)
    config

    n_mini_tiles = config['n_mini_tiles'] # number of non-overlaping subspots
    spot_diameter = config['spot_diameter'] = 50 # spot diameter
    spot_distance = config['spot_distance'] = 50# distance between spots
    image_feature_model = config['image_feature_model'] 
    image_feature_model


    import os
    import cv2
    import pandas as pd
    from tqdm import tqdm  # ✅ 핵심 수정

    output_dir = "output_crops"
    os.makedirs(output_dir, exist_ok=True)

    spot_diameter = 60
    spot_distance = 100

    coord_list = []

    image_dir = "/data/project/banana9903/DeepSpot/0050586/0050586_images"
    image_list = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    for img_name in tqdm(image_list):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ 이미지 불러오기 실패: {img_path}")
            continue

        h, w, _ = image.shape
        for i, x in enumerate(range(spot_diameter // 2, h - spot_diameter // 2, spot_distance)):
            for j, y in enumerate(range(spot_diameter // 2, w - spot_diameter // 2, spot_distance)):
                crop = image[x - spot_diameter//2:x + spot_diameter//2, y - spot_diameter//2:y + spot_diameter//2]

                if crop.shape[0] != spot_diameter or crop.shape[1] != spot_diameter:
                    continue

                save_path = os.path.join(output_dir, f"{img_name.replace('.png','')}_x{x}_y{y}.png")
                cv2.imwrite(save_path, crop)

                coord_list.append({
                    "source_image": img_name,
                    "x_pixel": x,
                    "y_pixel": y,
                    "crop_path": save_path
                })

    # 저장
    coord_df = pd.DataFrame(coord_list)
    coord_df.to_csv("crop_coordinates.csv", index=False)
    print(f"✅ Crop 완료! 총 {len(coord_df)}개 저장됨.")


    coord_df = pd.read_csv("crop_coordinates.csv")  # 앞에서 만든 coord_df

    is_white = []

    for idx, row in tqdm(coord_df.iterrows(), total=len(coord_df)):
        img_path = os.path.join(image_dir, row["source_image"])
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ 이미지 로드 실패: {img_path}")
            is_white.append(np.nan)
            continue

        # crop 위치 계산
        x = int(row.x_pixel - spot_diameter // 2)
        y = int(row.y_pixel - spot_diameter // 2)
        crop = image[x:x+spot_diameter, y:y+spot_diameter]

        if crop.shape[:2] != (spot_diameter, spot_diameter):
            is_white.append(np.nan)
            continue

        white_score = np.mean(crop[:, :, :3])
        is_white.append(white_score)

    # 결과 병합
    coord_df["is_white"] = is_white
    coord_df.to_csv("coord_with_white.csv", index=False)

    # 밝기 기준으로 유효 타일만 필터링
    filtered_coord_df = coord_df[coord_df["is_white"] < 230].reset_index(drop=True)

    # feature 추출을 위한 count matrix 자리 잡기
    #counts = np.empty((len(filtered_coord_df), selected_genes_bool.sum()))

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
