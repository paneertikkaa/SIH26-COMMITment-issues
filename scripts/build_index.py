from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import torch
import open_clip
import faiss
import pickle

# ==========================================================
# Setup — runs once when this file is loaded/imported
# ==========================================================

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
checkpoint = torch.load(r"models\ckpt\RS5M_ViT-B-32_RET-2.pt", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

gdal_env = rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY=5,
    GDAL_HTTP_RETRY_DELAY=1,
)

def fetch_tile(item, col_off, row_off, size=512):
    visual_asset = item.assets["visual"]
    with gdal_env:
        with rasterio.open(visual_asset.href) as src:
            window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
            chunk = src.read(window=window)

    if chunk.shape[1:] != (size, size):
        return None  # window fell off the edge of the scene

    rearranged = chunk.transpose(1, 2, 0)
    return Image.fromarray(rearranged).convert("RGB")

# ==========================================================
# Reusable function 1: fetch + embed one tile
# ==========================================================

def fetch_and_embed_tile(item, col_off, row_off, size=512):
    visual_asset = item.assets["visual"]
    with gdal_env:
        with rasterio.open(visual_asset.href) as src:
            window = Window(col_off=col_off, row_off=row_off, width=size, height=size)
            chunk = src.read(window=window)

    if chunk.shape[1:] != (size, size):
        return None  # window fell off the edge of the scene

    rearranged = chunk.transpose(1, 2, 0)
    tile_image = Image.fromarray(rearranged).convert("RGB")
    tile_input = preprocess(tile_image).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(tile_input)

    vec = features[0] / features[0].norm()
    return vec


# ==========================================================
# Reusable function 2: build a FAISS index from a scene
# ==========================================================

def build_tile_index(item, window_offsets):
    vectors = []
    metadata = []

    for col_off, row_off in window_offsets:
        vec = fetch_and_embed_tile(item, col_off, row_off)
        if vec is None:
            continue
        vectors.append(vec.numpy())
        metadata.append({
            "scene_id": item.id,
            "date": str(item.datetime),
            "window": (col_off, row_off),
        })
        print(f"Embedded tile at offset ({col_off}, {row_off})")

    vectors_np = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)

    faiss.write_index(index, "models/tile_index.faiss")
    with open("models/tile_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index built with {index.ntotal} tiles, saved to models/tile_index.faiss")
    return index, metadata


# ==========================================================
# Reusable function 3: text search over the index
# ==========================================================

def search_tiles(query_text, index, metadata, k=3):
    text_tokens = tokenizer([query_text])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_vec = (text_features[0] / text_features[0].norm()).numpy().astype("float32").reshape(1, -1)

    scores, indices = index.search(text_vec, k=k)
    results = [(metadata[idx], float(score)) for score, idx in zip(scores[0], indices[0])]
    return results


# ==========================================================
# Reusable function 4: compare two tile vectors for change
# ==========================================================

def analyze_temporal_change(vec1, vec2, threshold=0.6):
    similarity = torch.dot(vec1, vec2).item()
    return {
        "similarity": similarity,
        "detected": similarity < threshold,  # placeholder threshold, not tuned
    }


# ==========================================================
# Main — only runs when you execute this file directly
# ==========================================================

if __name__ == "__main__":
    bbox = [77.05, 28.45, 77.35, 28.75]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2026-01-01/2026-03-01",
    )
    items = list(search.items())
    tile_id = items[0].id.split("_")[-2]
    same_tile_items = [i for i in items if i.id.split("_")[-2] == tile_id]
    item = sorted(same_tile_items, key=lambda i: i.datetime)[-1]

    print("Indexing scene:", item.id)

    window_offsets = [(0, 0), (2000, 0), (4000, 2000), (0, 4000), (6000, 6000), (8000, 3000)]
    index, metadata = build_tile_index(item, window_offsets)

    query = "urban buildings"
    results = search_tiles(query, index, metadata)
    print(f"\nTop results for query: '{query}'")
    for meta, score in results:
        print(meta, "score:", score)