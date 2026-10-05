import os
import pickle

import faiss
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont

# ==========================================================
# Settings — change these if you want different output
# ==========================================================

QUERIES = ["urban buildings", "river", "dense vegetation"]
TOP_K = 3            # how many result tiles to show per query
TILE_SIZE = 384      # each tile is shown at this many pixels wide/tall
OUT_DIR = "figures"

INDEX_PATH = os.path.join("models", "tile_index.faiss")
META_PATH = os.path.join("models", "tile_metadata.pkl")
CKPT_PATH = os.path.join("models", "ckpt", "RS5M_ViT-B-32_RET-2.pt")

# ==========================================================
# Load model + index (no internet needed)
# ==========================================================

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(checkpoint, strict=True)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded index with {index.ntotal} tiles")


# ==========================================================
# Helpers
# ==========================================================

def load_font(size):
    # Windows font first, then fall back to PIL's built-in font
    for name in ["arial.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def search(query_text, k):
    tokens = tokenizer([query_text])
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    vec = (text_features[0] / text_features[0].norm()).numpy().astype("float32").reshape(1, -1)
    scores, indices = index.search(vec, k)
    return [(metadata[i], float(s)) for s, i in zip(scores[0], indices[0])]


def make_figure(query_text, results, out_path):
    pad = 20
    header_h = 90
    caption_h = 90
    width = pad + (TILE_SIZE + pad) * len(results)
    height = header_h + TILE_SIZE + caption_h + pad

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(32)
    sub_font = load_font(18)
    cap_font = load_font(17)

    draw.text((pad, 15), f'Query: "{query_text}"', fill=(20, 20, 20), font=title_font)
    draw.text((pad, 58), "Top matching Sentinel-2 tiles (GeoRSCLIP + FAISS)", fill=(90, 90, 90), font=sub_font)

    for rank, (meta, score) in enumerate(results, start=1):
        x = pad + (rank - 1) * (TILE_SIZE + pad)
        y = header_h

        tile = Image.open(meta["image_path"]).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + TILE_SIZE, y + TILE_SIZE], outline=(60, 60, 60), width=2)

        date = meta["date"][:10]
        col, row = meta["window"]
        draw.text((x, y + TILE_SIZE + 8), f"#{rank}   similarity {score:.3f}", fill=(20, 20, 20), font=cap_font)
        draw.text((x, y + TILE_SIZE + 32), f"Date: {date}", fill=(90, 90, 90), font=cap_font)
        draw.text((x, y + TILE_SIZE + 54), f"Tile offset: ({col}, {row})", fill=(90, 90, 90), font=cap_font)

    canvas.save(out_path)
    print("Saved", out_path)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for query in QUERIES:
        results = search(query, TOP_K)
        safe_name = query.replace(" ", "_")
        make_figure(query, results, os.path.join(OUT_DIR, f"search_{safe_name}.png"))

    print("\nDone. Open the 'figures' folder.")