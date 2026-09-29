# SIH26227 — Offline Vision-Language Platform for Satellite Image Semantic Search and Change Monitoring​

Team **COMMITment issuesUNI**

Search satellite imagery by describing what you're looking for, and flag likely change
between two dates at a location — designed to run fully offline once imagery, models and
the index have been staged onto a machine.

## What this does

- **Search by description or example** — type a query like *"newly built structures near a
  river"*, or point to an example tile, and get back the closest-matching imagery.
- **Change over time** — compare the same location across two dates and flag whether
  something changed, roughly what kind of change, and the earliest date it's visible.
- **Cloud/shadow-aware** — quality masks are part of the pipeline design, so clouds,
  shadows and misaligned captures are handled before a result is treated as real change.
- **Analyst-facing review** — a ranked before/after queue with confirm/reject and
  provenance (source scene, acquisition date, sensor) on every result.
- **Offline-first** — after one-time staging (downloading models and building the index),
  no internet connection or cloud API is required to run searches or comparisons.

---

## Project structure

```
.
├── requirements.txt         # top-level (frontend) dependencies
├── scripts/
│   ├── download_scene.py    # backend: STAC ingestion, GeoRSCLIP embedding, FAISS index, search + change comparison
│   ├── requirements.txt     # backend-only dependencies
│   └── app.py               # Streamlit frontend — search UI, before/after slider, review queue
├── test_tiles               # sample tile images used for local testing (not part of the live index)
└── .gitignore
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <https://github.com/paneertikkaa/SIH26-COMMITment-issues/blob/main/app.py>
cd <https://github.com/paneertikkaa/SIH26-COMMITment-issues/tree/main>
pip install -r requirements.txt
pip install -r scripts/requirements.txt
```

### 2. Get the GeoRSCLIP checkpoint

The embedding model's weights aren't fetched automatically. Download the checkpoint from
Hugging Face:

- Model: [`Zilun/GeoRSCLIP`](https://huggingface.co/Zilun/GeoRSCLIP)
- File needed: `RS5M_ViT-B-32_RET-2.pt`
- Place it at: `models/ckpt/RS5M_ViT-B-32_RET-2.pt`

### 3. Imagery access

Imagery is pulled from Sentinel-2 L2A via the Microsoft Planetary Computer STAC catalogue.

---

## Usage

**Build (or rebuild) the tile index and run a test search:**

```bash
python scripts/download_scene.py
```

This connects to the STAC catalogue, pulls a Sentinel-2 scene for the configured area and
date range, embeds a set of tiles from it with GeoRSCLIP, saves the resulting FAISS index
and metadata to `models/`, and runs one example text query against it as a sanity check.

**Launch the interface:**

```bash
streamlit run app.py
```

Provides a map view, a natural-language/example-image search box, a before/after
comparison slider, and a ranked review queue with provenance detail per result.

---

## Model & dataset attribution

Every pretrained model and dataset used needs its origin and licence declared.

| Component | Source | Licence |
|---|---|---|
| CLIP | Radford et al., ICML 2021 — [openai/CLIP](https://github.com/openai/CLIP) | MIT |
| GeoRSCLIP (RS5M) | Zhang et al., arXiv 2306.11300 — [Zilun/GeoRSCLIP](https://huggingface.co/Zilun/GeoRSCLIP) | CC |
| OpenCLIP | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | MIT |
| FAISS | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | MIT |
| Sentinel-2 L2A | ESA Copernicus, via Microsoft Planetary Computer | Copernicus open data licence (free, full, open) |
| STAC / COG | [stacspec.org](https://stacspec.org) / [cogeo.org](https://cogeo.org) | Open specifications (not licensed software) |

Evaluated and not adopted for this project: RemoteCLIP, Clay Foundation Model, TerraMind.

---

## Tech stack

Python, PyTorch, open_clip, GeoRSCLIP · FAISS · rasterio, pystac-client,
planetary-computer · Streamlit, Folium, streamlit-image-comparison

---

## License
This project uses third-party models. See the [licenses folder](./licenses/) for full copyright and license details.