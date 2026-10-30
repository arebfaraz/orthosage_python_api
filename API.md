# Teeth Segmentation API

Serves the trained U-Net (`dental_xray_seg.h5`) over HTTP: upload a panoramic
dental X-ray, get back a teeth segmentation mask, an overlay, a
connected-component analysis image, and a tooth count.

## Run

```powershell
# from the project root
.\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Then open the interactive docs: <http://localhost:8000/docs>

- Inference runs on **CPU by default** (~1 s/image) so it never fights the GPU
  trainer. To use the GPU: set `API_USE_GPU=1` before starting.
- Point at a different model: set `MODEL_PATH=C:\path\to\model.h5`.

## Endpoints

| Method | Path       | Returns | Notes |
|--------|------------|---------|-------|
| GET    | `/health`  | JSON    | status, model name, GPU list |
| POST   | `/segment` | `image/png` | binary teeth mask at the input resolution |
| POST   | `/overlay` | `image/png` | mask drawn semi-transparent over the input |
| POST   | `/analyze` | JSON    | `teeth_count`, coverage, + base64 mask/overlay/CCA PNGs |

Query params:

- `threshold` (0–1, default `0.5`) — sigmoid cutoff for the mask.
- `/analyze` also: `erode_iteration` (3), `open_iteration` (2),
  `area_threshold` (2000 px), `include_images` (true).

## Examples

### curl

```bash
# just the mask
curl -X POST "http://localhost:8000/segment" -F "file=@xray.png" -o mask.png

# overlay, custom threshold
curl -X POST "http://localhost:8000/overlay?threshold=0.4" -F "file=@xray.png" -o overlay.png

# full analysis (JSON with tooth count)
curl -X POST "http://localhost:8000/analyze" -F "file=@xray.png"
```

### Python

```python
import requests, base64

r = requests.post("http://localhost:8000/analyze",
                  files={"file": open("xray.png", "rb")})
d = r.json()
print("teeth:", d["teeth_count"])
open("cca.png", "wb").write(base64.b64decode(d["cca_png_b64"]))
```

## Response shape (`/analyze`)

```json
{
  "teeth_count": 28,
  "mask_coverage_fraction": 0.18,
  "threshold": 0.5,
  "image_size": {"width": 3040, "height": 1200},
  "cca_error": null,
  "mask_png_b64": "iVBORw0K...",
  "overlay_png_b64": "iVBORw0K...",
  "cca_png_b64": "iVBORw0K..."
}
```

## Gingivitis / intraoral-photo endpoints

Ported from `BinaModel_project` (see `gingivitis/`). Upload an **intraoral
photo** (not an X-ray); every model in `gingivitis/models.yaml` runs on it
(gingivitis specialist, recession specialist, intraoral seg for caries/plaque).

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| POST | `/analyze-gingivitis` | JSON | findings per condition, anatomy counts, timings |
| POST | `/analyze-gingivitis-image` | `image/jpeg` | boxes/masks drawn + legend; summary in `X-Conditions-Found`, `X-Total-Findings`, `X-Inference-Ms` headers |

Query params (both): `models` (comma list, e.g. `gingivitis,recession`),
`conf` (0–1, overrides each model's threshold), `labels` (`short`|`full`|`none`),
`masks` (polygons; default false for JSON, true for image).
`/analyze-gingivitis` also takes `annotate=true` to embed the JPEG as base64.

Weights are **not in git** (`intraoral_seg.pt` is 115 MB). Copy them to the
server at `gingivitis/weights/{gingivitis,recession,intraoral_seg}.pt`, or set
`GINGIVITIS_WEIGHTS_DIR`. If they or torch/ultralytics are missing, the X-ray
endpoints still work and these return 503 (reason shown in `/health`).

```bash
curl -X POST "https://py.ekict.com/analyze-gingivitis" -F "file=@mouth.jpg"
curl -X POST "https://py.ekict.com/analyze-gingivitis-image" -F "file=@mouth.jpg" -o annotated.jpg
```

## Lateral-ceph endpoint

Ported from `orthosage_python2_api` (`POST /predict` on py1.ekict.com), see
`lateral_ceph/`. HRNet-W32, 29 lateral-cephalogram landmarks. Request and
response are unchanged from py1; only the URL is new.

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| POST | `/analyze/lateral-ceph` | JSON | `{"landmarks": {"S": {"x", "y", "label"}, ...}, "width", "height"}` in original pixels |
| POST | `/analyze/lateral-ceph?format=image` | `image/png` | landmarks drawn on the input |

Weights are **not in git**: copy py1's `runs/train/full_run/best.pt` to
`lateral_ceph/weights/best.pt` (or set `LATERAL_CEPH_WEIGHTS`).

```bash
# was: curl -X POST "https://py1.ekict.com/predict" -F "file=@ceph.png"
curl -X POST "https://py.ekict.com/analyze/lateral-ceph" -F "file=@ceph.png"
curl -X POST "https://py.ekict.com/analyze/lateral-ceph?format=image" -F "file=@ceph.png" -o landmarks.png
```
