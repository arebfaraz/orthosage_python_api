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

## CA Ceph / PA-cephalogram endpoints

Ported from the `CA Ceph` project (AALNet, see `ca_ceph/`). Upload an original
**posteroanterior (PA) cephalogram**; it is skull-cropped, letterboxed to 800×800,
run through the fold ensemble, and mapped back to original pixels.
Research use only: not a medical device, not for clinical decisions
(AALNet code + weights are CC BY-NC 4.0, see `ca_ceph/aalnet/LICENSE`).

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| POST | `/analyze/ca-ceph` | JSON | 33 landmarks (`x`, `y` in original px, `fold_spread`) + asymmetry measurements |
| POST | `/analyze/ca-ceph-image` | `image/jpeg` | landmarks + midlines (`style=points`) or a tracing (`style=tracing`) |

Form fields: `file` (required); `pixel_spacing` (mm/px; measurements are in px
if omitted) on `/analyze/ca-ceph`; `style` (`points`|`tracing`) and
`show_names` (bool) on `/analyze/ca-ceph-image`.

Only **fold0** is shipped (best validation MRE of the five; the paper's
reported accuracy is the mean of single fold models, not an ensemble), so
`fold_spread` is always 0. Its weights are **not in git** (~294 MB): put them at
`ca_ceph/weights/fold0/checkpoint_best.pth` (or set `CA_CEPH_WEIGHTS_DIR`).
Every `fold*/` found there is loaded, so the other folds from the CA Ceph
project can be dropped back in for a 5-fold ensemble. Measured on 2 CPU
threads: 1 fold ≈ 10 s and ≈ 3.5 GB peak RAM per request; 5 folds ≈ 70–125 s.

```bash
curl -X POST "https://py.ekict.com/analyze/ca-ceph" -F "file=@pa_ceph.jpg" -F "pixel_spacing=0.1"
curl -X POST "https://py.ekict.com/analyze/ca-ceph-image" -F "file=@pa_ceph.jpg" -F "style=tracing" -o tracing.jpg
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

## Facial photo analysis endpoints

Ported from `orthosage_python2_api` (`/analyze-face` on py1.ekict.com), see
`face_analysis/`. Upload a **frontal face photo**; MediaPipe FaceLandmarker
(model bundled at `face_analysis/assets/face_landmarker.task`) finds the face,
and 22 named soft-tissue landmarks + reference lines (midsagittal, horizontal,
lip canting, jaw taper) are derived. Request and response are unchanged from
py1; only the domain is new.

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| POST | `/analyze-face` | JSON | `width`, `height`, `landmarks` (`key`, `label`, `group`, `x`, `y`, `movable`), `lines`; 422 if no face found |
| POST | `/analyze-face?format=image` | `image/png` | landmarks + lines drawn on the photo |
| POST | `/analyze-face/recompute` | JSON | body `{"landmarks": [{"key","x","y"}], "height"}` → `{"lines": [...]}` |

On a headless Linux server MediaPipe needs `apt-get install -y libgles2 libegl1`;
otherwise `/health` shows the error under `face` and these return 503.

```bash
curl -X POST "https://py.ekict.com/analyze-face" -F "file=@face.jpg"
curl -X POST "https://py.ekict.com/analyze-face?format=image" -F "file=@face.jpg" -o face_lines.png
curl -X POST "https://py.ekict.com/analyze-face/recompute" -H "Content-Type: application/json" \
     -d '{"landmarks":[{"key":"pupil_r","x":120,"y":88},{"key":"pupil_l","x":220,"y":90}],"height":400}'
```

## Intraoral occlusogram endpoint

Ported from `orthosage_occlusogram` (IOP-Compass API, its `POST /analyze`), see
`occlusogram/`. Upload **one intraoral photo** (frontal, left/right buccal,
upper/lower occlusal). A ResNet18 classifies the view, Mask R-CNN segments
FDI-numbered teeth, and on frontal views a YOLO model grades gingivitis per
tooth. Request fields and response body are unchanged; only the path is new
(`/analyze` here is the X-ray route).

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| POST | `/analyze/occlusogram` | JSON | `classification`, `tooth_count`, `instances` (`fdi`, `score`, `contours`, `bbox`, `centroid`, `gingivitis`), `issues`, `gingivitis`, `timings` |

Form fields (all optional except `file`):

- `file`: PNG, JPEG or WebP, up to 40 MB.
- `mirror_acquisition` (default `true`): photo taken through an intraoral
  mirror. If wrong, left/right buccal swap and teeth get the contralateral FDI.
- `view`: skip classification (`frontal`, `left_buccal`, `right_buccal`,
  `upper_occlusal`, `lower_occlusal`).
- `gingivitis` (default `true`): grade gingivitis (frontal views only).

Errors: 400 (not an image / unknown view), 413 (too large), 415 (unsupported
type), 503 (models not loaded). About 1.5–3.5 s per photo on CPU.

```bash
curl -X POST "https://py.ekict.com/analyze/occlusogram" -F "file=@photo.jpg"
curl -X POST "https://py.ekict.com/analyze/occlusogram" -F "file=@photo.jpg" -F "mirror_acquisition=false"
```
