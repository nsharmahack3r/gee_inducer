# gee-inducer

Adds Google Earth Engine (GEE) satellite covariates to animal GPS fix datasets.

- Source definitions (assets, bands, QA masks, composite windows) are in `covariates/sources.yaml`.
- Input and output folders are set in `.env`.

Only Sentinel-2 is implemented right now: `scripts/infuse_sentinel2.py`.

---

## 1. Prerequisites

| What | Why |
|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Installs Python 3.11 and the dependencies |
| A Google account with Earth Engine access | Needed to query Sentinel-2 |
| A Google Cloud project registered for Earth Engine | Every Earth Engine request is billed/quota'd against a project. Register one (free for non-commercial/research use) at <https://code.earthengine.google.com/register> |

## 2. Install

From the project folder (`gee_inducer`):

```powershell
uv sync
```

This creates `.venv` and installs `earthengine-api`, `pandas`, `numpy`, `pyproj`, `pyyaml` and `python-dotenv`.

## 3. Authenticate with Earth Engine (one time)

```powershell
uv run earthengine authenticate
```

A browser window opens. Sign in with the Google account that has Earth Engine access. The credentials are saved on your machine, so you only do this once.

**Service account instead (optional, e.g. on a server):** skip the command above and set these in `.env`:

```
GEE_SERVICE_ACCOUNT=my-sa@my-project.iam.gserviceaccount.com
GEE_KEY_FILE=C:\path\to\key.json
```

## 4. Configure `.env`

`.env` must contain the two data folders plus your Earth Engine project ID:

```
RAW_DATASETS_DIR=F:\dev\organisations\penn_state_univ\gee_inducer\data\raw
GEE_INDUCER_DIR=F:\dev\organisations\penn_state_univ\gee_inducer\data\induced
GEE_PROJECT=your-cloud-project-id
```

| Variable | Meaning |
|---|---|
| `RAW_DATASETS_DIR` | Folder with the raw fix CSVs (e.g. `boar_reshaped.csv`) |
| `GEE_INDUCER_DIR` | Folder the infused CSVs are written to (created if missing) |
| `GEE_PROJECT` | Earth Engine Cloud project ID (can also be passed with `--project`) |

> `.env` and `ee-key.json` are in `.gitignore`. Keep key files out of git.

### Input CSV format

Each raw CSV needs at least these columns (extra columns are kept as-is):

| Column | Example |
|---|---|
| `timestamp` | `2019-01-01 00:00:00.000` (treated as UTC) |
| `lon` | `-97.2202015890482` |
| `lat` | `34.0214648266973` |

The script only processes CSVs that are listed under `datasets:` in `covariates/sources.yaml`. To add a new dataset, add an entry there and drop the CSV into `RAW_DATASETS_DIR`.

## 5. Run

Run each step from the project folder.

**a. Dry run (no Earth Engine calls).** Checks the CSVs and shows how many requests a run will make:

```powershell
uv run python scripts/infuse_sentinel2.py --dry-run
```

Example output:

```
[boar] 45079 fixes -> 35181 unique (date, pixel) samples over 335 dates, 335 requests
```

**b. Smoke test.** Fetches just 3 requests to confirm authentication and the project work:

```powershell
uv run python scripts/infuse_sentinel2.py --datasets boar --limit-requests 3 --allow-partial
```

Example dry-run output:

```
[black_bear] 156366 fixes -> 131463 unique (date, pixel) samples; 131223 reused from the earlier run, 240 to compute
[black_bear] 240 pixel-dates to composite -> 9 Earth Engine read requests (4-day blocks)
```

**c. Full run for one dataset:**

```powershell
uv run python scripts/infuse_sentinel2.py --datasets boar
```

**d. Every dataset whose CSV is present in `RAW_DATASETS_DIR`:**

```powershell
uv run python scripts/infuse_sentinel2.py
```

The output for each dataset is `GEE_INDUCER_DIR\<csv name>_sentinel2.csv`, e.g. `data\induced\boar_reshaped_sentinel2.csv`.

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--datasets NAME [NAME ...]` | all found | Dataset names from `sources.yaml` (`boar`, `mule_deer`, ...) |
| `--project ID` | `$GEE_PROJECT` | Earth Engine Cloud project |
| `--workers N` | 4 | Parallel Earth Engine requests. Lower it if you see quota / "too many requests" errors |
| `--points-per-request N` | 2000 | Max pixels read per request |
| `--block-days N` | 4 | Time block per read request. Smaller means fewer wasted scene reads but more requests |
| `--buffer` | off | Also compute the `_buf30` 30 m-radius means. Reads about 20–29× more pixels from Earth Engine |
| `--dry-run` | off | Read and plan only; shows how many requests a run needs |
| `--limit-requests N` | all | Only fetch the first N requests (testing) |
| `--refresh` | off | Refetch raw scene values even if cached |
| `--ignore-legacy-cache` | off | Don't reuse composites from the earlier server-side version |
| `--allow-partial` | off | Write the CSV even if some requests are still missing |
| `-v`, `--verbose` | off | Debug logging |

## 6. Output columns

The original columns are kept, then these are added:

| Column | Meaning |
|---|---|
| `s2_composite_date` | Fix date the composite is centred on (window = ±`composite_days`, 8 by default) |
| `s2_B2` … `s2_B12` | Median surface reflectance (0–1) of cloud-free scenes in the window |
| `s2_ndvi`, `s2_evi`, `s2_savi`, `s2_ndwi`, `s2_ndmi`, `s2_nbr` | Vegetation / water / burn indices computed on the composite |
| `s2_snow_fraction` | Share of cloud-free observations classified snow/ice (SCL 11) |
| `s2_ndvi_rate` | NDVI change per day versus the previous 16-day window (green-up proxy) |
| `s2_n_obs` | Number of cloud-free scenes in the window (0 = no data) |
| `*_buf30` | Mean of each value within a 30 m radius of the fix (only with `--buffer`) |

Empty values mean no cloud-free Sentinel-2 observation was available. This is common in cloudy seasons, and always the case for fixes before 2017, when the SR archive starts.

## 7. How it works

To keep Earth Engine usage (EECU quota) low, Earth Engine only **reads raw pixel values**. All calculation happens locally.

**On Earth Engine:** for each Sentinel-2 scene in a time block, it returns the raw DN of each band, the SCL class, and the s2cloudless probability at the requested 10 m pixels. That's `filterBounds`/`filterDate` plus a point read, and nothing else.

**Locally (pandas/numpy),** using the `sentinel2` block of `covariates/sources.yaml`:

1. Join cloud probability to each scene (on scene ID).
2. Scale DN to reflectance.
3. Mask SCL classes 0, 1, 3, 8, 9, 10 and cloud probability above 40%. Snow is kept.
4. Build the median composite over fix date ± 8 days, plus `n_obs` and `snow_fraction`.
5. Compute NDVI, EVI, SAVI, NDWI, NDMI and NBR from the composite.
6. Compute `ndvi_rate` against the composite centred 16 days earlier.
7. Compute the 30 m-radius means (`--buffer` only).

**Fetch plan:**

- Each fix is snapped to its 10 m UTM pixel.
- Time is cut into 4-day blocks.
- Each (pixel, block) pair is read **once**, however many fixes or dates use it.

**Caches:**

- Raw scene values: `GEE_INDUCER_DIR\.cache\sentinel2_scenes\<dataset>\`
- Finished composites from the earlier version of the script (`GEE_INDUCER_DIR\.cache\sentinel2\`) are reused, so that work is never re-requested.

## 8. Resuming and troubleshooting

- **Interrupted or some requests failed:** run the same command again. Cached requests are skipped, and the CSV is written once everything has succeeded.
- **Changed masking, index or window settings in `sources.yaml`:** composites are recomputed locally on every run, so raw scene reads don't need refetching. Do pass `--ignore-legacy-cache`, because the old server-side composites were built with the old settings. If you change `bands` or the collection, also pass `--refresh`.
- **`Please authorize access` / credential errors:** run `uv run earthengine authenticate` again.
- **`Project ... not registered` / permission denied:** check that `GEE_PROJECT` is registered for Earth Engine and that the Earth Engine API is enabled in that project.
- **Quota / `Too many concurrent aggregations` / timeouts:** these are retried automatically. If they persist, lower `--workers` (e.g. 2) or `--points-per-request` (e.g. 1000).
- **Monthly noncommercial EECU limit reached:** requests still run, just slower, until the reset on the 1st (Pacific time). Linking a billing account moves the project to the free Contributor tier (1,000 EECU-hours/month).
- **`not in sources.yaml datasets`:** add the dataset under `datasets:` in `covariates/sources.yaml`.
