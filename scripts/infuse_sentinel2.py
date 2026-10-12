"""Infuse GPS fixes with Sentinel-2 covariates, doing all the maths locally.

Reads every raw fix CSV listed under ``datasets:`` in ``covariates/sources.yaml``
from ``RAW_DATASETS_DIR`` and writes ``<csv stem>_sentinel2.csv`` into
``GEE_INDUCER_DIR`` (both from ``.env``).

Earth Engine is only used to *read raw pixel values*: for each Sentinel-2
scene it returns the DN of every band, the SCL class and the s2cloudless
probability at the requested 10 m pixels. No compositing, masking, joins,
index maths or neighbourhood reducers run on Google's side. Everything
else runs on this machine with pandas/numpy, using the settings in the
``sentinel2`` block of ``sources.yaml``:

* scale factors (DN -> reflectance)
* QA masking (SCL classes + cloud probability threshold, snow kept)
* the joining of cloud probability to each scene (on ``system:index``)
* the median composite over ``fix date ± composite_days``
* ``snow_fraction``, ``n_obs``
* derived indices (NDVI, EVI, SAVI, NDWI, NDMI, NBR)
* ``ndvi_rate`` (vs. the composite centred one window earlier)
* optional ``*_buf30`` 30 m-radius means (``--buffer``)

Fetch plan: fixes are snapped to their 10 m UTM pixel. Each pixel only needs
the scenes inside the windows of the dates it was visited. Time is cut into
``--block-days`` blocks, and each (pixel, block) pair is read exactly once,
however many fixes/dates use it.

Raw scene values are cached under ``GEE_INDUCER_DIR/.cache/sentinel2_scenes``.
Finished composites from the previous (server-side) version of this script in
``GEE_INDUCER_DIR/.cache/sentinel2`` are reused as-is, so that work is not
re-requested.

Usage
-----
    uv run python scripts/infuse_sentinel2.py --dry-run
    uv run python scripts/infuse_sentinel2.py --datasets boar
    uv run python scripts/infuse_sentinel2.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pyproj import Transformer

log = logging.getLogger("infuse_sentinel2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_YAML = PROJECT_ROOT / "covariates" / "sources.yaml"
SOURCE_NAME = "sentinel2"
PREFIX = "s2_"
KEY_COLS = ["s2_date", "s2_epsg", "s2_ix", "s2_iy"]
PIX_COLS = ["s2_epsg", "s2_ix", "s2_iy"]
EPOCH = date(2015, 1, 1)  # block numbering origin (before Sentinel-2A launch)
DEFAULT_POINTS_PER_REQUEST = 2000
DEFAULT_BLOCK_DAYS = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S2Config:
    asset: str
    bands: list[str]
    scale_factors: dict[str, float]
    derived: list[dict]
    temporal_rates: list[dict]
    class_fractions: list[dict]
    composite_days: int
    res_m: float
    scl_band: str
    scl_classes: list[int]
    cloud_prob_asset: str | None
    cloud_prob_band: str | None
    cloud_prob_threshold: float | None
    buffer: bool
    buffer_radius_m: float
    footprint_pad_deg: float
    retry_max_attempts: int
    retry_backoff_seconds: float
    retry_jitter_seconds: float

    @property
    def fraction_names(self) -> list[str]:
        return [c["name"] for c in self.class_fractions]

    @property
    def value_bands(self) -> list[str]:
        return (list(self.bands) + [d["name"] for d in self.derived]
                + self.fraction_names + [r["name"] for r in self.temporal_rates])

    @property
    def buf_suffix(self) -> str:
        return f"_buf{int(self.buffer_radius_m)}"

    @property
    def output_bands(self) -> list[str]:
        names = self.value_bands + ["n_obs"]
        if self.buffer:
            names += [b + self.buf_suffix for b in self.value_bands]
        return names

    @property
    def lookback_days(self) -> int:
        """Days before the fix date whose scenes are needed."""
        cd = self.composite_days
        return 3 * cd if self.temporal_rates else cd


def load_config(buffer: bool) -> tuple[dict, S2Config]:
    with open(SOURCES_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    src = cfg["sources"][SOURCE_NAME]
    export = cfg.get("export", {})
    if src.get("cadence") != "composite":
        raise ValueError(f"{SOURCE_NAME}: expected cadence 'composite', got {src.get('cadence')!r}")
    grid = src.get("grid", {})
    if grid.get("mode") != "utm":
        raise ValueError(f"{SOURCE_NAME}: expected grid mode 'utm', got {grid.get('mode')!r}")
    qa = src.get("qa", {})
    if qa.get("kind") != "s2_scl":
        raise ValueError(f"{SOURCE_NAME}: unsupported qa kind {qa.get('kind')!r}")
    params = qa.get("params", {})
    cp = params.get("cloud_probability") or {}
    for rate in src.get("temporal_rates", []):
        if rate.get("lag_windows", 1) != 1:
            raise ValueError("only lag_windows: 1 is supported for temporal_rates")
    scl_bands = {c["band"] for c in src.get("class_fractions", [])} | {"SCL"}
    if len(scl_bands) != 1:
        raise ValueError("class_fractions must all use the SCL band")
    return cfg, S2Config(
        asset=src["asset"],
        bands=list(src["bands"]),
        scale_factors={k: float(v) for k, v in (src.get("scale_factors") or {}).items()},
        derived=list(src.get("derived") or []),
        temporal_rates=list(src.get("temporal_rates") or []),
        class_fractions=list(src.get("class_fractions") or []),
        composite_days=int(src["composite_days"]),
        res_m=float(grid.get("res", [10.0, 10.0])[0]),
        scl_band="SCL",
        scl_classes=[int(c) for c in params.get("scl_classes", [])],
        cloud_prob_asset=cp.get("asset"),
        cloud_prob_band=cp.get("band"),
        cloud_prob_threshold=cp.get("threshold"),
        buffer=buffer,
        buffer_radius_m=float(export.get("buffer_radius_m", 30)),
        footprint_pad_deg=float(export.get("footprint_pad_deg", 0.02)),
        retry_max_attempts=int(export.get("retry_max_attempts", 5)),
        retry_backoff_seconds=float(export.get("retry_backoff_seconds", 5)),
        retry_jitter_seconds=float(export.get("retry_jitter_seconds", 3)),
    )


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (expected in {PROJECT_ROOT / '.env'})")
    return Path(value)


# ---------------------------------------------------------------------------
# Reading fixes, pixel keys, fetch plan
# ---------------------------------------------------------------------------

def read_fixes(path: Path, duplicate_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in duplicate_columns:
        dupes = [c for c in df.columns if c.startswith(f"{col}.") and c[len(col) + 1:].isdigit()]
        for dup in dupes:
            if not df[col].astype(str).equals(df[dup].astype(str)):
                raise ValueError(f"{path.name}: duplicate column {col!r} disagrees with {dup!r}")
            df = df.drop(columns=dup)
    missing = {"timestamp", "lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
    return df


@lru_cache(maxsize=None)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def add_keys(df: pd.DataFrame, res_m: float) -> pd.DataFrame:
    """(fix date, UTM EPSG, pixel column, pixel row) for every fix."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    valid = ts.notna() & lon.between(-180, 180) & lat.between(-80, 84)
    if (~valid).any():
        log.warning("  %d fixes have an unparseable timestamp or out-of-range coordinates; "
                    "they will have empty Sentinel-2 columns", int((~valid).sum()))
    zone = np.floor((lon + 180.0) / 6.0).clip(0, 59) + 1
    epsg = np.where(lat >= 0, 32600, 32700) + zone
    keys = pd.DataFrame(index=df.index)
    keys["s2_date"] = ts.dt.strftime("%Y-%m-%d").where(valid)
    keys["s2_epsg"] = pd.array(np.where(valid, epsg, np.nan), dtype="Int64")
    keys["s2_ix"] = pd.array([pd.NA] * len(df), dtype="Int64")
    keys["s2_iy"] = pd.array([pd.NA] * len(df), dtype="Int64")
    for code in keys.loc[valid, "s2_epsg"].dropna().unique():
        sel = valid & (keys["s2_epsg"] == code)
        x, y = _transformer("EPSG:4326", f"EPSG:{int(code)}").transform(
            lon[sel].to_numpy(), lat[sel].to_numpy())
        keys.loc[sel, "s2_ix"] = np.floor(x / res_m).astype(np.int64)
        keys.loc[sel, "s2_iy"] = np.floor(y / res_m).astype(np.int64)
    return keys


def pixel_centers(epsg: int, ix: np.ndarray, iy: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _transformer(f"EPSG:{epsg}", "EPSG:4326").transform((ix + 0.5) * res_m, (iy + 0.5) * res_m)
    return np.round(lon, 8), np.round(lat, 8)


def buffer_offsets(cfg: S2Config) -> list[tuple[int, int]]:
    """Pixel offsets inside a circle of buffer_radius_m (matches ee.Kernel.circle)."""
    r = cfg.buffer_radius_m / cfg.res_m
    n = int(math.floor(r))
    return [(dx, dy) for dx in range(-n, n + 1) for dy in range(-n, n + 1) if dx * dx + dy * dy <= r * r]


def expand_with_neighbours(keys: pd.DataFrame, cfg: S2Config) -> pd.DataFrame:
    parts = []
    for dx, dy in buffer_offsets(cfg):
        p = keys.copy()
        p["s2_ix"] = p["s2_ix"] + dx
        p["s2_iy"] = p["s2_iy"] + dy
        parts.append(p)
    return pd.concat(parts, ignore_index=True).drop_duplicates(KEY_COLS)


@dataclass(frozen=True)
class Request:
    epsg: int
    start: date   # inclusive
    end: date     # exclusive
    points: pd.DataFrame  # s2_ix, s2_iy, lon, lat

    @property
    def cache_name(self) -> str:
        h = hashlib.sha1(self.points[["s2_ix", "s2_iy"]].to_numpy(np.int64).tobytes()).hexdigest()[:10]
        return f"{self.epsg}_{self.start.isoformat()}_{h}.csv.gz"


def plan_requests(need: pd.DataFrame, cfg: S2Config, block_days: int, points_per_request: int) -> list[Request]:
    """Every (pixel, time block) that some needed composite window touches, read once."""
    if need.empty:
        return []
    d = pd.to_datetime(need["s2_date"])
    epoch = pd.Timestamp(EPOCH)
    b_lo = ((d - pd.Timedelta(days=cfg.lookback_days)) - epoch).dt.days // block_days
    b_hi = ((d + pd.Timedelta(days=cfg.composite_days)) - epoch).dt.days // block_days
    n = (b_hi - b_lo + 1).to_numpy()
    rep = need.loc[need.index.repeat(n), PIX_COLS].reset_index(drop=True)
    start = np.repeat(b_lo.to_numpy(), n)
    offset = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
    rep["block"] = start + offset
    pairs = rep.drop_duplicates().sort_values(["s2_epsg", "block", "s2_ix", "s2_iy"])

    requests: list[Request] = []
    for (epsg, block), grp in pairs.groupby(["s2_epsg", "block"], sort=True):
        epsg, block = int(epsg), int(block)
        ix = grp["s2_ix"].to_numpy(np.int64)
        iy = grp["s2_iy"].to_numpy(np.int64)
        lon, lat = pixel_centers(epsg, ix, iy, cfg.res_m)
        pts = pd.DataFrame({"s2_ix": ix, "s2_iy": iy, "lon": lon, "lat": lat})
        b_start = EPOCH + timedelta(days=block * block_days)
        for s in range(0, len(pts), points_per_request):
            requests.append(Request(epsg, b_start, b_start + timedelta(days=block_days),
                                    pts.iloc[s:s + points_per_request].reset_index(drop=True)))
    return requests


# ---------------------------------------------------------------------------
# Earth Engine: raw pixel reads only
# ---------------------------------------------------------------------------

def init_ee(project: str | None):
    import ee

    url = "https://earthengine-highvolume.googleapis.com"
    sa, key = os.environ.get("GEE_SERVICE_ACCOUNT"), os.environ.get("GEE_KEY_FILE")
    if sa and key:
        ee.Initialize(ee.ServiceAccountCredentials(sa, key), project=project, opt_url=url)
    else:
        ee.Initialize(project=project, opt_url=url)
    return ee


class SceneReader:
    """Returns one row per (scene, pixel) with raw DN, SCL and cloud probability."""

    def __init__(self, ee, cfg: S2Config):
        self.ee = ee
        self.cfg = cfg

    def _extract(self, asset, bands, props, region, fc, crs, transform, start, end):
        ee = self.ee
        # Single-band images come back under the reducer's name, not the band's.
        test = bands[0] if len(bands) > 1 else "first"

        def per_scene(img):
            img = ee.Image(img)
            sid = img.get("system:index")
            return (img.select(bands)
                    .reduceRegions(collection=fc, reducer=ee.Reducer.first(), crs=crs, crsTransform=transform)
                    .filter(ee.Filter.notNull([test]))
                    .map(lambda f: f.set("sid", sid)))

        coll = ee.ImageCollection(asset).filterBounds(region).filterDate(start, end)
        rows = ee.FeatureCollection(coll.map(per_scene)).flatten()
        src = ["k", "sid"] + (bands if len(bands) > 1 else ["first"])
        return rows.select(src, ["k", "sid"] + props, False)

    def read(self, req: Request) -> pd.DataFrame:
        ee, cfg = self.ee, self.cfg
        pts, pad = req.points, cfg.footprint_pad_deg
        region = ee.Geometry.Rectangle([float(pts.lon.min()) - pad, float(pts.lat.min()) - pad,
                                        float(pts.lon.max()) + pad, float(pts.lat.max()) + pad])
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"k": int(i)})
                                   for i, (lo, la) in enumerate(zip(pts.lon, pts.lat))])
        crs = f"EPSG:{req.epsg}"
        transform = [cfg.res_m, 0, 0, 0, -cfg.res_m, 0]
        start, end = req.start.isoformat(), req.end.isoformat()

        s2_bands = cfg.bands + [cfg.scl_band]
        rows = self._extract(cfg.asset, s2_bands, s2_bands, region, fc, crs, transform, start, end)
        if cfg.cloud_prob_asset:
            rows = rows.merge(self._extract(cfg.cloud_prob_asset, [cfg.cloud_prob_band], ["cloud_prob"],
                                            region, fc, crs, transform, start, end))

        df = ee.data.computeFeatures({"expression": rows, "fileFormat": "PANDAS_DATAFRAME"})
        return tidy_scene_rows(df, req, cfg)


def tidy_scene_rows(df: pd.DataFrame, req: Request, cfg: S2Config) -> pd.DataFrame:
    """Local join of cloud probability onto the S2 rows (what ee.Join.saveFirst did)."""
    s2_bands = cfg.bands + [cfg.scl_band]
    df = df.reindex(columns=["k", "sid"] + s2_bands + ["cloud_prob"])
    s2 = df[df[cfg.bands[0]].notna()].drop(columns="cloud_prob")
    if cfg.cloud_prob_asset:
        cp = df.loc[df["cloud_prob"].notna(), ["k", "sid", "cloud_prob"]].drop_duplicates(["k", "sid"])
        s2 = s2.merge(cp, on=["k", "sid"], how="inner")  # scenes without a cloud mask are dropped
    else:
        s2["cloud_prob"] = 0.0
    s2 = s2.drop_duplicates(["k", "sid"])
    k = s2["k"].astype(int).to_numpy()
    out = pd.DataFrame({
        "s2_epsg": req.epsg,
        "s2_ix": req.points["s2_ix"].to_numpy()[k],
        "s2_iy": req.points["s2_iy"].to_numpy()[k],
        "sid": s2["sid"].to_numpy(),
    })
    for b in s2_bands + ["cloud_prob"]:
        out[b] = s2[b].to_numpy()
    return out


def run_with_retry(fn, req: Request, cfg: S2Config):
    for attempt in range(1, cfg.retry_max_attempts + 1):
        try:
            return fn(req)
        except Exception as exc:  # EE surfaces quota/timeouts as generic errors
            if attempt == cfg.retry_max_attempts:
                raise
            wait = cfg.retry_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, cfg.retry_jitter_seconds)
            log.warning("  %s attempt %d failed (%s); retrying in %.0fs",
                        req.cache_name, attempt, str(exc).splitlines()[0][:200], wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Local computation
# ---------------------------------------------------------------------------

_BAND_REF = re.compile(r"b\('([^']+)'\)")
_EXPR_NS = {"__builtins__": {}, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs}


def eval_expr(expr: str, frame: pd.DataFrame) -> pd.Series:
    """Evaluate an ee.Image.expression-style formula (b('B8') ...) from sources.yaml."""
    py = _BAND_REF.sub(lambda m: f"c[{m.group(1)!r}]", expr)
    with np.errstate(all="ignore"):
        out = eval(py, _EXPR_NS, {"c": frame})  # expressions come from the local sources.yaml
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_scenes(raw: pd.DataFrame, cfg: S2Config) -> pd.DataFrame:
    """Scale, QA-mask and tag snow. Masked observations are dropped (they count for nothing)."""
    scl = raw[cfg.scl_band].round().astype("Int64")
    keep = ~scl.isin(cfg.scl_classes) & scl.notna()
    if cfg.cloud_prob_asset:
        keep &= raw["cloud_prob"] <= cfg.cloud_prob_threshold
    kept = raw[keep.fillna(False).to_numpy()]
    out = kept[PIX_COLS].copy()
    out["day"] = pd.to_datetime(kept["sid"].str[:8], format="%Y%m%d")
    for b in cfg.bands:
        out[b] = kept[b].astype(float) * cfg.scale_factors.get(b, 1.0)
    for frac in cfg.class_fractions:
        out[frac["name"]] = scl[keep.fillna(False)].isin(frac["classes"]).astype(float).to_numpy()
    return out.reset_index(drop=True)


def _median_composite(rows: pd.DataFrame, keys: list[str], cfg: S2Config) -> pd.DataFrame:
    g = rows.groupby(keys, sort=False)
    comp = g[cfg.bands].median()
    for d in cfg.derived:
        comp[d["name"]] = eval_expr(d["expr"], comp)
    return comp, g


def compute_composites(need: pd.DataFrame, scenes: pd.DataFrame, cfg: S2Config,
                       chunk_keys: int = 20000) -> pd.DataFrame:
    """Median composite (+ indices, snow fraction, n_obs, rates) for each needed (date, pixel)."""
    cd = cfg.composite_days
    keys = need[KEY_COLS].copy()
    keys["date"] = pd.to_datetime(keys["s2_date"])
    keys = keys.sort_values(PIX_COLS + ["date"]).reset_index(drop=True)
    kc = ["s2_date"] + PIX_COLS
    out = []
    for s in range(0, len(keys), chunk_keys):
        chunk = keys.iloc[s:s + chunk_keys]
        m = chunk.merge(scenes, on=PIX_COLS, how="inner")
        dd = (m["day"] - m["date"]).dt.days
        cur = m[(dd >= -cd) & (dd <= cd)]
        comp, g = _median_composite(cur, kc, cfg)
        for name in cfg.fraction_names:
            comp[name] = g[name].mean()
        comp["n_obs"] = g.size()
        if cfg.temporal_rates:
            window = 2 * cd
            prev, _ = _median_composite(m[(dd >= -3 * cd) & (dd <= -cd)], kc, cfg)
            prev = prev.reindex(comp.index)
            for r in cfg.temporal_rates:
                comp[r["name"]] = (comp[r["band"]] - prev[r["band"]]) / window
        idx = pd.MultiIndex.from_frame(chunk[kc])
        out.append(comp.reindex(idx))
    if not out:
        return pd.DataFrame(columns=cfg.value_bands + ["n_obs"],
                            index=pd.MultiIndex.from_arrays([[]] * 4, names=kc))
    res = pd.concat(out)
    res["n_obs"] = res["n_obs"].fillna(0).astype(int)
    return res[cfg.value_bands + ["n_obs"]]


def add_buffer_means(comp: pd.DataFrame, targets: pd.DataFrame, cfg: S2Config) -> pd.DataFrame:
    """30 m-radius mean around each target pixel; null where the centre pixel is null
    (same behaviour as reduceNeighborhood with skipMasked)."""
    kc = ["s2_date"] + PIX_COLS
    vals = comp[cfg.value_bands].astype(float)
    base = targets[kc].reset_index(drop=True)
    total = np.zeros((len(base), len(cfg.value_bands)))
    count = np.zeros_like(total)
    for dx, dy in buffer_offsets(cfg):
        look = base.assign(s2_ix=base["s2_ix"] + dx, s2_iy=base["s2_iy"] + dy)
        v = vals.reindex(pd.MultiIndex.from_frame(look)).to_numpy()
        ok = ~np.isnan(v)
        total += np.where(ok, v, 0.0)
        count += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    centre = vals.reindex(pd.MultiIndex.from_frame(base)).to_numpy()
    mean[np.isnan(centre)] = np.nan
    idx = pd.MultiIndex.from_frame(base)
    return pd.DataFrame(mean, index=idx, columns=[b + cfg.buf_suffix for b in cfg.value_bands])


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

def load_legacy(cache_dir: Path, cfg: S2Config) -> pd.DataFrame:
    """Finished composites written by the earlier server-side version of this script."""
    files = sorted(cache_dir.glob("*.csv")) if cache_dir.is_dir() else []
    if not files:
        return pd.DataFrame(columns=KEY_COLS)
    df = pd.concat([pd.read_csv(f, dtype={"s2_date": str}) for f in files], ignore_index=True)
    if cfg.buffer and not all(c in df.columns for c in cfg.output_bands):
        log.warning("  legacy cache lacks buffer columns; not reusing it")
        return pd.DataFrame(columns=KEY_COLS)
    for c in PIX_COLS:
        df[c] = df[c].astype(np.int64)
    return df[KEY_COLS + cfg.output_bands].drop_duplicates(KEY_COLS)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_dataset(name: str, entry: dict, raw_dir: Path, out_dir: Path, cfg: S2Config,
                    args, reader: SceneReader | None) -> bool:
    csv_path = raw_dir / entry["csv"]
    log.info("[%s] reading %s", name, csv_path)
    fixes = read_fixes(csv_path, entry.get("duplicate_columns", []))
    keys = add_keys(fixes, cfg.res_m)
    uniq = keys.dropna(subset=["s2_date", "s2_epsg"]).drop_duplicates(KEY_COLS).astype(
        {"s2_epsg": np.int64, "s2_ix": np.int64, "s2_iy": np.int64}).reset_index(drop=True)

    legacy = pd.DataFrame(columns=KEY_COLS)
    if not args.ignore_legacy_cache:
        legacy = load_legacy(out_dir / ".cache" / SOURCE_NAME / name, cfg)
    todo = uniq.merge(legacy[KEY_COLS], on=KEY_COLS, how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)

    need = expand_with_neighbours(todo, cfg) if cfg.buffer else todo
    requests = plan_requests(need, cfg, args.block_days, args.points_per_request)
    log.info("[%s] %d fixes -> %d unique (date, pixel) samples; %d reused from the earlier run, %d to compute",
             name, len(fixes), len(uniq), len(uniq) - len(todo), len(todo))
    log.info("[%s] %d pixel-dates to composite%s -> %d Earth Engine read requests (%d-day blocks)",
             name, len(need), " (incl. 30 m neighbours)" if cfg.buffer else "", len(requests), args.block_days)
    if args.dry_run:
        return True

    cache_dir = out_dir / ".cache" / f"{SOURCE_NAME}_scenes" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetch = requests[: args.limit_requests] if args.limit_requests else requests
    pending = [r for r in fetch if args.refresh or not (cache_dir / r.cache_name).exists()]
    log.info("[%s] %d requests cached, %d to fetch with %d workers",
             name, len(fetch) - len(pending), len(pending), args.workers)

    failed = 0
    if pending:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_with_retry, reader.read, r, cfg): r for r in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                req = futures[fut]
                try:
                    df = fut.result()
                    tmp = cache_dir / (req.cache_name + ".tmp")
                    df.to_csv(tmp, index=False, compression="gzip")
                    tmp.replace(cache_dir / req.cache_name)
                except Exception as exc:
                    failed += 1
                    log.error("  %s failed: %s", req.cache_name, str(exc).splitlines()[0][:300])
                if i % 25 == 0 or i == len(pending):
                    rate = i / max(time.monotonic() - started, 1e-6)
                    log.info("[%s] %d/%d requests (%.1f/s, ~%.0f min left)",
                             name, i, len(pending), rate, (len(pending) - i) / rate / 60)

    missing = [r for r in requests if not (cache_dir / r.cache_name).exists()]
    if missing and not args.allow_partial:
        log.error("[%s] %d of %d requests are not fetched yet%s; rerun to continue "
                  "(fetched ones are cached) or pass --allow-partial", name, len(missing),
                  len(requests), f" ({failed} failed)" if failed else "")
        return False

    log.info("[%s] computing composites locally", name)
    raw = [pd.read_csv(cache_dir / r.cache_name) for r in requests if (cache_dir / r.cache_name).exists()]
    raw = [r for r in raw if not r.empty]
    raw = (pd.concat(raw, ignore_index=True).drop_duplicates(PIX_COLS + ["sid"]) if raw
           else pd.DataFrame(columns=PIX_COLS + ["sid", cfg.scl_band, "cloud_prob"] + cfg.bands))
    scenes = prepare_scenes(raw, cfg)
    comp = compute_composites(need, scenes, cfg)
    todo_idx = pd.MultiIndex.from_frame(todo[KEY_COLS])
    fresh = comp.reindex(todo_idx)
    if cfg.buffer:
        fresh = fresh.join(add_buffer_means(comp, todo, cfg))
    fresh = fresh.reset_index()

    samples = pd.concat([legacy, fresh[KEY_COLS + cfg.output_bands]], ignore_index=True)
    for col in PIX_COLS:
        samples[col] = samples[col].astype("Int64")
    samples = samples.drop_duplicates(KEY_COLS).rename(columns={b: PREFIX + b for b in cfg.output_bands})

    merged = pd.concat([fixes, keys[KEY_COLS]], axis=1).merge(samples, on=KEY_COLS, how="left")
    merged = merged.drop(columns=PIX_COLS).rename(columns={"s2_date": "s2_composite_date"})
    assert len(merged) == len(fixes), "join changed the row count"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(entry['csv']).stem}_{SOURCE_NAME}.csv"
    merged.to_csv(out_path, index=False)
    covered = merged[PREFIX + cfg.bands[0]].notna().mean() * 100 if len(merged) else 0
    log.info("[%s] wrote %s (%d rows, %.1f%% with a cloud-free S2 value)", name, out_path, len(merged), covered)
    return not missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", help="dataset names from sources.yaml (default: all found in RAW_DATASETS_DIR)")
    parser.add_argument("--project", help="Earth Engine Cloud project (default: $GEE_PROJECT)")
    parser.add_argument("--workers", type=int, default=4, help="parallel Earth Engine requests (default 4)")
    parser.add_argument("--points-per-request", type=int, default=DEFAULT_POINTS_PER_REQUEST)
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS,
                        help="time block per read request; smaller = fewer wasted scene reads, more requests")
    parser.add_argument("--buffer", action="store_true",
                        help="also compute *_buf30 30 m-radius means (reads ~29x more pixels from Earth Engine)")
    parser.add_argument("--dry-run", action="store_true", help="read and plan only; no Earth Engine calls")
    parser.add_argument("--limit-requests", type=int, help="only fetch the first N requests (for testing)")
    parser.add_argument("--refresh", action="store_true", help="refetch raw scene values even if cached")
    parser.add_argument("--ignore-legacy-cache", action="store_true",
                        help="do not reuse composites from the earlier server-side version")
    parser.add_argument("--allow-partial", action="store_true", help="write the CSV even if some requests are missing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    load_dotenv(PROJECT_ROOT / ".env")
    raw_dir, out_dir = env_path("RAW_DATASETS_DIR"), env_path("GEE_INDUCER_DIR")
    full_cfg, cfg = load_config(args.buffer)
    registry: dict = full_cfg.get("datasets", {})

    if args.datasets:
        unknown = [d for d in args.datasets if d not in registry]
        if unknown:
            raise SystemExit(f"not in sources.yaml datasets: {', '.join(unknown)}")
        names = args.datasets
        missing = [d for d in names if not (raw_dir / registry[d]["csv"]).exists()]
        if missing:
            raise SystemExit(f"CSV not found in {raw_dir}: {', '.join(registry[d]['csv'] for d in missing)}")
    else:
        names = [d for d, e in registry.items() if (raw_dir / e["csv"]).exists()]
        skipped = [d for d in registry if d not in names]
        if skipped:
            log.info("no raw CSV yet for: %s", ", ".join(skipped))
        if not names:
            raise SystemExit(f"no dataset CSVs from sources.yaml found in {raw_dir}")

    reader = None
    if not args.dry_run:
        reader = SceneReader(init_ee(args.project or os.environ.get("GEE_PROJECT")), cfg)

    ok = True
    for name in names:
        ok &= process_dataset(name, registry[name], raw_dir, out_dir, cfg, args, reader)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
