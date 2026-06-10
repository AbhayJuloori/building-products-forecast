"""Create SKU, contractor, and branch demand segmentation artifacts."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path

LOCAL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "building_products_forecast_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.special import ndtri
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SALES_INPUT_PATH = RAW_DATA_DIR / "sales_history.parquet"
PRODUCTS_INPUT_PATH = RAW_DATA_DIR / "products.parquet"
BRANCHES_INPUT_PATH = RAW_DATA_DIR / "branches.parquet"
CONTRACTORS_INPUT_PATH = RAW_DATA_DIR / "contractors.parquet"

SKU_SEGMENTS_OUTPUT_PATH = PROCESSED_DATA_DIR / "sku_abc_xyz.parquet"
CONTRACTOR_SEGMENTS_OUTPUT_PATH = PROCESSED_DATA_DIR / "contractor_segments.parquet"
BRANCH_CLUSTERS_OUTPUT_PATH = PROCESSED_DATA_DIR / "branch_demand_clusters.parquet"

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "segmentation"
RANDOM_SEED = 42

ABC_CLASSES = ["A", "B", "C"]
XYZ_CLASSES = ["X", "Y", "Z"]
CONTRACTOR_CLUSTER_COUNT = 4
BRANCH_CLUSTER_COUNT = 3
KMEANS_K_RANGE = range(2, 7)
CONTRACTOR_FEATURE_COLUMNS = [
    "total_spend",
    "order_frequency",
    "avg_order_size",
    "product_breadth",
    "recency_days",
]

TIER_ORDER_RATE_RANGES = {
    "A": (80.0, 260.0),
    "B": (30.0, 80.0),
    "C": (5.0, 30.0),
}
TIER_AVG_ORDER_VALUE = {
    "A": 6500.0,
    "B": 2200.0,
    "C": 750.0,
}
TIER_ORDER_SIZE_SIGMA = {
    "A": 0.45,
    "B": 0.55,
    "C": 0.65,
}

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging for direct module execution."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _require(condition: bool, message: str) -> None:
    """Raise a clear validation error when an expected invariant is missing."""
    if not condition:
        raise ValueError(message)


def _require_columns(frame: pd.DataFrame, required_columns: set[str], frame_name: str) -> None:
    """Validate that a frame contains the required columns."""
    missing_columns = sorted(required_columns.difference(frame.columns))
    _require(not missing_columns, f"{frame_name} is missing required columns: {missing_columns}")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet artifact, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("Wrote %s with shape %s", path, frame.shape)


def _log_current_figure(artifact_path: Path) -> None:
    """Save the active matplotlib figure and log it as an MLflow artifact."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(artifact_path, dpi=160, bbox_inches="tight")
    mlflow.log_artifact(str(artifact_path))
    plt.close()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate all raw inputs used by segmentation."""
    for path in [SALES_INPUT_PATH, PRODUCTS_INPUT_PATH, BRANCHES_INPUT_PATH, CONTRACTORS_INPUT_PATH]:
        _require(path.exists(), f"Input file not found: {path}")

    sales = pd.read_parquet(
        SALES_INPUT_PATH,
        columns=["week_start_date", "branch_id", "sku_id", "units_sold", "revenue", "stockout_flag"],
    )
    products = pd.read_parquet(PRODUCTS_INPUT_PATH, columns=["sku_id", "category", "unit_cost"])
    branches = pd.read_parquet(BRANCHES_INPUT_PATH, columns=["branch_id", "region", "climate_zone"])
    contractors = pd.read_parquet(
        CONTRACTORS_INPUT_PATH,
        columns=["contractor_id", "branch_id", "annual_spend_tier"],
    )

    _require_columns(
        sales,
        {"week_start_date", "branch_id", "sku_id", "units_sold", "revenue", "stockout_flag"},
        "sales_history",
    )
    _require_columns(products, {"sku_id", "category", "unit_cost"}, "products")
    _require_columns(branches, {"branch_id", "region", "climate_zone"}, "branches")
    _require_columns(contractors, {"contractor_id", "branch_id", "annual_spend_tier"}, "contractors")

    sales = sales.copy()
    sales["week_start_date"] = pd.to_datetime(sales["week_start_date"])

    LOGGER.info(
        "Loaded raw inputs: sales=%s products=%s branches=%s contractors=%s",
        sales.shape,
        products.shape,
        branches.shape,
        contractors.shape,
    )
    return sales, products, branches, contractors


def _abc_class_from_revenue_share(revenue_share: pd.Series) -> pd.Series:
    """Assign ABC classes from descending revenue share using cumulative contribution."""
    prior_share = revenue_share.cumsum().shift(fill_value=0.0)
    return pd.Series(
        np.select(
            [prior_share < 0.70, prior_share < 0.90],
            ["A", "B"],
            default="C",
        ),
        index=revenue_share.index,
    )


def _xyz_class_from_cv(demand_cv: pd.Series) -> pd.Series:
    """Assign XYZ classes from coefficient of variation thresholds."""
    return pd.Series(
        np.select(
            [demand_cv < 0.5, demand_cv <= 1.0],
            ["X", "Y"],
            default="Z",
        ),
        index=demand_cv.index,
    )


def build_sku_abc_xyz(sales: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Build SKU-level ABC-XYZ classification output and matrix counts."""
    revenue_by_sku = (
        sales.groupby("sku_id", as_index=False, observed=True)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values(["total_revenue", "sku_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    total_revenue = float(revenue_by_sku["total_revenue"].sum())
    _require(total_revenue > 0.0, "Total SKU revenue must be positive")
    revenue_by_sku["revenue_share"] = revenue_by_sku["total_revenue"] / total_revenue
    revenue_by_sku["abc_class"] = _abc_class_from_revenue_share(revenue_by_sku["revenue_share"])

    non_stockout_sales = sales.loc[~sales["stockout_flag"], ["week_start_date", "sku_id", "units_sold"]]
    weekly_demand = non_stockout_sales.groupby(["sku_id", "week_start_date"], observed=True)["units_sold"].sum()
    demand_stats = weekly_demand.groupby("sku_id").agg(["mean", "std"]).rename(columns={"mean": "demand_mean", "std": "demand_std"})
    demand_cv = demand_stats["demand_std"].div(demand_stats["demand_mean"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    demand_cv = demand_cv.fillna(0.0).rename("demand_cv").reset_index()
    demand_cv["xyz_class"] = _xyz_class_from_cv(demand_cv["demand_cv"])

    sku_segments = (
        revenue_by_sku.merge(products[["sku_id", "category"]], on="sku_id", how="left")
        .merge(demand_cv, on="sku_id", how="left")
        .sort_values(["total_revenue", "sku_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    sku_segments["category"] = sku_segments["category"].fillna("unknown")
    sku_segments["demand_cv"] = sku_segments["demand_cv"].fillna(0.0)
    sku_segments["xyz_class"] = sku_segments["xyz_class"].fillna("X")

    output_columns = ["sku_id", "category", "total_revenue", "revenue_share", "abc_class", "demand_cv", "xyz_class"]
    sku_segments = sku_segments[output_columns]

    matrix = (
        pd.crosstab(sku_segments["abc_class"], sku_segments["xyz_class"])
        .reindex(index=ABC_CLASSES, columns=XYZ_CLASSES, fill_value=0)
        .astype(int)
    )
    matrix_counts = {
        abc_class: {xyz_class: int(matrix.loc[abc_class, xyz_class]) for xyz_class in XYZ_CLASSES}
        for abc_class in ABC_CLASSES
    }
    return sku_segments, matrix_counts


def plot_abc_xyz_heatmap(matrix_counts: dict[str, dict[str, int]], artifact_path: Path) -> None:
    """Plot ABC-XYZ matrix counts as a heatmap and log the PNG artifact."""
    matrix = pd.DataFrame(matrix_counts).T.reindex(index=ABC_CLASSES, columns=XYZ_CLASSES, fill_value=0)
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        image = ax.imshow(matrix.to_numpy(), cmap="viridis")
        ax.set_xticks(np.arange(len(XYZ_CLASSES)), labels=XYZ_CLASSES)
        ax.set_yticks(np.arange(len(ABC_CLASSES)), labels=ABC_CLASSES)
        ax.set_xlabel("XYZ class")
        ax.set_ylabel("ABC class")
        ax.set_title("SKU ABC-XYZ Matrix")
        for row_idx, abc_class in enumerate(ABC_CLASSES):
            for col_idx, xyz_class in enumerate(XYZ_CLASSES):
                ax.text(col_idx, row_idx, int(matrix.loc[abc_class, xyz_class]), ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, shrink=0.82, label="SKU count")
        _log_current_figure(artifact_path)


def _hash_unit_interval(values: pd.Series, salt: str) -> np.ndarray:
    """Return deterministic pseudo-random values in (0, 1) from string IDs."""
    salted_values = values.astype("string").fillna("<missing>") + f"|{salt}"
    hashed = pd.util.hash_pandas_object(salted_values, index=False).to_numpy(dtype=np.uint64)
    units = (hashed >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.clip(units, 1e-12, 1.0 - 1e-12)


def _tier_values(tiers: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    """Map annual spend tiers to numeric values with a safe default."""
    return tiers.map(mapping).fillna(default).astype("float64").to_numpy()


def simulate_contractor_features(
    contractors: pd.DataFrame,
    sales: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate deterministic contractor activity and aggregate it into modeling features."""
    contractor_features = contractors[["contractor_id", "branch_id", "annual_spend_tier"]].copy()
    contractor_features = contractor_features.sort_values("contractor_id", kind="mergesort").reset_index(drop=True)
    tiers = contractor_features["annual_spend_tier"].astype("string").str.upper().fillna("C")

    sales_dates = pd.DatetimeIndex(sales["week_start_date"].dropna().sort_values())
    _require(not sales_dates.empty, "Sales history has no valid week_start_date values")
    min_sales_date = sales_dates.min()
    max_sales_date = sales_dates.max()
    window_years = max(float(sales_dates.nunique()) / 52.0, 1.0)
    window_days = max((max_sales_date - min_sales_date).days, 1)

    tier_min_rates = _tier_values(tiers, {tier: bounds[0] for tier, bounds in TIER_ORDER_RATE_RANGES.items()}, 5.0)
    tier_max_rates = _tier_values(tiers, {tier: bounds[1] for tier, bounds in TIER_ORDER_RATE_RANGES.items()}, 30.0)
    rate_units = _hash_unit_interval(contractor_features["contractor_id"], "annual_order_rate")
    annual_order_rates = tier_min_rates + rate_units * (tier_max_rates - tier_min_rates)
    order_counts = np.maximum(np.rint(annual_order_rates * window_years).astype("int64"), 12)

    target_order_values = _tier_values(tiers, TIER_AVG_ORDER_VALUE, TIER_AVG_ORDER_VALUE["C"])
    order_sigmas = _tier_values(tiers, TIER_ORDER_SIZE_SIGMA, TIER_ORDER_SIZE_SIGMA["C"])
    order_mus = np.log(target_order_values) - 0.5 * np.square(order_sigmas)
    order_size_units = _hash_unit_interval(contractor_features["contractor_id"], "avg_order_size")
    avg_order_sizes = np.exp(order_mus + order_sigmas * ndtri(order_size_units))

    category_count = max(int(products["category"].nunique()), 1)
    c_breadth_units = _hash_unit_interval(contractor_features["contractor_id"], "category_breadth")
    product_breadth = np.select(
        [tiers.eq("A"), tiers.eq("B")],
        [min(3, category_count), min(2, category_count)],
        default=1 + (c_breadth_units > 0.35).astype("int64"),
    ).astype("int64")
    product_breadth = np.minimum(product_breadth, category_count)

    last_order_units = _hash_unit_interval(contractor_features["contractor_id"], "last_order_position")
    last_order_position = np.power(last_order_units, 1.0 / order_counts)
    recency_days = np.floor(window_days * (1.0 - last_order_position)).astype("int64")

    contractor_features["total_spend"] = order_counts * avg_order_sizes
    contractor_features["order_frequency"] = order_counts / window_years
    contractor_features["avg_order_size"] = avg_order_sizes
    contractor_features["product_breadth"] = product_breadth
    contractor_features["recency_days"] = recency_days

    return contractor_features[["contractor_id", "branch_id", *CONTRACTOR_FEATURE_COLUMNS]]


def _elbow_k(diagnostics: pd.DataFrame) -> int:
    """Estimate the elbow as the point farthest from the endpoint line."""
    points = diagnostics[["k", "inertia"]].to_numpy(dtype="float64")
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0.0:
        return int(points[0, 0])
    distances = np.abs(line[0] * (points[:, 1] - start[1]) - line[1] * (points[:, 0] - start[0])) / line_norm
    return int(points[int(np.argmax(distances)), 0])


def evaluate_kmeans_k(matrix: np.ndarray) -> pd.DataFrame:
    """Compute KMeans elbow and silhouette diagnostics for candidate k values."""
    diagnostics: list[dict[str, float]] = []
    for k in KMEANS_K_RANGE:
        model, labels = _fit_kmeans(matrix, k)
        diagnostics.append(
            {
                "k": float(k),
                "inertia": float(model.inertia_),
                "silhouette": _silhouette_score(matrix, labels),
            }
        )
    return pd.DataFrame.from_records(diagnostics)


def _silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for finite KMeans inputs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        score = silhouette_score(matrix, labels)
    return float(score)


def _fit_kmeans(matrix: np.ndarray, n_clusters: int) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans while suppressing a known finite-input sklearn matmul warning."""
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        labels = model.fit_predict(matrix)
    return model, labels


def _standardize_features(values: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Standardize finite feature values for KMeans clustering."""
    _require(np.isfinite(values).all(), "Contractor feature values contain non-finite values")
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        matrix = scaler.fit_transform(values)
    _require(np.isfinite(matrix).all(), "Contractor feature matrix contains non-finite values")
    return scaler, matrix


def _rank01(values: pd.Series, ascending: bool = True) -> pd.Series:
    """Return percentile-like ranks in [0, 1] for cluster scoring."""
    if len(values) == 1:
        return pd.Series(1.0, index=values.index)
    return values.rank(method="first", ascending=ascending).sub(1.0).div(len(values) - 1.0)


def label_contractor_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign meaningful contractor segment labels from original-scale centroids."""
    scored = centroids.set_index("cluster_id").copy()
    spend_rank = _rank01(scored["total_spend"])
    frequency_rank = _rank01(scored["order_frequency"])
    recency_rank = _rank01(scored["recency_days"])

    remaining = set(scored.index.astype(int))
    labels: dict[int, str] = {}

    high_value = int((spend_rank + frequency_rank).idxmax())
    labels[high_value] = "High-Value Regular"
    remaining.discard(high_value)

    if remaining:
        seasonal = int((spend_rank.loc[list(remaining)] - frequency_rank.loc[list(remaining)]).idxmax())
        labels[seasonal] = "Seasonal Bulk"
        remaining.discard(seasonal)

    if remaining:
        steady = int((frequency_rank.loc[list(remaining)] - spend_rank.loc[list(remaining)]).idxmax())
        labels[steady] = "Small Steady"
        remaining.discard(steady)

    if remaining:
        at_risk = int(((1.0 - spend_rank.loc[list(remaining)]) + (1.0 - recency_rank.loc[list(remaining)])).idxmax())
        labels[at_risk] = "At-Risk"
        remaining.discard(at_risk)

    for cluster_id in remaining:
        labels[int(cluster_id)] = "At-Risk"
    return labels


def build_contractor_segments(
    contractors: pd.DataFrame,
    sales: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build contractor feature segments with final k=4 KMeans labels."""
    contractor_features = simulate_contractor_features(contractors, sales, products)
    scaler, feature_matrix = _standardize_features(contractor_features[CONTRACTOR_FEATURE_COLUMNS].to_numpy(dtype="float64"))

    diagnostics = evaluate_kmeans_k(feature_matrix)
    final_model, cluster_ids = _fit_kmeans(feature_matrix, CONTRACTOR_CLUSTER_COUNT)

    centroids = pd.DataFrame(
        scaler.inverse_transform(final_model.cluster_centers_),
        columns=CONTRACTOR_FEATURE_COLUMNS,
    )
    centroids.insert(0, "cluster_id", np.arange(CONTRACTOR_CLUSTER_COUNT))
    label_map = label_contractor_clusters(centroids)

    contractor_segments = contractor_features.copy()
    contractor_segments["cluster_id"] = cluster_ids.astype("int64")
    contractor_segments["segment_label"] = contractor_segments["cluster_id"].map(label_map)
    contractor_segments = contractor_segments[
        [
            "contractor_id",
            "branch_id",
            "total_spend",
            "order_frequency",
            "avg_order_size",
            "product_breadth",
            "recency_days",
            "cluster_id",
            "segment_label",
        ]
    ].sort_values("contractor_id", kind="mergesort")

    return contractor_segments, diagnostics, centroids


def plot_kmeans_curve(diagnostics: pd.DataFrame, metric_column: str, title: str, ylabel: str, artifact_path: Path) -> None:
    """Plot one KMeans diagnostic curve and log the PNG artifact."""
    with plt.style.context("dark_background"):
        plt.figure(figsize=(6.8, 4.6))
        plt.plot(diagnostics["k"], diagnostics[metric_column], marker="o", linewidth=2)
        plt.xticks(list(KMEANS_K_RANGE))
        plt.xlabel("k")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        _log_current_figure(artifact_path)


def _fit_agglomerative_precomputed(distance_matrix: np.ndarray) -> np.ndarray:
    """Fit average-linkage agglomerative clustering with sklearn compatibility."""
    try:
        model = AgglomerativeClustering(
            n_clusters=BRANCH_CLUSTER_COUNT,
            linkage="average",
            metric="precomputed",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=BRANCH_CLUSTER_COUNT,
            linkage="average",
            affinity="precomputed",
        )
    return model.fit_predict(distance_matrix)


def _branch_cluster_label(cluster_series: pd.Series) -> str:
    """Label a branch cluster by quarterly demand shape."""
    quarter_means = cluster_series.groupby(cluster_series.index.quarter).mean().reindex([1, 2, 3, 4], fill_value=0.0)
    mean_demand = max(float(cluster_series.mean()), 1e-9)
    quarterly_spread = float(quarter_means.max() - quarter_means.min()) / mean_demand
    if quarterly_spread < 0.12:
        return "flat_demand"

    max_quarterly_demand = float(quarter_means.max())
    if quarter_means.loc[2] >= 0.90 * max_quarterly_demand and quarter_means.loc[3] >= 0.90 * max_quarterly_demand:
        return "spring_summer_peak"

    top_quarter = int(quarter_means.idxmax())
    if top_quarter == 2:
        return "spring_peak"
    if top_quarter == 3:
        return "summer_peak"
    if top_quarter == 4:
        return "fall_peak"
    return "winter_peak"


def build_branch_demand_clusters(sales: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Cluster branches by normalized weekly demand shape using DTW distance."""
    weekly_branch_demand = (
        sales.groupby(["branch_id", "week_start_date"], observed=True)["units_sold"]
        .sum()
        .reset_index()
    )
    demand_pivot = (
        weekly_branch_demand.pivot(index="branch_id", columns="week_start_date", values="units_sold")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )
    _require(len(demand_pivot) >= BRANCH_CLUSTER_COUNT, "Not enough branches for demand clustering")

    row_means = demand_pivot.mean(axis=1)
    row_stds = demand_pivot.std(axis=1, ddof=0).replace(0.0, 1.0)
    normalized = demand_pivot.sub(row_means, axis=0).div(row_stds, axis=0)
    normalized_matrix = np.ascontiguousarray(normalized.to_numpy(dtype=np.double))

    dtw_distance_matrix = np.asarray(
        dtw.distance_matrix_fast(normalized_matrix, compact=False, parallel=False),
        dtype="float64",
    )
    np.fill_diagonal(dtw_distance_matrix, 0.0)
    cluster_ids = _fit_agglomerative_precomputed(dtw_distance_matrix)

    cluster_labels: dict[int, str] = {}
    for cluster_id in sorted(np.unique(cluster_ids)):
        branch_ids = demand_pivot.index[cluster_ids == cluster_id]
        cluster_series = demand_pivot.loc[branch_ids].mean(axis=0)
        cluster_labels[int(cluster_id)] = _branch_cluster_label(cluster_series)

    branch_clusters = pd.DataFrame(
        {
            "branch_id": demand_pivot.index.to_numpy(),
            "cluster_id": cluster_ids.astype("int64"),
        }
    )
    branch_clusters["cluster_label"] = branch_clusters["cluster_id"].map(cluster_labels)
    branch_clusters = branch_clusters.sort_values("branch_id", kind="mergesort").reset_index(drop=True)
    return branch_clusters, dtw_distance_matrix, demand_pivot


def plot_branch_dendrogram(distance_matrix: np.ndarray, branch_ids: pd.Index, artifact_path: Path) -> None:
    """Plot average-linkage branch dendrogram from the DTW distance matrix."""
    condensed_distances = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distances, method="average")
    with plt.style.context("dark_background"):
        plt.figure(figsize=(8.0, 4.8))
        dendrogram(linkage_matrix, labels=branch_ids.to_list(), leaf_rotation=45)
        plt.title("Branch Demand DTW Dendrogram")
        plt.xlabel("Branch")
        plt.ylabel("DTW distance")
        _log_current_figure(artifact_path)


def run_sku_abc_xyz(sales: pd.DataFrame, products: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Run SKU ABC-XYZ classification, logging outputs and artifacts."""
    sku_segments, matrix_counts = build_sku_abc_xyz(sales, products)
    _write_parquet(sku_segments, SKU_SEGMENTS_OUTPUT_PATH)

    mlflow.log_param("sku_count", len(sku_segments))
    mlflow.log_param("abc_xyz_matrix", json.dumps(matrix_counts, sort_keys=True))
    mlflow.log_dict(matrix_counts, "abc_xyz_matrix.json")
    plot_abc_xyz_heatmap(matrix_counts, artifact_dir / "abc_xyz_heatmap.png")
    return sku_segments


def run_contractor_kmeans(
    contractors: pd.DataFrame,
    sales: pd.DataFrame,
    products: pd.DataFrame,
    artifact_dir: Path,
) -> pd.DataFrame:
    """Run contractor KMeans segmentation, logging diagnostics and artifacts."""
    contractor_segments, diagnostics, centroids = build_contractor_segments(contractors, sales, products)
    _write_parquet(contractor_segments, CONTRACTOR_SEGMENTS_OUTPUT_PATH)

    for row in diagnostics.itertuples(index=False):
        k = int(row.k)
        mlflow.log_metric(f"kmeans_elbow_k_{k}", float(row.inertia))
        mlflow.log_metric(f"kmeans_silhouette_k_{k}", float(row.silhouette))
    best_silhouette_k = int(diagnostics.loc[diagnostics["silhouette"].idxmax(), "k"])
    mlflow.log_param("elbow_k", _elbow_k(diagnostics))
    mlflow.log_param("best_silhouette_k", best_silhouette_k)
    mlflow.log_param("final_k", CONTRACTOR_CLUSTER_COUNT)
    mlflow.log_dict(
        {str(int(row.cluster_id)): row.drop(labels="cluster_id").to_dict() for _, row in centroids.iterrows()},
        "contractor_cluster_centroids.json",
    )

    plot_kmeans_curve(diagnostics, "inertia", "KMeans Elbow Curve", "Inertia", artifact_dir / "kmeans_elbow.png")
    plot_kmeans_curve(diagnostics, "silhouette", "KMeans Silhouette Curve", "Silhouette score", artifact_dir / "kmeans_silhouette.png")
    return contractor_segments


def run_branch_dtw(sales: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    """Run branch demand shape clustering with DTW distances."""
    branch_clusters, distance_matrix, demand_pivot = build_branch_demand_clusters(sales)
    _write_parquet(branch_clusters, BRANCH_CLUSTERS_OUTPUT_PATH)

    mlflow.log_param("branch_count", len(branch_clusters))
    mlflow.log_param("branch_cluster_count", BRANCH_CLUSTER_COUNT)
    mlflow.log_metric("mean_dtw_distance", float(distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean()))
    plot_branch_dendrogram(distance_matrix, demand_pivot.index, artifact_dir / "branch_dendrogram.png")
    return branch_clusters


def main() -> None:
    """Build all segmentation outputs and log MLflow artifacts."""
    _configure_logging()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    sales, products, branches, contractors = load_inputs()
    _require(set(sales["branch_id"]).issubset(set(branches["branch_id"])), "Sales contains branch IDs missing from branches")
    _require(set(contractors["branch_id"]).issubset(set(branches["branch_id"])), "Contractors contains branch IDs missing from branches")

    with tempfile.TemporaryDirectory(prefix="segmentation_artifacts_") as tmpdir:
        artifact_dir = Path(tmpdir)
        with mlflow.start_run(run_name="segmentation_pipeline"):
            with mlflow.start_run(run_name="sku_abc_xyz", nested=True):
                run_sku_abc_xyz(sales, products, artifact_dir)

            with mlflow.start_run(run_name="contractor_kmeans", nested=True):
                run_contractor_kmeans(contractors, sales, products, artifact_dir)

            with mlflow.start_run(run_name="branch_dtw", nested=True):
                run_branch_dtw(sales, artifact_dir)

    LOGGER.info("Segmentation pipeline completed")


if __name__ == "__main__":
    main()
