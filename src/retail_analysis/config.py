"""Load config.yaml into typed dataclasses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Project directories and output files, resolved against the project root."""

    raw_dir: Path
    processed_dir: Path
    sql_dir: Path
    prepare_stats_file: Path
    quality_file: Path
    figures_dir: Path
    results_file: Path


@dataclass(frozen=True)
class SourceFile:
    """One downloadable raw file."""

    url: str
    sha256: str


@dataclass(frozen=True)
class Source:
    """Where the raw data comes from."""

    name: str
    url: str
    files: dict[str, SourceFile]


@dataclass(frozen=True)
class PrepareParams:
    """Spark settings and raw code conventions."""

    spark_driver_memory: str
    mailer_absent_code: str
    display_absent_code: str


@dataclass(frozen=True)
class QualityParams:
    """Data quality checks and analysis-window rules."""

    key_columns: dict[str, list[str]]
    hierarchy_columns: list[str]
    min_week: int
    max_week: int
    stable_activity_ratio: float
    full_week_days: int


@dataclass(frozen=True)
class AnalysisParams:
    """General analysis settings."""

    top_categories: int
    trend_weeks: int


@dataclass(frozen=True)
class SegmentRule:
    """Inclusive RFM score ranges that define one segment."""

    name: str
    recency: tuple[int, int]
    frequency: tuple[int, int]
    monetary: tuple[int, int]


@dataclass(frozen=True)
class RfmParams:
    """RFM scoring and segment rules."""

    score_buckets: int
    top_customers_share: float
    segments: list[SegmentRule]


@dataclass(frozen=True)
class CohortParams:
    """Cohort retention settings."""

    period_weeks: int
    min_cohort_size: int
    report_periods: list[int]


@dataclass(frozen=True)
class ChurningLoyalParams:
    """Definition of churning loyal households."""

    loyal_top_share: float
    recent_weeks: int
    activity_drop: float
    top_departments: int
    alpha: float
    min_expected_count: float


@dataclass(frozen=True)
class PromoParams:
    """Difference-in-differences design for mailer placements."""

    pre_weeks: int
    treated_min_store_share: float
    min_pre_weeks_with_sales: int
    excluded_departments: list[str]
    confidence_level: float


@dataclass(frozen=True)
class PlotParams:
    """Shared chart settings."""

    dpi: int
    palette: list[str]


@dataclass(frozen=True)
class Config:
    """Full project configuration."""

    paths: Paths
    source: Source
    prepare: PrepareParams
    quality: QualityParams
    analysis: AnalysisParams
    rfm: RfmParams
    cohorts: CohortParams
    churning_loyal: ChurningLoyalParams
    promo: PromoParams
    plots: PlotParams


def _parse_paths(raw: dict[str, str], root: Path) -> Paths:
    return Paths(**{key: root / value for key, value in raw.items()})


def _parse_source(raw: dict[str, Any]) -> Source:
    files = {name: SourceFile(**spec) for name, spec in raw["files"].items()}
    return Source(name=raw["name"], url=raw["url"], files=files)


def _parse_rfm(raw: dict[str, Any]) -> RfmParams:
    segments = [
        SegmentRule(
            name=rule["name"],
            recency=tuple(rule["recency"]),
            frequency=tuple(rule["frequency"]),
            monetary=tuple(rule["monetary"]),
        )
        for rule in raw["segments"]
    ]
    return RfmParams(
        score_buckets=raw["score_buckets"],
        top_customers_share=raw["top_customers_share"],
        segments=segments,
    )


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read the YAML configuration file.

    Args:
        path: Path to config.yaml.

    Returns:
        Parsed configuration with paths resolved against the config file's directory.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(
        paths=_parse_paths(raw["paths"], path.parent),
        source=_parse_source(raw["source"]),
        prepare=PrepareParams(**raw["prepare"]),
        quality=QualityParams(**raw["quality"]),
        analysis=AnalysisParams(**raw["analysis"]),
        rfm=_parse_rfm(raw["rfm"]),
        cohorts=CohortParams(**raw["cohorts"]),
        churning_loyal=ChurningLoyalParams(**raw["churning_loyal"]),
        promo=PromoParams(**raw["promo"]),
        plots=PlotParams(**raw["plots"]),
    )
