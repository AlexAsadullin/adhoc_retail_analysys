"""Difference-in-differences estimate of mailer placements and cannibalisation.

The design is a stacked event study on the product-week panel. An event is a promo week:
for every week `w0` the panel holds the `pre_weeks` weeks before it and `w0` itself.

* Promo effect: treated products enter the mailer in `w0` after `pre_weeks` weeks without
  it; controls are products of the same sub-commodity with no mailer in the whole window.
* Cannibalisation: exposed products are never in the mailer themselves, but a product of
  their sub-commodity enters the mailer in `w0` (and no product of it was in the mailer in
  the pre-period); controls are products of other sub-commodities of the same commodity
  with no mailer at all in the window.

Product and week fixed effects are event specific and absorbed by within-event two-way
demeaning, which is exact because every event panel is balanced. Standard errors are
clustered by product.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from retail_analysis.config import PromoParams

EVENT_COLUMNS = ["event_id", "product_id", "promo_week", "treated"]


@dataclass(frozen=True)
class WideTables:
    """Product x week matrices and the product hierarchy."""

    mailer_share: pd.DataFrame
    display_share: pd.DataFrame
    units: pd.DataFrame
    revenue: pd.DataFrame
    hierarchy: pd.DataFrame


@dataclass(frozen=True)
class DidResult:
    """Difference-in-differences estimate on the log(1 + outcome) scale."""

    outcome: str
    coefficient: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_pct: float
    effect_pct_ci_low: float
    effect_pct_ci_high: float
    observations: int
    events: int
    treated_units: int
    control_units: int
    clusters: int


def load_product_weeks(processed_dir: Path, excluded_departments: list[str]) -> pd.DataFrame:
    """Read the product-week promo table built by `prepare`.

    Args:
        processed_dir: Directory with processed Parquet tables.
        excluded_departments: Departments where quantity is not an item count.

    Returns:
        Product-week rows of products with a known, non-excluded department.
    """
    frame = pd.read_parquet(processed_dir / "promo_product_week")
    keep = frame["department"].notna() & ~frame["department"].isin(excluded_departments)
    return frame.loc[keep].reset_index(drop=True)


def to_wide(product_weeks: pd.DataFrame) -> WideTables:
    """Pivot the product-week table into product x week matrices.

    Placement shares stay NaN in weeks not covered by causal data and are 0 for products
    without placements in covered weeks. Units and revenue are 0 in weeks without sales.

    Args:
        product_weeks: Output of `load_product_weeks`.

    Returns:
        Wide matrices with a common product index and consecutive week columns.
    """
    weeks = range(int(product_weeks["week_no"].min()), int(product_weeks["week_no"].max()) + 1)
    covered = product_weeks.loc[product_weeks["mailer_store_share"].notna(), "week_no"].unique()

    def pivot(column: str, fill_uncovered: bool) -> pd.DataFrame:
        wide = product_weeks.pivot(index="product_id", columns="week_no", values=column)
        wide = wide.reindex(columns=weeks)
        if fill_uncovered:
            return wide.fillna(0)
        filled = wide.copy()
        filled[covered] = filled[covered].fillna(0)
        return filled

    hierarchy = product_weeks.drop_duplicates("product_id").set_index("product_id")[
        ["department", "commodity_desc", "sub_commodity_desc"]
    ]
    units = pivot("units", fill_uncovered=True)
    return WideTables(
        mailer_share=pivot("mailer_store_share", fill_uncovered=False),
        display_share=pivot("display_store_share", fill_uncovered=False),
        units=units,
        revenue=pivot("revenue", fill_uncovered=True),
        hierarchy=hierarchy.reindex(units.index),
    )


def _window_flags(tables: WideTables, promo_week: int, params: PromoParams) -> dict[str, pd.Series]:
    pre = list(range(promo_week - params.pre_weeks, promo_week))
    window = [*pre, promo_week]
    share = tables.mailer_share
    has_sales = (tables.units[pre] > 0).sum(axis=1) >= params.min_pre_weeks_with_sales
    pre_mailer = share[pre].max(axis=1) > 0
    promoted = share[promo_week] >= params.treated_min_store_share
    return {
        "treated": promoted & ~pre_mailer & has_sales,
        "clean": (share[window].max(axis=1) == 0) & has_sales,
        "promoted": promoted,
        "pre_mailer": pre_mailer,
        "window_mailer": share[window].max(axis=1) > 0,
    }


def _promo_weeks(
    tables: WideTables, params: PromoParams, start_week: int, end_week: int
) -> list[int]:
    covered = set(tables.mailer_share.columns[tables.mailer_share.notna().any()])
    return [
        week
        for week in range(start_week + params.pre_weeks, end_week + 1)
        if all(w in covered for w in range(week - params.pre_weeks, week + 1))
    ]


def _event_frame(
    products: pd.Index, event_id: pd.Series, promo_week: int, treated: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": event_id.loc[products].to_numpy(),
            "product_id": products.to_numpy(),
            "promo_week": promo_week,
            "treated": treated.loc[products].to_numpy(),
        }
    )


def promo_events(
    tables: WideTables, params: PromoParams, start_week: int, end_week: int
) -> pd.DataFrame:
    """Treated and control products for every promo week.

    Args:
        tables: Output of `to_wide`.
        params: Promo design parameters.
        start_week: First week of the analysis window.
        end_week: Last week of the analysis window.

    Returns:
        Columns `event_id` (sub-commodity and promo week), `product_id`, `promo_week`
        and boolean `treated`; only events with both treated and control products.
    """
    sub_commodity = tables.hierarchy["sub_commodity_desc"]
    frames = []
    for week in _promo_weeks(tables, params, start_week, end_week):
        flags = _window_flags(tables, week, params)
        groups = set(sub_commodity[flags["treated"]]) & set(sub_commodity[flags["clean"]])
        members = (flags["treated"] | flags["clean"]) & sub_commodity.isin(groups)
        event_id = sub_commodity + f"|{week}"
        frames.append(_event_frame(members[members].index, event_id, week, flags["treated"]))
    return pd.concat(frames, ignore_index=True)


def cannibalization_events(
    tables: WideTables, params: PromoParams, start_week: int, end_week: int
) -> pd.DataFrame:
    """Exposed non-promo siblings and control products for every promo week.

    Args:
        tables: Output of `to_wide`.
        params: Promo design parameters.
        start_week: First week of the analysis window.
        end_week: Last week of the analysis window.

    Returns:
        Same columns as `promo_events`; `treated` marks exposed siblings and `event_id`
        is the commodity and promo week.
    """
    sub_commodity = tables.hierarchy["sub_commodity_desc"]
    commodity = tables.hierarchy["commodity_desc"]
    frames = []
    for week in _promo_weeks(tables, params, start_week, end_week):
        flags = _window_flags(tables, week, params)
        new_promo_groups = set(sub_commodity[flags["promoted"]]) - set(
            sub_commodity[flags["pre_mailer"]]
        )
        exposed = flags["clean"] & sub_commodity.isin(new_promo_groups)
        quiet_groups = set(sub_commodity) - set(sub_commodity[flags["window_mailer"]])
        control = flags["clean"] & sub_commodity.isin(quiet_groups)
        commodities = set(commodity[exposed]) & set(commodity[control])
        members = (exposed | control) & commodity.isin(commodities)
        event_id = commodity + f"|{week}"
        frames.append(_event_frame(members[members].index, event_id, week, exposed))
    return pd.concat(frames, ignore_index=True)


def expand_to_panel(events: pd.DataFrame, tables: WideTables, pre_weeks: int) -> pd.DataFrame:
    """Add one row per event week with outcomes and display share.

    Args:
        events: Output of `promo_events` or `cannibalization_events`.
        tables: Output of `to_wide`.
        pre_weeks: Weeks before the promo week.

    Returns:
        Event panel with `rel_week` from `-pre_weeks` to 0, `post`, `units`, `revenue`
        and `display_share`.
    """
    rel_weeks = np.arange(-pre_weeks, 1)
    panel = events.loc[events.index.repeat(len(rel_weeks))].reset_index(drop=True)
    panel["rel_week"] = np.tile(rel_weeks, len(events))
    panel["post"] = panel["rel_week"] == 0
    rows = tables.units.index.get_indexer(panel["product_id"])
    columns = tables.units.columns.get_indexer(panel["promo_week"] + panel["rel_week"])
    panel["units"] = tables.units.to_numpy()[rows, columns]
    panel["revenue"] = tables.revenue.to_numpy()[rows, columns]
    panel["display_share"] = tables.display_share.to_numpy()[rows, columns]
    return panel


def two_way_demean(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove event-specific product and week means from balanced event panels.

    Args:
        panel: Panel with `event_id`, `product_id`, `rel_week` and the given columns.
        columns: Columns to demean.

    Returns:
        Demeaned columns, aligned with `panel`.
    """
    values = panel[columns].astype(float)
    unit_mean = values.groupby([panel["event_id"], panel["product_id"]]).transform("mean")
    week_mean = values.groupby([panel["event_id"], panel["rel_week"]]).transform("mean")
    event_mean = values.groupby(panel["event_id"]).transform("mean")
    return values - unit_mean - week_mean + event_mean


def _fit_clustered(panel: pd.DataFrame, outcome: str, regressors: pd.DataFrame):
    keys = panel[["event_id", "product_id", "rel_week"]]
    design = pd.concat([keys, regressors, np.log1p(panel[outcome]).rename("y")], axis=1)
    demeaned = two_way_demean(design, [*regressors.columns, "y"])
    clusters = pd.factorize(panel["product_id"])[0]
    return sm.OLS(demeaned["y"], demeaned[list(regressors.columns)]).fit(
        cov_type="cluster", cov_kwds={"groups": clusters}, use_t=True
    )


def estimate_did(
    panel: pd.DataFrame, outcome: str, confidence_level: float, controls: tuple[str, ...] = ()
) -> DidResult:
    """Estimate the effect of treatment in the promo week on log(1 + outcome).

    Args:
        panel: Output of `expand_to_panel`.
        outcome: "units" or "revenue".
        confidence_level: Confidence level of the interval, e.g. 0.95.
        controls: Time-varying panel columns added as covariates, e.g. ("display_share",).

    Returns:
        Coefficient of `treated * post`, its interval, p-value and the effect in percent.
    """
    regressors = pd.concat(
        [(panel["treated"] & panel["post"]).rename("treated_post"), panel[list(controls)]], axis=1
    )
    fit = _fit_clustered(panel, outcome, regressors)
    low, high = fit.conf_int(alpha=1 - confidence_level).loc["treated_post"]
    coefficient = fit.params["treated_post"]
    units = panel.drop_duplicates(["event_id", "product_id"])
    return DidResult(
        outcome=outcome,
        coefficient=float(coefficient),
        ci_low=float(low),
        ci_high=float(high),
        p_value=float(fit.pvalues["treated_post"]),
        effect_pct=float(100 * np.expm1(coefficient)),
        effect_pct_ci_low=float(100 * np.expm1(low)),
        effect_pct_ci_high=float(100 * np.expm1(high)),
        observations=len(panel),
        events=int(panel["event_id"].nunique()),
        treated_units=int(units["treated"].sum()),
        control_units=int((~units["treated"]).sum()),
        clusters=int(panel["product_id"].nunique()),
    )


def estimate_event_study(
    panel: pd.DataFrame, outcome: str, confidence_level: float
) -> tuple[pd.DataFrame, float]:
    """Estimate treated-control differences by week relative to the promo week.

    The week right before the promo (`rel_week = -1`) is the reference.

    Args:
        panel: Output of `expand_to_panel`.
        outcome: "units" or "revenue".
        confidence_level: Confidence level of the intervals.

    Returns:
        Table with `rel_week`, `coefficient`, `ci_low`, `ci_high`, `p_value` (the reference
        week has coefficient 0), and the p-value of the joint test that all pre-period
        coefficients are zero.
    """
    weeks = sorted(week for week in panel["rel_week"].unique() if week != -1)
    regressors = pd.DataFrame(
        {f"rel_{week}": panel["treated"] & (panel["rel_week"] == week) for week in weeks}
    )
    fit = _fit_clustered(panel, outcome, regressors)
    intervals = fit.conf_int(alpha=1 - confidence_level)
    table = pd.DataFrame(
        {
            "rel_week": weeks,
            "coefficient": fit.params.to_numpy(),
            "ci_low": intervals[0].to_numpy(),
            "ci_high": intervals[1].to_numpy(),
            "p_value": fit.pvalues.to_numpy(),
        }
    )
    reference = pd.DataFrame(
        {
            "rel_week": [-1],
            "coefficient": [0.0],
            "ci_low": [0.0],
            "ci_high": [0.0],
            "p_value": [np.nan],
        }
    )
    table = pd.concat([table, reference]).sort_values("rel_week").reset_index(drop=True)
    pre_terms = [f"rel_{week}" for week in weeks if week < -1]
    pretrend_p = float(fit.f_test(", ".join(f"{term} = 0" for term in pre_terms)).pvalue)
    return table, pretrend_p


def group_means(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Mean log(1 + outcome) of treated and control products by relative week.

    Args:
        panel: Output of `expand_to_panel`.
        outcome: "units" or "revenue".

    Returns:
        Table indexed by `rel_week` with columns `treated` and `control`.
    """
    means = (
        panel.assign(y=np.log1p(panel[outcome]))
        .groupby(["rel_week", "treated"])["y"]
        .mean()
        .unstack("treated")
    )
    return means.rename(columns={True: "treated", False: "control"})[["treated", "control"]]


def concurrent_display_share(panel: pd.DataFrame) -> dict[str, float]:
    """Share of treated and control products that were also on display in the promo week.

    Args:
        panel: Output of `expand_to_panel`.

    Returns:
        Shares for treated and control product-events.
    """
    promo_week = panel[panel["post"]]
    on_display = promo_week["display_share"] > 0
    return {
        "treated": float(on_display[promo_week["treated"]].mean()),
        "control": float(on_display[~promo_week["treated"]].mean()),
    }
