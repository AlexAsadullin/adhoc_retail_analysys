"""Clean raw CSV files with PySpark and store them as Parquet."""

import json
import logging
import re
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from retail_analysis.config import Config, PrepareParams, load_config

logger = logging.getLogger(__name__)

SCHEMAS: dict[str, StructType] = {
    "transaction_data": StructType(
        [
            StructField("household_key", IntegerType(), nullable=True),
            StructField("BASKET_ID", LongType(), nullable=True),
            StructField("DAY", IntegerType(), nullable=True),
            StructField("PRODUCT_ID", IntegerType(), nullable=True),
            StructField("QUANTITY", IntegerType(), nullable=True),
            StructField("SALES_VALUE", DoubleType(), nullable=True),
            StructField("STORE_ID", IntegerType(), nullable=True),
            StructField("RETAIL_DISC", DoubleType(), nullable=True),
            StructField("TRANS_TIME", IntegerType(), nullable=True),
            StructField("WEEK_NO", IntegerType(), nullable=True),
            StructField("COUPON_DISC", DoubleType(), nullable=True),
            StructField("COUPON_MATCH_DISC", DoubleType(), nullable=True),
        ]
    ),
    "product": StructType(
        [
            StructField("PRODUCT_ID", IntegerType(), nullable=True),
            StructField("MANUFACTURER", IntegerType(), nullable=True),
            StructField("DEPARTMENT", StringType(), nullable=True),
            StructField("BRAND", StringType(), nullable=True),
            StructField("COMMODITY_DESC", StringType(), nullable=True),
            StructField("SUB_COMMODITY_DESC", StringType(), nullable=True),
            StructField("CURR_SIZE_OF_PRODUCT", StringType(), nullable=True),
        ]
    ),
    "hh_demographic": StructType(
        [
            StructField("classification_1", StringType(), nullable=True),
            StructField("classification_2", StringType(), nullable=True),
            StructField("classification_3", StringType(), nullable=True),
            StructField("HOMEOWNER_DESC", StringType(), nullable=True),
            StructField("classification_5", StringType(), nullable=True),
            StructField("classification_4", StringType(), nullable=True),
            StructField("KID_CATEGORY_DESC", StringType(), nullable=True),
            StructField("household_key", IntegerType(), nullable=True),
        ]
    ),
    "causal_data": StructType(
        [
            StructField("PRODUCT_ID", IntegerType(), nullable=True),
            StructField("STORE_ID", IntegerType(), nullable=True),
            StructField("WEEK_NO", IntegerType(), nullable=True),
            StructField("display", StringType(), nullable=True),
            StructField("mailer", StringType(), nullable=True),
        ]
    ),
}


def to_snake_case(name: str) -> str:
    """Convert a column name such as `BASKET_ID` or `householdKey` to snake_case.

    Args:
        name: Original column name.

    Returns:
        Lowercase snake_case name.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[^0-9a-zA-Z]+", "_", spaced).strip("_").lower()


def build_spark(driver_memory: str) -> SparkSession:
    """Create a local Spark session.

    Args:
        driver_memory: Spark driver memory, e.g. "4g".

    Returns:
        Active Spark session.
    """
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("retail-adhoc-analysis")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        # Commit task files directly, so stray files such as macOS .DS_Store inside
        # _temporary do not break the job commit.
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_raw(spark: SparkSession, raw_dir: Path, table: str) -> DataFrame:
    """Read a raw CSV with an explicit schema and snake_case, trimmed columns.

    Args:
        spark: Spark session.
        raw_dir: Directory with raw CSV files.
        table: Table name, also the CSV file stem.

    Returns:
        DataFrame with snake_case column names; empty strings become nulls.
    """
    frame = spark.read.csv(
        str(raw_dir / f"{table}.csv"), header=True, schema=SCHEMAS[table], mode="FAILFAST"
    )
    columns = []
    for field in frame.schema.fields:
        column = F.col(field.name)
        if isinstance(field.dataType, StringType):
            column = F.when(F.trim(column) == "", None).otherwise(F.trim(column))
        columns.append(column.alias(to_snake_case(field.name)))
    return frame.select(columns)


def drop_duplicates_logged(
    frame: DataFrame, table: str, stats: dict[str, dict[str, int]]
) -> DataFrame:
    """Remove fully duplicated rows and record how many were removed.

    Args:
        frame: Input table.
        table: Table name used as the key in `stats`.
        stats: Mutable dictionary collecting row counts per table.

    Returns:
        Deduplicated table.
    """
    raw_rows = frame.count()
    deduplicated = frame.dropDuplicates()
    clean_rows = deduplicated.count()
    stats[table] = {"rows_raw": raw_rows, "duplicates_removed": raw_rows - clean_rows}
    logger.info("%s: %d rows, %d full duplicates removed", table, raw_rows, raw_rows - clean_rows)
    return deduplicated


def drop_invalid_transactions(frame: DataFrame, stats: dict[str, int]) -> DataFrame:
    """Remove line items with zero quantity or non-positive sales value.

    Args:
        frame: Deduplicated transactions.
        stats: Mutable row-count dictionary of the transactions table.

    Returns:
        Transactions with `quantity > 0` and `sales_value > 0`.
    """
    rows_before = stats["rows_raw"] - stats["duplicates_removed"]
    zero_quantity = frame.filter(F.col("quantity") == 0).count()
    valid_quantity = frame.filter(F.col("quantity") != 0)
    non_positive_sales = valid_quantity.filter(F.col("sales_value") <= 0).count()
    clean = valid_quantity.filter(F.col("sales_value") > 0)
    stats["zero_quantity_removed"] = zero_quantity
    stats["non_positive_sales_removed"] = non_positive_sales
    stats["rows_clean"] = rows_before - zero_quantity - non_positive_sales
    removed_share = (zero_quantity + non_positive_sales) / rows_before
    logger.info(
        "transaction_data: %d rows with quantity = 0 and %d rows with sales_value <= 0 removed "
        "(%.2f%% of rows)",
        zero_quantity,
        non_positive_sales,
        100 * removed_share,
    )
    return clean


def write_parquet(frame: DataFrame, target: Path, partition_by: str | None = None) -> None:
    """Write a table to Parquet, replacing previous output.

    Args:
        frame: Table to write.
        target: Output directory.
        partition_by: Optional partition column.
    """
    writer = frame.write.mode("overwrite")
    if partition_by is not None:
        writer = writer.partitionBy(partition_by)
    writer.parquet(str(target))
    logger.info("Saved %s", target)


def build_promo_product_week(
    causal: DataFrame, transactions: DataFrame, products: DataFrame, params: PrepareParams
) -> DataFrame:
    """Aggregate mailer/display placements and sales to the product-week level.

    Placement shares are the share of stores present in `causal_data` that week which
    placed the product in the mailer or on display. Weeks absent from `causal_data`
    get null shares because placements are unknown there.

    Args:
        causal: Clean causal_data.
        transactions: Clean transactions.
        products: Clean product reference.
        params: Raw code conventions.

    Returns:
        One row per product and week with units, revenue, placement shares and hierarchy.
    """
    stores_in_week = causal.groupBy("week_no").agg(F.countDistinct("store_id").alias("stores"))
    placements = causal.groupBy("product_id", "week_no").agg(
        F.countDistinct(
            F.when(F.col("mailer") != params.mailer_absent_code, F.col("store_id"))
        ).alias("mailer_stores"),
        F.countDistinct(
            F.when(F.col("display") != params.display_absent_code, F.col("store_id"))
        ).alias("display_stores"),
    )
    sales = transactions.groupBy("product_id", "week_no").agg(
        F.sum("quantity").alias("units"), F.sum("sales_value").alias("revenue")
    )
    return (
        sales.join(placements, ["product_id", "week_no"], "full")
        .join(stores_in_week, "week_no", "left")
        .join(
            products.select("product_id", "department", "commodity_desc", "sub_commodity_desc"),
            "product_id",
            "left",
        )
        .select(
            "product_id",
            "week_no",
            "department",
            "commodity_desc",
            "sub_commodity_desc",
            F.coalesce("units", F.lit(0)).alias("units"),
            F.coalesce("revenue", F.lit(0.0)).alias("revenue"),
            (F.coalesce("mailer_stores", F.lit(0)) / F.col("stores")).alias("mailer_store_share"),
            (F.coalesce("display_stores", F.lit(0)) / F.col("stores")).alias("display_store_share"),
        )
    )


def clean_tables(spark: SparkSession, config: Config) -> dict[str, dict[str, int]]:
    """Clean every raw table and save it as Parquet.

    Args:
        spark: Spark session.
        config: Project configuration.

    Returns:
        Row counts per table: raw rows, removed rows by reason and clean rows.
    """
    stats: dict[str, dict[str, int]] = {}
    for table in SCHEMAS:
        frame = drop_duplicates_logged(read_raw(spark, config.paths.raw_dir, table), table, stats)
        if table == "transaction_data":
            frame = drop_invalid_transactions(frame, stats[table])
        else:
            stats[table]["rows_clean"] = (
                stats[table]["rows_raw"] - stats[table]["duplicates_removed"]
            )
        partition_by = "week_no" if table == "causal_data" else None
        write_parquet(frame, config.paths.processed_dir / table, partition_by)
    return stats


def prepare_all(config: Config) -> dict[str, dict[str, int]]:
    """Clean raw tables and build the product-week promo table.

    Args:
        config: Project configuration.

    Returns:
        Row counts per table: raw rows, removed rows by reason and clean rows.
    """
    spark = build_spark(config.prepare.spark_driver_memory)
    processed_dir = config.paths.processed_dir
    try:
        stats = clean_tables(spark, config)
        promo = build_promo_product_week(
            spark.read.parquet(str(processed_dir / "causal_data")),
            spark.read.parquet(str(processed_dir / "transaction_data")),
            spark.read.parquet(str(processed_dir / "product")),
            config.prepare,
        )
        write_parquet(promo, processed_dir / "promo_product_week")
    finally:
        spark.stop()
    return stats


def main() -> None:
    """Entry point for `make prepare`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    stats = prepare_all(config)
    config.paths.prepare_stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
