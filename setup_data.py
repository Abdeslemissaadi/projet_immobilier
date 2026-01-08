import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round, sqrt, pow as spark_pow
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

import pandas as pd
from sklearn.datasets import fetch_california_housing


def setup_spark():
    # ====================================================================
    # BLOC 1 : SETUP COMPLET (Imports + Spark + Données)
    # ====================================================================

    spark = (SparkSession.builder
        .appName("Immobilier_CA")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Télécharger et charger données
    housing = fetch_california_housing(as_frame=True)
    df_pandas = housing.frame
    df_pandas.to_csv("california_housing.csv", index=False)

    df = spark.read.option("header", "true").option("inferSchema", "true").csv("california_housing.csv")
    df = df.repartition(8).cache()

    print("✅ BLOC 1 OK: Setup + Données chargées")
    print(f"📊 {df.count():,} lignes × {len(df.columns)} colonnes\n")
    df.show(10)

    return spark, df


def preprocess_and_split(df):
    # ====================================================================
    # BLOC 2 : EXPLORATION + PREPROCESSING + FEATURE ENGINEERING
    # ====================================================================

    # Exploration rapide
    print("\n📊 STATISTIQUES:")
    df.describe().show()

    # Feature Engineering (5 nouvelles features)
    df = (df
        .withColumn("RoomsPerHH", spark_round(col("AveRooms") / col("AveBedrms"), 2))
        .withColumn("BedroomsRatio", spark_round(col("AveBedrms") / col("AveRooms"), 4))
        .withColumn("PopPerHH", spark_round(col("Population") / col("AveOccup"), 2))
        .withColumn("DistCenter", spark_round(sqrt(spark_pow(col("Latitude")-36.7, 2) + spark_pow(col("Longitude")+119.4, 2)), 4))
        .withColumn("PopDensity", spark_round(col("Population") / col("AveOccup"), 2))
    )

    # Préparer features
    feature_cols = [c for c in df.columns if c != "MedHouseVal"]
    target_col = "MedHouseVal"

    # Pipeline preprocessing
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    preprocessing_pipeline = Pipeline(stages=[assembler, scaler])
    df_processed = preprocessing_pipeline.fit(df).transform(df).select("features", target_col)

    # Split Train/Test
    train, test = df_processed.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()

    print(f"\n✅ BLOC 2 OK: {len(feature_cols)} features | Train: {train.count():,} | Test: {test.count():,}")

    return feature_cols, target_col, train, test, preprocessing_pipeline
