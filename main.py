from setup_data import setup_spark, preprocess_and_split
from training import train_models, compare_and_save_results
from tuning import tune_random_forest
from visualisation import exploration_plots


if __name__ == "__main__":
    # 1. Spark + données
    spark, df = setup_spark()

    # 2. Preprocessing + split
    feature_cols, target_col, train, test, preprocessing_pipeline = preprocess_and_split(df)

    # 2bis. Visualisation exploration
    exploration_plots(train, df, target_col)

    # 3. Entraînement modèles de base
    lr_model, rf_model, metrics, feature_importance_df, test_pred_rf, evaluator_rmse, evaluator_mae, evaluator_r2 = \
        train_models(train, test, feature_cols, target_col)

    # 4. Comparaison + visus
    compare_and_save_results(metrics, feature_importance_df, test_pred_rf, target_col)

    # 5. Optimisation Random Forest
    cv_success, opt_test_rmse, opt_test_r2 = tune_random_forest(
        train=train,
        test=test,
        target_col=target_col,
        rf_metrics=metrics["rf"],
        rf_model=rf_model,
        evaluator_rmse=evaluator_rmse,
        evaluator_mae=evaluator_mae,
        evaluator_r2=evaluator_r2
    )

    print("\n🎉 PROJET COMPLET TERMINÉ AVEC SUCCÈS!")
    spark.stop()
