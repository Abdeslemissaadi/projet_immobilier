from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_models(train, test, feature_cols, target_col):
    # ====================================================================
    # BLOC 3 : MODÈLE 1 - RÉGRESSION LINÉAIRE (PySpark ML)
    # ====================================================================

    print("🤖 MODÈLE 1: RÉGRESSION LINÉAIRE")
    print("="*60)

    # Créer et entraîner modèle
    lr = LinearRegression(featuresCol="features", labelCol=target_col, maxIter=100, regParam=0.01, elasticNetParam=0.5)
    lr_model = lr.fit(train)

    # Prédictions
    train_pred_lr = lr_model.transform(train)
    test_pred_lr = lr_model.transform(test)

    # Évaluation
    evaluator_rmse = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")

    lr_train_rmse = evaluator_rmse.evaluate(train_pred_lr)
    lr_train_mae = evaluator_mae.evaluate(train_pred_lr)
    lr_train_r2 = evaluator_r2.evaluate(train_pred_lr)

    lr_test_rmse = evaluator_rmse.evaluate(test_pred_lr)
    lr_test_mae = evaluator_mae.evaluate(test_pred_lr)
    lr_test_r2 = evaluator_r2.evaluate(test_pred_lr)

    print(f"\n📊 RÉSULTATS RÉGRESSION LINÉAIRE:")
    print(f"   TRAIN → RMSE: {lr_train_rmse:.4f} | MAE: {lr_train_mae:.4f} | R²: {lr_train_r2:.4f}")
    print(f"   TEST  → RMSE: {lr_test_rmse:.4f} | MAE: {lr_test_mae:.4f} | R²: {lr_test_r2:.4f}")
    print(f"\n✅ BLOC 3 OK: Régression Linéaire terminée\n")

    # ====================================================================
    # BLOC 4 : MODÈLE 2 - RANDOM FOREST (PySpark ML)
    # ====================================================================

    print("🌲 MODÈLE 2: RANDOM FOREST")
    print("="*60)

    # Créer et entraîner modèle
    rf = RandomForestRegressor(featuresCol="features", labelCol=target_col, numTrees=100, maxDepth=10, seed=42)
    rf_model = rf.fit(train)

    # Prédictions
    train_pred_rf = rf_model.transform(train)
    test_pred_rf = rf_model.transform(test)

    # Évaluation
    rf_train_rmse = evaluator_rmse.evaluate(train_pred_rf)
    rf_train_mae = evaluator_mae.evaluate(train_pred_rf)
    rf_train_r2 = evaluator_r2.evaluate(train_pred_rf)

    rf_test_rmse = evaluator_rmse.evaluate(test_pred_rf)
    rf_test_mae = evaluator_mae.evaluate(test_pred_rf)
    rf_test_r2 = evaluator_r2.evaluate(test_pred_rf)

    print(f"\n📊 RÉSULTATS RANDOM FOREST:")
    print(f"   TRAIN → RMSE: {rf_train_rmse:.4f} | MAE: {rf_train_mae:.4f} | R²: {rf_train_r2:.4f}")
    print(f"   TEST  → RMSE: {rf_test_rmse:.4f} | MAE: {rf_test_mae:.4f} | R²: {rf_test_r2:.4f}")

    # Feature Importance
    importances = rf_model.featureImportances.toArray()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    print(f"\n🔝 TOP 10 FEATURES IMPORTANTES:")
    print(feature_importance_df.to_string(index=False))
    print(f"\n✅ BLOC 4 OK: Random Forest terminé\n")

    metrics = {
        "lr": {
            "train_rmse": lr_train_rmse,
            "train_mae": lr_train_mae,
            "train_r2": lr_train_r2,
            "test_rmse": lr_test_rmse,
            "test_mae": lr_test_mae,
            "test_r2": lr_test_r2
        },
        "rf": {
            "train_rmse": rf_train_rmse,
            "train_mae": rf_train_mae,
            "train_r2": rf_train_r2,
            "test_rmse": rf_test_rmse,
            "test_mae": rf_test_mae,
            "test_r2": rf_test_r2
        }
    }

    return lr_model, rf_model, metrics, feature_importance_df, test_pred_rf, evaluator_rmse, evaluator_mae, evaluator_r2


def compare_and_save_results(metrics, feature_importance_df, test_pred_rf, target_col):
    # ====================================================================
    # BLOC 5 : COMPARAISON + VISUALISATIONS + SAUVEGARDE
    # ====================================================================

    print("📊 COMPARAISON FINALE DES MODÈLES")
    print("="*60)

    lr = metrics["lr"]
    rf = metrics["rf"]

    results = pd.DataFrame({
        'Modèle': ['Régression Linéaire', 'Random Forest'],
        'RMSE_Train': [lr["train_rmse"], rf["train_rmse"]],
        'RMSE_Test': [lr["test_rmse"], rf["test_rmse"]],
        'MAE_Train': [lr["train_mae"], rf["train_mae"]],
        'MAE_Test': [lr["test_mae"], rf["test_mae"]],
        'R²_Train': [lr["train_r2"], rf["train_r2"]],
        'R²_Test': [lr["test_r2"], rf["test_r2"]]
    })

    print("\n" + results.to_string(index=False))

    # Déterminer le meilleur modèle
    best_model = "Random Forest" if rf["test_rmse"] < lr["test_rmse"] else "Régression Linéaire"
    improvement = abs(lr["test_rmse"] - rf["test_rmse"]) / lr["test_rmse"] * 100

    print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
    print(f"📈 Amélioration RMSE: {improvement:.2f}%")

    # Sauvegarder résultats
    results.to_csv("resultats_comparaison.csv", index=False)
    feature_importance_df.to_csv("feature_importance.csv", index=False)

    # Visualisations finales
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Graph 1: Comparaison RMSE
    axes[0,0].bar(['LR Train', 'LR Test', 'RF Train', 'RF Test'], 
                  [lr["train_rmse"], lr["test_rmse"], rf["train_rmse"], rf["test_rmse"]],
                  color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_title('Comparaison RMSE')
    axes[0,0].grid(axis='y', alpha=0.3)

    # Graph 2: R² Score
    axes[0,1].bar(['LR Train', 'LR Test', 'RF Train', 'RF Test'],
                  [lr["train_r2"], lr["test_r2"], rf["train_r2"], rf["test_r2"]],
                  color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].set_title('Comparaison R²')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].grid(axis='y', alpha=0.3)

    # Graph 3: Prédictions vs Réel (Random Forest)
    sample_pred = test_pred_rf.sample(0.1, seed=42).toPandas()
    axes[1,0].scatter(sample_pred[target_col], sample_pred['prediction'], alpha=0.5, s=10)
    axes[1,0].plot([0, 5], [0, 5], 'r--', lw=2)
    axes[1,0].set_xlabel('Prix Réel')
    axes[1,0].set_ylabel('Prix Prédit')
    axes[1,0].set_title('Random Forest: Prédictions vs Réel')
    axes[1,0].grid(True, alpha=0.3)

    # Graph 4: Feature Importance (Top 10)
    top10 = feature_importance_df.head(10)
    axes[1,1].barh(top10['Feature'], top10['Importance'], color='teal')
    axes[1,1].set_xlabel('Importance')
    axes[1,1].set_title('Top 10 Features Importantes (RF)')
    axes[1,1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('resultats_finaux.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n💾 Fichiers sauvegardés:")
    print("   📊 exploration.png")
    print("   📊 resultats_finaux.png")
    print("   📄 resultats_comparaison.csv")
    print("   📄 feature_importance.csv")
