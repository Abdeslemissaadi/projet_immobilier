import matplotlib.pyplot as plt
import seaborn as sns


def exploration_plots(train, df, target_col):
    sample = train.sample(0.1, seed=42).toPandas()
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(sample[target_col], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('Distribution Prix (Train)')
    plt.xlabel('Prix (×$100k)')
    plt.subplot(122)
    corr_sample = df.sample(0.05, seed=42).toPandas().corr()
    sns.heatmap(corr_sample[[target_col]].sort_values(target_col, ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Corrélations avec Prix')
    plt.tight_layout()
    plt.savefig('exploration.png', dpi=200)
    plt.show()
    print("💾 exploration.png sauvegardé\n")
