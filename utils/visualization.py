import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualization(X, y, graph: str):

    if graph == 'boxplot':  # observe if different categories has different price distribution, outlier can also be observed
        cols = X.select_dtypes(include=['object']).columns.to_list()
        for col in cols:
            X[col] = X[col].fillna('Unknown')
            X[col] = X[col].replace(True, 'True')
        for col in cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=X[col], y=y)
            plt.title(f'Box Plot: {col} vs price')
            plt.xlabel(col)
            plt.ylabel('price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        print('end of boxplot')

    if graph == 'scatter': # observe the correlation type (linear/non-linear) and how strong corelated to price is for numerical cols, outlier can also be observed.
        cols = X.select_dtypes(include=['number']).columns.to_list()
        for col in cols:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=X[col], y=y)
            plt.title(f'Scatter Plot: {col} vs price')
            plt.xlabel(col)
            plt.ylabel('price')
            plt.tight_layout()
            plt.show()
        print('end of scatter')