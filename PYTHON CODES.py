import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StreetDataAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        print("✅ CSV loaded successfully!")
        print("Shape of dataset:", self.df.shape)
        print("Columns in dataset:", self.df.columns.tolist(), "\n")

    def top_common_streets(self, top_n=10):
        print("Objective 1: Top", top_n, "Most Common Street Names:")
        top_streets = self.df['ST_NAME'].value_counts().head(top_n)
        print(top_streets)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_streets.values, y=top_streets.index, palette='viridis')
        plt.title('Top {} Most Common Street Names'.format(top_n))
        plt.xlabel('Frequency')
        plt.ylabel('Street Name')
        plt.tight_layout()
        plt.show()

    def longest_street_segments(self, top_n=10):
        print("\nObjective 2: Top", top_n, "Longest Street Segments:")
        top_segments = self.df[['ST_NAME', 'LENGTH']].dropna()
        top_segments = top_segments.sort_values(by='LENGTH', ascending=False).head(top_n)
        print(top_segments)

        lengths = top_segments['LENGTH'].to_numpy()
        print("Segment Stats → Max:", np.max(lengths), "| Min:", np.min(lengths), "| Mean:", np.mean(lengths), "\n")

        top_segments = top_segments.reset_index(drop=True)
        plt.figure(figsize=(10, 6))
        plt.hlines(y=top_segments['ST_NAME'], xmin=0, xmax=top_segments['LENGTH'], color='steelblue')
        plt.plot(top_segments['LENGTH'], top_segments['ST_NAME'], "o", color='orange')
        plt.title('Top {} Longest Street Segments'.format(top_n))
        plt.xlabel('Segment Length')
        plt.ylabel('Street Name')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def segment_distribution_by_authority(self, top_n=8):
        print("Objective 3: Responsible Authority and Segment Length Distribution")
        cleaned_df = self.df[['RESPONSIBL', 'LENGTH']].dropna()
        top_authorities = cleaned_df['RESPONSIBL'].value_counts().nlargest(top_n).index
        filtered_df = cleaned_df[cleaned_df['RESPONSIBL'].isin(top_authorities)]
        print("Top Authorities:\n", filtered_df['RESPONSIBL'].value_counts())

        print("\nUsing describe():")
        print(filtered_df.groupby('RESPONSIBL')['LENGTH'].describe())

        plt.figure(figsize=(12, 6))
        sns.stripplot(data=filtered_df, x='RESPONSIBL', y='LENGTH', jitter=True, palette='Set2', alpha=0.6)
        mean_lengths = filtered_df.groupby('RESPONSIBL')['LENGTH'].mean()
        for i, authority in enumerate(top_authorities):
            plt.plot(i, mean_lengths[authority], 'D', color='red', label='Mean' if i == 0 else "")
        plt.title('Distribution of Segment Lengths by Responsible Authority')
        plt.xlabel('Responsible Authority')
        plt.ylabel('Segment Length')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def street_direction_counts(self):
        print("\nObjective 4: Street Direction Counts")
        oneway_counts = self.df['ONEWAY'].value_counts()
        print(oneway_counts)

        plt.figure(figsize=(7, 5))
        sns.barplot(x=oneway_counts.index, y=oneway_counts.values, palette='Set2')
        plt.title('Street Direction Types')
        plt.xlabel('Direction Type (B = Both, T = Toward, TF = Toward/From)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self):
        print("\nObjective 5: Correlation Matrix")
        correlation_data = self.df[['LENGTH', 'Shape__Length', 'CLASS']].dropna()
        print("Dataset shape before correlation:", correlation_data.shape)
        print("\nUsing df.describe():\n", correlation_data.describe())

        correlation_matrix = correlation_data.corr()
        print("\nCorrelation Table:\n", correlation_matrix)

        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray')
        plt.title('Correlation Matrix: Length, Shape Length, and Class')
        plt.tight_layout()
        plt.show()


# ==== USAGE ====
if __name__ == "__main__":
    analyzer = StreetDataAnalyzer("D:/Python/Python_DataScience/PYTHON CA2/Street_Centerline_DATASET.csv")
    analyzer.top_common_streets()
    analyzer.longest_street_segments()
    analyzer.segment_distribution_by_authority()
    analyzer.street_direction_counts()
    analyzer.correlation_matrix()
