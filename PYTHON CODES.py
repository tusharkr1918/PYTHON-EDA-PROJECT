import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV Data
df = pd.read_csv("D:/Python/Python_DataScience/PYTHON CA2/Street_Centerline_DATASET.csv")
print("✅ CSV loaded successfully!")
print("Shape of dataset:", df.shape)
print("Columns in dataset:", df.columns.tolist(), "\n")

# OBJECTIVE 01: Top 10 Most Common Street Names
print("Objective 1: Top 10 Most Common Street Names:")
top_streets = df['ST_NAME'].value_counts().head(10)
print(top_streets)

plt.figure(figsize=(10,6))
sns.barplot(x=top_streets.values, y=top_streets.index, palette='viridis')
plt.title('Top 10 Most Common Street Names')
plt.xlabel('Frequency')
plt.ylabel('Street Name')
plt.tight_layout()
plt.show()

# OBJECTIVE 02: Top 10 Longest Street Segments
print("\nObjective 2: Top 10 Longest Street Segments:")
top_segments = df[['ST_NAME', 'LENGTH']].dropna()
top_segments = top_segments.sort_values(by='LENGTH', ascending=False).head(10)
print(top_segments)

# Use NumPy for basic stats
lengths = top_segments['LENGTH'].to_numpy()
print("Segment Stats → Max:", np.max(lengths), "| Min:", np.min(lengths), "| Mean:", np.mean(lengths), "\n")

top_segments = top_segments.reset_index(drop=True)
plt.figure(figsize=(10,6))
plt.hlines(y=top_segments['ST_NAME'], xmin=0, xmax=top_segments['LENGTH'], color='steelblue')
plt.plot(top_segments['LENGTH'], top_segments['ST_NAME'], "o", color='orange')
plt.title('Top 10 Longest Street Segments')
plt.xlabel('Segment Length')
plt.ylabel('Street Name')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# OBJECTIVE 03: Segment Length Distribution by Responsible Authority
print("Objective 3: Responsible Authority and Segment Length Distribution")
cleaned_df = df[['RESPONSIBL', 'LENGTH']].dropna()
top_authorities = cleaned_df['RESPONSIBL'].value_counts().nlargest(8).index
filtered_df = cleaned_df[cleaned_df['RESPONSIBL'].isin(top_authorities)]
print("Top Authorities:\n", filtered_df['RESPONSIBL'].value_counts())

print("\nUsing describe():")
print(filtered_df.groupby('RESPONSIBL')['LENGTH'].describe())

plt.figure(figsize=(12,6))
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

# OBJECTIVE 04: Street Direction Types
print("\nObjective 4: Street Direction Counts")
oneway_counts = df['ONEWAY'].value_counts()
print(oneway_counts)

plt.figure(figsize=(7,5))
sns.barplot(x=oneway_counts.index, y=oneway_counts.values, palette='Set2')
plt.title('Street Direction Types')
plt.xlabel('Direction Type (B = Both, T = Toward, TF = Toward/From)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# OBJECTIVE 05: Correlation Matrix
print("\nObjective 5: Correlation Matrix")
correlation_data = df[['LENGTH', 'Shape__Length', 'CLASS']].dropna()
print("Dataset shape before correlation:", correlation_data.shape)
print("\nUsing df.describe():\n", correlation_data.describe())

correlation_matrix = correlation_data.corr()
print("\nCorrelation Table:\n", correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title('Correlation Matrix: Length, Shape Length, and Class')
plt.tight_layout()
plt.show()
