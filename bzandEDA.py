
# Data Extraction and Processing

import bz2
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to extract data from a bz2 file
def extract_bz2(file_name):
    with bz2.BZ2File(file_name, 'rb') as file:
        data = file.readlines()
    return data

# Extracting the test and train data
test_data = extract_bz2('test.ft.txt.bz2')
train_data = extract_bz2('train.ft.txt.bz2')

# Optionally, converting bytes to strings if the content is textual
test_data = [line.decode('utf-8') for line in test_data]
train_data = [line.decode('utf-8') for line in train_data]

# Example: print the first 5 lines from each dataset
print("First 5 lines of test data:")
print(test_data[:5])

print("\nFirst 5 lines of train data:")
print(train_data[:5])

# Process Amazon Reviews Function

def process_amazon_reviews(file_name):
    with bz2.BZ2File(file_name, 'rb') as file:
        lines = [x.decode('utf-8') for x in file.readlines()]
        
        # Extract labels and sentences
        labels = [0 if line.split(' ')[0] == '__label__1' else 1 for line in lines]
        sentences = [line.split(' ', 1)[1].lower() for line in lines]

        # Clean sentences
        for i in range(len(sentences)):
            # Replace digits with zero
            sentences[i] = re.sub(r'\d', '0', sentences[i])
            # Replace URLs with <url>
            if any(substring in sentences[i] for substring in ['www.', 'http:', 'https:', '.com']):
                sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", sentences[i])

    return labels, sentences

# Process train and test data
train_labels, train_sentences = process_amazon_reviews('train.ft.txt.bz2')
test_labels, test_sentences = process_amazon_reviews('test.ft.txt.bz2')

# Create pandas DataFrames
train_df = pd.DataFrame({'labels': train_labels, 'sentences': train_sentences})
test_df = pd.DataFrame({'labels': test_labels, 'sentences': test_sentences})

# Save processed data to CSV files
train_df.to_csv('amazon_train_reviews.csv', index=False)
test_df.to_csv('amazon_test_reviews.csv', index=False)

## Exploratory Data Analysis (EDA)

# Load train dataset (optional if you already have train_df in memory)
train_val = pd.read_csv('amazon_train_reviews.csv')

# Reset index (in case it's read with index_col or from another source)
train_val.reset_index(drop=True, inplace=True)

# Inspect the dataset
print("\nTrain dataset info:")
train_val.info()

# View first few rows of the data
print("\nFirst few rows of train dataset:")
print(train_val.head())

# Check distribution of label classes
print("\nLabel distribution:")
print(train_val['labels'].value_counts())

# Create a new column 'len' that counts the number of words in each sentence
train_val['len'] = train_val['sentences'].apply(lambda x: len(str(x).split()))

# Plot the distribution of sentence lengths
plt.figure(figsize=(10, 6))
sns.histplot(train_val['len'], bins=50, kde=True)
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length (number of words)')
plt.ylabel('Frequency')
plt.show()

# Calculate average sentence length per sentiment class
neg_mean_len = train_val.groupby('labels')['len'].mean().values[0]
pos_mean_len = train_val.groupby('labels')['len'].mean().values[1]

print(f"\nNegative mean length: {neg_mean_len:.2f}")
print(f"Positive mean length: {pos_mean_len:.2f}")
print(f"Mean Difference: {neg_mean_len - pos_mean_len:.2f}")

# Boxplot for sentence lengths by sentiment label
sns.catplot(x='labels', y='len', data=train_val, kind='box', height=6, aspect=1.5)
plt.title('Boxplot of Sentence Length by Label')
plt.xlabel('Sentiment Label (0=Negative, 1=Positive)')
plt.ylabel('Sentence Length')
plt.show()
