#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import LabelEncoder 


# ## Acquire Data

# In[2]:


# Read the churn dataset into a dataframe as variable df.
df = pd.read_csv('churn_raw_data.csv')
# Quick look at the data.
df.head()


# ## Gather insight

# In[3]:


# Print information about the dataset.
print(df.info())


# In[4]:


# Print statistical description of columns in the dataset. 
print(df.describe(include='all'))


# In[5]:


# Print the sum of mising values in each column.
print(df.isnull().sum())


# In[6]:


# Print the percentage of missing values for each column.
print(df.isnull().sum() / len(df) * 100)


# In[7]:


# Print the sum of duplicates in each column. 
print(df.duplicated().sum())


# In[8]:


# Print the data type of each column.
print(df.dtypes)


# In[9]:


# Print unique values for categorical columns.
for col in df.select_dtypes(include='object').columns:
    print(f"Unique values in {col}: {df[col].unique()}")


# In[10]:


# Create boxplots to detect outliers in columns.
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title(f'{col}')
    plt.show()


# ## Data Cleaning

# In[11]:


# Dropped unused column
df.drop(columns=['Unnamed: 0', 'CaseOrder', 'Customer_id', 'Interaction', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Job'], inplace=True)

# Change categorical columns to lowercase.
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower()
print(df[col])


# In[12]:


# Create variable to hold numerical to categorical dtype changes
data_type_corrections = {  
    'item1': 'str',
    'item2': 'str',
    'item3': 'str',
    'item4': 'str',
    'item5': 'str',
    'item6': 'str',
    'item7': 'str',
    'item8': 'str'
}

# Change data types
for column, dtype in data_type_corrections.items():
    df[column] = df[column].astype(dtype)


# In[13]:


# Renaming item columns.
rename_columns = {
    'item1': 'Timely response',
    'item2': 'Timely fixes',
    'item3': 'Timely replacements',
    'item4': 'Reliability',
    'item5': 'Options',
    'item6': 'Respectful response',
    'item7': 'Courteous exchange',
    'item8': 'Evidence of active listening'
}

df.rename(columns=rename_columns, inplace=True)

# Function to replace outliers with the median.
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find the median value.
    median_value = df[column].median()
    
    # Replace outliers with median.
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median_value, df[column])

# Apply the function to all numerical columns.
numerical_columns = df.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    replace_outliers(df, col)


# In[14]:


# Fill missing values with median for numerical columns and mode for categorical columns.
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
    

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)


# In[15]:


# Round numerical values and final look over dataset before saving to csv.
df = df.round(2)
df.head()


# In[16]:


# Create a variable for the cleaned data and save file to csv. 
d206_cleaned_data = 'd206_cleaned_data.csv'
df.to_csv(d206_cleaned_data, index=False)


# ## PCA

# In[17]:


# Create LabelEncoder.
label_encoders = {}

# Encode categorical columns.
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select numerical columns.
numerical_columns = df.select_dtypes(include=[np.number]).columns
df_numerical = df[numerical_columns]

# Standardize the data.
scaler = StandardScaler()
df_numerical_standardized = scaler.fit_transform(df_numerical)

# PCA.
pca = PCA()
pca.fit(df_numerical_standardized)

# Get the principal component loading matrix.
loading_matrix = pca.components_

# Create a df for the loading matrix.
loading_matrix_df = pd.DataFrame(loading_matrix, columns=numerical_columns)

# Save the matrix to csv.
loading_matrix_df.to_csv('loading_matrix.csv', index=False)

# Show top five rows of loading matrix df.
loading_matrix_df.head()


# In[18]:


# Plot the Scree plot and save as a .png
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('scree_plot.png')
plt.show()

