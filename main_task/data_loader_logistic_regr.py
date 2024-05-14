import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# .T transzponált azért kell, mert a páciensek oszloponként vannak az eredeti táblázatban
bc_df = pd.read_csv('bc_data_mrna_illumina_microarray.txt', sep='\t') \
    .T \
    .drop(index=['Entrez_Gene_Id']) \
    .reset_index()

g_df = pd.read_csv('glioblastoma_data_mrna_agilent_microarray.txt', sep='\t') \
    .T \
    .drop(index=['Entrez_Gene_Id']) \
    .reset_index()

ov_df = pd.read_csv('ovarian_data_mrna_agilent_microarray.txt', sep='\t') \
    .T \
    .drop(index=['Entrez_Gene_Id']) \
    .reset_index()


# Rename columns using values from the first row (hugo_symbols)
bc_df.columns = bc_df.iloc[0]
g_df.columns = g_df.iloc[0]
ov_df.columns = ov_df.iloc[0]

# delete duplicate column names for inner concat - unique column name is needed
bc_df = bc_df.loc[:,~bc_df.columns.duplicated()].copy()
g_df = g_df.loc[:,~g_df.columns.duplicated()].copy()
ov_df = ov_df.loc[:,~ov_df.columns.duplicated()].copy()


# drop the first column (the "Hugo_Symbol" string and the patient IDs) then the first row (because we named the columns after the hugo_symbols)
# sort columns alphabetically
# replace NaN values with median
bc_df = bc_df.iloc[:, 1:] \
    .drop(0)
bc_df = bc_df.fillna(bc_df.median())

g_df = g_df.iloc[:, 1:] \
    .drop(0)
g_df = g_df.fillna(g_df.median())


ov_df = ov_df.iloc[:, 1:] \
    .drop(0)
ov_df = ov_df.fillna(ov_df.median())

# check the dataframes' structure:
# print(bc_df.head(), '\n')
# print(g_df.head())


# it retains only the merged columns
merged_df = pd.concat([bc_df, g_df, ov_df], join='inner', ignore_index=True)

# check the merged_df:
# print(merged_df.head())
print('number of common genes:', merged_df.shape[1], "n_samples: ", merged_df.shape[0], "bc_samples: ", bc_df.shape[0], "g_samples: ", g_df.shape[0], "ov_samples: ", ov_df.shape[0])


# ------- PCA -------
X = merged_df.values
pca = PCA(n_components=200)  # Select the number of components you want
X_pca = pca.fit_transform(X)
merged_df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# ------ normalize -------
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
merged_df_pca = pd.DataFrame(min_max_scaler.fit_transform(merged_df_pca))

# make the labels
cancer_types = {0: "Breast cancer", 1: "Glioblastomaa", 2: "Ovarian cancer"}

bc_labels = pd.Series(0, index=range(bc_df.shape[0]))
g_labels = pd.Series(1, index=range(g_df.shape[0]))
ov_labels = pd.Series(2, index=range(ov_df.shape[0]))

conc_labels = pd.concat([bc_labels, g_labels, ov_labels], ignore_index=True)
conc_labels.name="conc_labels"

# the ultimate beast unified dataset
labeled_combined_df = pd.concat([conc_labels, merged_df_pca], axis=1)

# Shuffle the dataset
labeled_combined_df = labeled_combined_df.sample(frac=1, random_state=42)

# TRAIN-TEST split: 80-20
train_X, test_X, train_y, test_y = train_test_split(labeled_combined_df.iloc[:, 1:], labeled_combined_df.iloc[:, :1], test_size=0.2, random_state=42)

# print("train_x head: ", train_X.head())
# print(train_y.head())



# --------- initialize the logistic regreession model, train and test -------
model = LogisticRegression()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(test_y, y_pred))