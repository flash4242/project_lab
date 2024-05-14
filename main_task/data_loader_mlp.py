import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

'''
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
'''

# .T transzponált azért kell, mert a páciensek oszloponként vannak az eredeti táblázatban
bc_df = pd.read_csv('bc_data_mrna_illumina_microarray.txt', sep='\t') \
    .T \
    .drop(index=['Entrez_Gene_Id']) \
    .reset_index()

g_df = pd.read_csv('glioblastoma_data_mrna_agilent_microarray.txt', sep='\t') \
    .T \
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
# print(bc_df.head())
# print('\n')
# print(g_df.head())
'''

# it retains only the merged columns
merged_df = pd.concat([bc_df, g_df, ov_df], join='inner', ignore_index=True)

# check the merged_df:
# print(merged_df.head())
# print('number of common genes:', merged_df.shape[1])

# ------- PCA -------
X = merged_df.values
pca = PCA(n_components=200)  # Select the number of components you want
X_pca = pca.fit_transform(X)
merged_df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# ------ normalize --------
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
merged_df = pd.DataFrame(min_max_scaler.fit_transform(merged_df))

# make the labels
cancer_types = {0: "Breast cancer", 1: "Glioblastomaa", 2: "Ovarian cancer"}

bc_labels = pd.Series(0, index=range(bc_df.shape[0]))
g_labels = pd.Series(1, index=range(g_df.shape[0]))
ov_labels = pd.Series(2, index=range(ov_df.shape[0]))

conc_labels = pd.concat([bc_labels, g_labels, ov_labels], ignore_index=True)
conc_labels.name="conc_labels"

# the ultimate beast unified dataset
labeled_combined_df = pd.concat([conc_labels, merged_df], axis=1)

# Shuffle the dataset
labeled_combined_df = labeled_combined_df.sample(frac=1, random_state=42)



# TRAIN-VALIDATION-TEST split: 60-20-20
train_val, test = train_test_split(labeled_combined_df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.25, random_state=42)

print(train.head())


# make the tensors
train = torch.tensor(train.values).to(torch.float32)
val = torch.tensor(val.values).to(torch.float32)
test = torch.tensor(test.values).to(torch.float32)
print(train.shape)

# defining the dataset for dataloaders
class MyDataset(Dataset):
    def __init__(self, tensor, labels):
        self.X = tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

# making the minbatches for DL
train_ds = MyDataset(train[:, 1:], train[:, :1])
val_ds = MyDataset(val[:, 1:], val[:, :1])
test_ds = MyDataset(test[:, 1:], test[:, :1])

batch_size = 32
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
validationloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

# define the net architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, len(cancer_types)) 

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# check if everything is as expected
model = Net()
print(model)

if train_on_gpu:
    model.cuda()

# specify loss function (categorical cross-entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)


# -------- TRAINING ----------
n_epochs = 50
valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        if train_on_gpu:         # move tensors to GPU if CUDA is available
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in validationloader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(validationloader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cancer.pth')
        valid_loss_min = valid_loss


# load the model with the lowest val_loss
model.load_state_dict(torch.load('model_cancer.pth'))


# -------------- TEST --------------
# track test loss
test_loss = 0.0
n_classes = cancer_types.len
class_correct = list(0. for i in range(n_classes))
class_total = list(0. for i in range(n_classes))

model.eval()
for data, target in testloader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(n_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            cancer_types.get(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (cancer_types.get(i)))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
'''