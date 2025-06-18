import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

print ("Hello World!")
def load_dataset(fold, task, SNP_mapping=True, return_tensor=False):
    mytask = []
    for t in task:
        if t == 'CN':
            mytask.append(1)

        if t == 'EMCI':
            mytask.append(2)

        if t == 'LMCI':
            mytask.append(3)

        if t == 'AD':
            mytask.append(4)

    mytask = np.array(mytask)

    path = '/mnt/Data/'

    Y_dis = np.load(path + 'label.npy')
    
    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=Y_dis.shape)

    for t in range(len(mytask)):
        task_idx += np.array(Y_dis == mytask[t])
    task_idx = task_idx.astype(bool)
    Y_dis = Y_dis[task_idx]

    task_idx=np.squeeze(task_idx,1)
    X_MRI = np.load(path + 'MRIdata.npy')[task_idx, :]
    X_PET = np.load(path + 'FDGdata.npy')[task_idx, :]
    X_SNP = np.load(path + 'SNP.npy')[task_idx, :]
    C_sex = np.load(path + 'C_sex.npy')[task_idx]
    C_edu = np.load(path + 'C_edu.npy')[task_idx]
    C_age = np.load(path + 'C_age.npy')[task_idx]
    S_cog = np.load(path + 'Score.npy')[task_idx]

    for i in range(np.unique(Y_dis).shape[0]):
        Y_dis[Y_dis == np.unique(Y_dis)[i]] = i
    y_labels = Y_dis.copy().astype(int)
    Y_dis = np.eye(np.unique(Y_dis).shape[0])[Y_dis]

    # Normalization
    C_age = C_age /100
    C_edu = C_edu /20
    S_cog = S_cog / 30

    # Categorical encoding for the sex code
    for i in range(np.unique(C_sex).shape[0]):
        C_sex[C_sex == np.unique(C_sex)[i]] = i

    # Demographic information concatenation
    C_dmg = np.concatenate((C_sex, C_age, C_edu), 1)
    
    rand_idx = np.random.RandomState(seed=951014).permutation(Y_dis.shape[0])
#     print ( rand_idx)
    X_MRI = X_MRI[rand_idx, ...]
    X_PET = X_PET[rand_idx, ...]
    X_SNP = X_SNP[rand_idx, ...]
    C_dmg = C_dmg[rand_idx, ...]
    Y_dis = Y_dis[rand_idx, ...]
    S_cog = S_cog[rand_idx, ...]

    # Fold dividing
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_idx = np.arange(y_labels.shape[0])
    folds = list(skf.split(all_idx, y_labels))

    test_idx = folds[fold - 1][1]
    remain_idx = folds[fold - 1][0]

    train_idx, val_idx = train_test_split(
        remain_idx, test_size=0.25,
        stratify=y_labels[remain_idx], random_state=42
    )

    # Train
    X_MRI_tr = X_MRI[train_idx, :]
    X_PET_tr = X_PET[train_idx, :]
    X_SNP_tr = X_SNP[train_idx, :]
    C_dmg_tr = C_dmg[train_idx, :]
    Y_dis_tr = Y_dis[train_idx, :]
    S_cog_tr = S_cog[train_idx]

    # Val
    X_MRI_val = X_MRI[val_idx, :]
    X_PET_val = X_PET[val_idx, :]
    X_SNP_val = X_SNP[val_idx, :]
    C_dmg_val = C_dmg[val_idx, :]
    Y_dis_val = Y_dis[val_idx, :]
    S_cog_val = S_cog[val_idx]

    # Test
    X_MRI_ts = X_MRI[test_idx, :]
    X_PET_ts = X_PET[test_idx, :]
    X_SNP_ts = X_SNP[test_idx, :]
    C_dmg_ts = C_dmg[test_idx, :]
    Y_dis_ts = Y_dis[test_idx, :]
    S_cog_ts = S_cog[test_idx]

#     print('---------------------------------------------')

    print (X_SNP_tr)
    print(X_SNP_val)
    print (X_SNP_ts)
    print ('----------------------------------------')
    if SNP_mapping:
        # SNP encoding
        X_SNP_tr1 = X_SNP_tr.copy()
        X_SNP_tr, X_SNP_val = SNP_encoder(X_SNP_tr=X_SNP_tr, X_SNP_ts=X_SNP_val)
        _, X_SNP_ts = SNP_encoder(X_SNP_tr=X_SNP_tr1, X_SNP_ts=X_SNP_ts)
        print (X_SNP_tr)
        print(X_SNP_val)
        print (X_SNP_ts)
    if return_tensor:
        return tf.data.Dataset.from_tensor_slices((X_MRI_tr, X_PET_tr, X_SNP_tr, C_dmg_tr, Y_dis_tr, S_cog_tr)), \
               tf.data.Dataset.from_tensor_slices((X_MRI_val, X_PET_val, X_SNP_val, C_dmg_val, Y_dis_val, S_cog_val)), \
               tf.data.Dataset.from_tensor_slices((X_MRI_ts, X_PET_ts, X_SNP_ts, C_dmg_ts, Y_dis_ts, S_cog_ts))









def SNP_encoder(X_SNP_tr, X_SNP_ts):
    # Based on population, this encoder transforms the discrete SNP vectors to be numerical.
    # The encoder is fit by the training SNP data and applied to the testing SNP data.

    # Fit the encoding table
    encoder = np.empty(shape=(3, X_SNP_tr.shape[1]))
    for i in range(X_SNP_tr.shape[1]):
        for j in [0, 1, 2]:
            encoder[j, i] = np.array(X_SNP_tr[:, i] == j).sum()
    
    encoder /= X_SNP_tr.shape[0]  # (3, 1275)

    X_E_SNP_tr = np.empty(shape=X_SNP_tr.shape)
    X_E_SNP_ts = np.empty(shape=X_SNP_ts.shape)

    # Map the SNP values
    for sbj in range(X_SNP_tr.shape[0]):
        for dna in range(X_SNP_tr.shape[-1]):

            X_E_SNP_tr[sbj, dna] = encoder[..., dna][int(X_SNP_tr[sbj, dna])]
#             print (encoder[..., dna])
#             print ('---------------------------------------------------------')
#             print (X_E_SNP_tr[sbj, dna])
            
    for sbj in range(X_SNP_ts.shape[0]):
        for dna in range(X_SNP_ts.shape[-1]):
            X_E_SNP_ts[sbj, dna] = encoder[..., dna][int(X_SNP_ts[sbj, dna])]

    return X_E_SNP_tr, X_E_SNP_ts

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    pdf = tf.math.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.math.exp(-logvar) + logvar + log2pi), axis=raxis)
    return pdf
  
    
    

task = ['CN', 'AD']
for fold in range(5):
    exp = load_dataset(fold + 1, task)
