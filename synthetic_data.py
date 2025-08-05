import numpy as np
import scipy.ndimage as ndimage

# Parameters
num_samples = 512
mri_features = 116
pet_features = 116
gene_features = 2800

# Noise controls
risk_noise_sigma = 1.0
mri_pet_noise_sigma = 0.8
mri_pet_blur_sigma = 1

# Demographic features
np.random.seed(42)
gender_numeric = np.random.randint(0, 2, size=(num_samples, 1))
gender_onehot = np.zeros((num_samples, 2), dtype=int)
gender_onehot[gender_numeric.flatten() == 0, 1] = 1
gender_onehot[gender_numeric.flatten() == 1, 0] = 1
age = np.random.randint(65, 76, size=(num_samples, 1))
education_years = np.random.randint(15, 21, size=(num_samples, 1))

# Gene features
informative_probs = [0.3, 0.3, 0.4]
gene_informative = np.random.choice([0, 1, 2], p=informative_probs, size=(num_samples, 5))

# Noise genes
noise_probs = [0.8, 0.1, 0.05, 0.05]
gene_noise = np.random.choice([0, 0, 1, 2], p=noise_probs, size=(num_samples, gene_features - 5))
gene_data = np.hstack((gene_informative, gene_noise))

# Risk score & labels
a, b, c, d = 0.25, 0.45, 0.15, 0.15
risk_score = (
 a * (age.flatten() - 65) +
 b * gene_informative.sum(axis=1) +
 c * gender_numeric.flatten() +
 d * (20 - education_years.flatten()) +
 np.random.normal(0, risk_noise_sigma, size=num_samples)
)

labels = np.digitize(risk_score, bins=np.percentile(risk_score, [20, 40, 60, 80])) + 1
labels = labels.reshape(-1, 1)

# MRI / PET features
mri_data = np.zeros((num_samples, mri_features))
pet_data = np.zeros((num_samples, pet_features))

for cls in range(1, 6):
    idx = labels.flatten() == cls
    mri_base = np.random.rand(mri_features) * cls / 5
    pet_base = np.random.rand(pet_features) * (6 - cls) / 5

    mri_data[idx] = np.random.normal(loc=mri_base, scale=0.05, size=(idx.sum(), mri_features))
    pet_data[idx] = np.random.normal(loc=pet_base, scale=0.05, size=(idx.sum(), pet_features))

# spatial correlation
mri_data = ndimage.gaussian_filter(mri_data, sigma=1)
pet_data = ndimage.gaussian_filter(pet_data, sigma=1)

# Aggregate demographic matrix
demographic_data = np.hstack((gender_onehot, age, education_years))

# save
np.save("mri_data.npy", mri_data)
np.save("pet_data.npy", pet_data)
np.save("gene_data.npy", gene_data)
np.save("labels.npy", labels)
np.save("demographic_data.npy", demographic_data)
print("saved successfully!")
