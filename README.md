# Deep association analysis framework with multi-modal attention fusion for brain imaging genetics

A preliminary implementation of DAAMAF.

**Environment**

See the 'requirements.txt' for environment configuration. 
pip install -r requirements.txt

**Data Preprocess**

All imaging data and SNP data are downloaded from ADNI database.
The imaging data and SNP data are preprocessed by the SPM package and Plink software, respectively.

**Multimodal Attention Fusion Module**

Extracts complementary features from MRI and PET using self-expression and cross-modal attention mechanisms.

**Association Analysis Module** 

The potential feature representation of SNP data was extracted using a variational autocoder and the genetic representation was mapped to the imaging space via a generative network, and SNP-derived features were aligned to the imaging representation via a discriminator.

**Diagnosis Module** 

Uses the fused imaging features representation and genetic-guided mask vector for:
Classification of AD stages (NC, EMCI, LMCI, AD)
Biomarker detection via attentive vectors and SHAP analysis

### How to run classification?

**Data availability**

The data used in this study can be openly available at http://adni.loni.usc.edu/ or you can use the provided synthetic data generation script to simulate multimodal features and labels.

```Run the following script file 'synthetic_data.py' to generate synthetic data.

**Run**

Training and testing are integrated in file `engine_experiment.py`. To run python engine_experiment.py

**code file**

python utils.py	
Loads and processes data; The SNP data are converted into embedding vectors for processing
python engine.py	
Builds VAE encoder/decoder, imaging encoder, cross-modal attention, generator, discriminator, and classifiers
python engine_experiment.py	
Training loop, optimizer calls, metric calculation