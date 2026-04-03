# TriDeepDSM

We developed a novel machine learning framework, **TriDeepDSM**, for predicting cancer driver synonymous mutations. TriDeepDSM employs a tri-view deep learning architecture that facilitates the learning of discriminative features from various representations. We constructed the benchmark datasets and encoded an expansive 2003-dimensional feature space, including sequence embeddings (DNABERT-2, Hyena-DNA), DNA shape (DNAshapeR), physicochemical properties, and basic biological information such as conservation and splicing features. TriDeepDSM combines Tabular Expert Models (LGBM, XGBoost, RF, ERT, HistGB) with a Tri-View Deep Neural Network to enhance predictive accuracy. The evaluation results on benchmark test datasets demonstrate that TriDeepDSM outperforms existing methods, making it a valuable tool for researchers in identifying functional synonymous mutations in cancer.

The details are summarized as follows:

* **data**: it contains the original mutation data and template data in the form of VCF, including COSMIC as the benchmark dataset, and SomaMutDB and EOSM as independent test sets.
* **Datasets**: it contains PyTorch Dataset classes and data alignment scripts used in the project, including `dataset_tabular.py`, `dataset_triview.py`, and `load_vcf.py`.
* **environment**: it contains the `environment.yml` file for detailed Conda environment settings.
* **feature**: it contains the processed data and encoded features used in the project, including `Biological`, `DNAbert-2`, `DNAshapeR`, `hyena-dna`, `one-hot`, and `physicochemical`.
* **models**: neural network architectures used and saved during the project, including the core `TriView_Net.py`.
* **out**: it contains intermediate output result files, including scoring of test and training data, generated meta-features, optimal processing pipelines, and saved model weights (`.pkl` and `.pth`).
* **src**: it contains the code used in the project, including the process of training, feature extraction, and testing the model.

## Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/). See the file for detailed environment Settings at `/environment/environment.yml`.

To create a conda environment for TriDeepDSM, you can import the entire environment for:
 ```
conda env create -f environment/environment.yml
 ```
Then activate the environment:
 ```
conda activate TriDeepDSM
 ```
Ensure the project root is added to your Python path to avoid module import errors:

 ```
export PYTHONPATH=$(pwd)
 ```
Usage
Please see the template data at /data, it contains various characteristic data and synonymous mutations in the form of VCF. If you are trying to using TriDeepDSM with your own data, please process your data into the same format as it, and encode your mutations with the aforementioned features into the /feature directory.

## Examples
In TriDeepDSM, we have listed three types of data, including COSMIC, SomaMutDB, and EOSM. If you want to use these data to run our model, you can follow the example below to get started faster.
1.If you want to generate meta-features and retrain the 5 tabular expert models (LGBM, XGB, RF, ERT, HistGB), you can ideally run like this:

 ```
python src/run_experts.py
 ```

2.If you want to retrain a new deep learning model (TriView_Net) via 10-fold cross-validation, you can get it by training COSMIC data. Of course, it will take a certain amount of time, please be patient:

```
python src/deep_feature.py
```

3.If you want to extract deep representations from the trained Tri-View Network, fuse them with tabular features, perform advanced feature selection, and train the final ultimate classifier (LightGBM), you can run:

```
python src/train.py
```

4.If you want to get the prediction scores and comprehensive metrics (AUC, AUPR, ACC, F1, MCC) in independent test sets (EOSM and SomaMutDB) via TriDeepDSM, you can ideally run like this:CC, F1, MCC) in independent test sets (EOSM and SomaMutDB) via TriDeepDSM, you can ideally run like this:
 
```
python src/pipeline_test2.py
 ```

Citing
If you find TriDeepDSM useful for your research, please consider citing this paper: