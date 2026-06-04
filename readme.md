# MetaOrion

MetaOrion is a species-level, abundance-aware representation learning framework designed for personalized metagenomics. Built on a customized causal transformer architecture, it is pretrained on a harmonized collection of over 100,000 human metagenomes to capture transferable ecological priors, and subsequently fine-tuned for disease prediction. Beyond classification, MetaOrion identifies critical condition-specific biomarkers and leverages its learned embeddings to reconstruct individualized microbial interaction networks directly from single-sample taxonomic profiles. Based on these networks, the framework derives a Microbial Dysbiosis Index (MDI) to precisely quantify structural ecological shifts. Ultimately, MetaOrion provides a unified framework for understanding microbiome organization across health and disease.



## 📁 Repository Structure

Based on the project structure, the main directories are organized as follows:

Plaintext

```
MetaOrion/
├── configs/                  # Configuration files
├── demo_data/                # Example datasets (e.g., profiles, metadata)
│   └── datapaths/            # Processed JSON samples and datapath index files
├── results/                  # Output directory for evaluation results and logs
├── scripts/                  # Execution scripts (preprocessing, evaluation, etc.)
├── src/                      # Source code
│   └── metagenome_model/     # Core model implementations
│       ├── basic/            # Basic modules and utilities
│       ├── config/           # Model configuration classes
│       ├── inference/        # Inference components
│       └── models/           # Pre-train and fine-tune model architectures
└── weights/                  # Pre-trained and fine-tuned model checkpoints
```

## 🛠️ Environment Setup

Before running the code, please ensure you have the required dependencies installed. You can install them via the provided `requirements.txt` file.

Bash

```
# Install dependencies
# pip
pip install -r requirements.txt
# conda
conda install --yes --file requirements.txt
```

## 🚀 Usage Guide

The pipeline generally consists of two main steps: data preprocessing and model evaluation.

### Step 1: Data Preprocessing

First, convert the raw tabular profile and metadata into individual JSON samples for optimized data loading. Run the following command (assuming you are executing this from the `scripts` or root directory):

Bash

```
python data_preprocess.py \
    --cohort "13_Ning_2023" \
    --profile "../demo_data/13_Ning_2023.profile" \
    --metadata "../demo_data/13_Ning_2023.info" \
    --out_dir "../demo_data/datapaths/"
```

This script will filter the taxa sequences and abundances, generating independent `.json` files for each sample and saving them into the specified `--out_dir`.

### Step 2: Model Evaluation & Inference

Once the data is preprocessed, use `accelerate` to launch the evaluation script. This handles model inference and calculates metrics for disease phenotype prediction.

Bash

```
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port 35612 \
    ./MetaOrion/scripts/evaluation_phenotype_model.py \
    --data_dir ./MetaOrion/demo_data/datapaths/ \
    --cohort 13_Ning_2023 \
    --model_name_or_path ./MetaOrion/weights/ \
    --batch_size 48 \
    --mixed_precision fp16 \
    --accumulation_step 2 \
    --output_home ./MetaOrion/results/ \
    --seed 42 
```

## 📊 Results

After the evaluation script completes successfully, all output files, including prediction logs, evaluation metrics, and figures, will be automatically saved in the `results/` directory.

<img src="D:\zhangkexin_work\code\MetaOrion\results\split1\figs\pandisease.umap.png" alt="pandisease.confusion.matrix" style="zoom:9%;" />



<img src="D:\zhangkexin_work\code\MetaOrion\results\split1\figs\pandisease.confusion.matrix.png" alt="pandisease.confusion.matrix" style="zoom:4%;" />