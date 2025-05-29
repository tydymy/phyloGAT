# **Host Body-Site Classification GAT**

Phylogenetic GAT \+ MLP to classify host body-site from microbiome profiles, outputting both performance plots and per-class phylogenies with learned attention.

## **Overview**

This project implements a Graph Attention Network (GAT) that leverages phylogenetic information to classify the host body site (e.g., feces, saliva, sebum) based on microbiome profiles. The model takes a feature table (BIOM format), a phylogenetic tree (Newick format), and sample metadata as input. It performs preprocessing, constructs a graph from the phylogenetic tree, and then trains a GAT model combined with a Multi-Layer Perceptron (MLP) for classification.

The script evaluates the model using stratified cross-validation and can train a final model on the entire dataset. It outputs performance metrics, a confusion matrix, a classification report, and importantly, visualizes the learned attention weights on the phylogenetic tree for each class. This allows for insights into which microbial lineages are discriminative for different body sites. MLflow is integrated for experiment tracking.

## **Features**

* **Phylogenetic-aware Classification:** Incorporates phylogenetic relationships into the GAT model.  
* **Data Preprocessing:** Includes filtering by sample prevalence and balancing classes.  
* **Graph Construction:** Efficiently builds graph representations from Newick trees using dendropy.  
* **GAT Model:** Implements a PhyloGAT using torch\_geometric, followed by an MLP for classification.  
* **Cross-Validation:** Employs Stratified K-Fold cross-validation for robust performance evaluation.  
* **Attention Visualization:** Generates plots of phylogenetic trees where branches are colored by their learned attention weights, highlighting important taxa for each body site classification.  
* **Performance Metrics:** Outputs accuracy, confusion matrix, and a detailed classification report.  
* **Configuration Management:** Uses a dataclass for easy configuration of paths, model hyperparameters, and training settings.  
* **MLflow Integration:** Logs parameters, metrics, and artifacts (plots, models, configuration files) for experiment tracking and reproducibility.

## **How it Works**

1. **Data Loading & Preprocessing:**  
   * Loads microbiome feature data from a BIOM file.  
   * Loads sample metadata (including host and body site).  
   * Loads a phylogenetic tree (Newick format).  
   * Filters samples and features based on configured prevalence and minimum group sizes.  
   * Applies log transformation to feature counts.  
2. **Phylogenetic Graph Construction:**  
   * The PhyloTreeProcessor class converts the Newick tree into a graph format suitable for PyTorch Geometric. It prunes the tree to include only taxa present in the feature data and caches the graph structure.  
3. **Model Architecture (HostBodySiteGAT):**  
   * **Node Embeddings:** Input features (presence/abundance of taxa) are linearly transformed. An additional embedding layer (internal\_emb) learns representations for all nodes in the phylogenetic tree (leaves and internal nodes).  
   * **Phylogenetic GAT (PhyloGAT):** A two-layer GATConv network processes the node features and graph structure. It learns to weigh the importance of neighboring nodes (taxa) in the phylogenetic tree.  
   * **Global Pooling:** global\_mean\_pool aggregates node embeddings to produce a graph-level representation for each sample.  
   * **MLP Classifier:** A Multi-Layer Perceptron takes the graph embedding and predicts the host body site.  
4. **Training & Evaluation:**  
   * The script performs Stratified K-Fold cross-validation.  
   * For each fold, the model is trained, and performance is evaluated on a test set.  
   * Early stopping is used based on validation accuracy.  
   * A final model can be trained on the entire dataset.  
5. **Attention Visualization:**  
   * After training the final model, attention weights from the GAT layers are extracted.  
   * For each predicted class (body site), an average attention map is generated and overlaid on the phylogenetic tree. This highlights which branches (clades or specific taxa) the model "attends" to most when classifying a sample to that body site.

## **Dependencies**

The script relies on the following Python libraries:

* torch  
* torch-geometric  
* numpy  
* pandas  
* scikit-learn  
* dendropy  
* biopython  
* networkx  
* tqdm  
* matplotlib  
* seaborn  
* biom-format  
* mlflow

## **Installation**

1. Clone the repository or download the script.  
2. Ensure you have Python 3.8+ installed.  
3. Install the required dependencies:  
   pip install torch torch-geometric numpy pandas scikit-learn dendropy biopython networkx tqdm matplotlib seaborn biom-format mlflow

   *Note: Installing PyTorch and PyTorch Geometric might require specific commands depending on your CUDA version if you plan to use a GPU. Refer to their official installation guides.*

## **Data**

Place your data files in a directory (default is data/). The script expects the following files:

* **Feature Table:** A BIOM file containing feature (e.g., ASV/OTU) counts per sample.  
  * Default name: 127152\_reference-hit.biom  
* **Phylogenetic Tree:** A Newick tree file representing the phylogenetic relationships between features.  
  * Default name: 127152\_insertion\_tree.relabelled.tre  
* **Metadata File:** A TSV (Tab-Separated Values) file with sample information. It must contain columns for sample identifiers, 'host', and 'env\_material' (the target variable for body site).  
  * Default name: sample\_information\_from\_prep\_72.tsv

Update the file names and paths in the Config class within the script if your files are named or located differently.

## **Configuration**

The script uses a Config dataclass for managing all parameters. You can modify these at the beginning of the script:

@dataclass  
class Config:  
    \# Data paths  
    data\_dir: str \= 'data'  
    feat\_file: str \= '127152\_reference-hit.biom'  
    tree\_file: str \= '127152\_insertion\_tree.relabelled.tre'  
    meta\_file: str \= 'sample\_information\_from\_prep\_72.tsv'

    \# Model parameters  
    feat\_emb: int \= 128       \# Feature embedding dimension  
    gnn\_hid: int \= 128        \# GNN hidden dimension  
    mlp\_hid: int \= 128        \# MLP hidden dimension  
    dropout: float \= 0.2      \# Dropout rate

    \# Training parameters  
    n\_folds: int \= 5          \# Number of cross-validation folds  
    epochs: int \= 50          \# Maximum training epochs  
    lr: float \= 1e-3          \# Learning rate  
    weight\_decay: float \= 1e-2 \# Weight decay for optimizer  
    batch\_size: int \= 16      \# Batch size  
    patience: int \= 10        \# Patience for early stopping

    \# Preprocessing  
    min\_prevalence: float \= 0.1 \# Minimum feature prevalence  
    min\_samples\_per\_group: int \= 2 \# Min samples per class for balancing

    \# Output  
    output\_dir: str \= 'outputs' \# Directory for saving results  
    save\_models: bool \= True    \# Whether to save trained models  
    save\_plots: bool \= True     \# Whether to save plots

    \# Device  
    device: str \= 'cuda' if torch.cuda.is\_available() else 'cpu'

## **Usage**

1. Ensure your data files are in the specified data\_dir or update the Config paths.  
2. Modify any other parameters in the Config class as needed.  
3. Run the Python script:  
   python phyloGAT.py

The script will perform cross-validation, log results using MLflow (if an MLflow server is running or it will log locally to mlruns), train a final model, and save outputs to the outputs/ directory.

## **Output**

The script generates several outputs, typically saved in the outputs/ directory (unless changed in Config):

* config.json: A JSON file saving the configuration used for the run.  
* cv\_results.csv: A CSV file with the accuracy for each cross-validation fold.  
* classification\_report.csv: A CSV file detailing precision, recall, F1-score for each class.  
* confusion\_matrix.png: A plot of the overall confusion matrix from the cross-validation.  
* best\_model\_fold\<N\>.pt: Saved PyTorch model state dictionary for the best model of each fold (if save\_models is True).  
* outputs/attention\_plots/: This subdirectory will contain:  
  * attention\_\<class\_name\>.png: Phylogenetic tree plots showing attention weights for each classified body site.

MLflow Tracking:  
If MLflow is set up, the script will create an experiment (default name: "phylo\_gat\_body\_site") and log parameters, metrics (like mean accuracy), and artifacts (all saved plots, config file, CSV reports).

## **Results Visualization Examples**

### Confusion Matrix

The model's performance in distinguishing between different body sites is summarized in a confusion matrix. Below is an example:

![Confusion Matrix](https://raw.githubusercontent.com/tydymy/phyloGAT/main/data/cm_gat.png)

This matrix shows the number of true versus predicted classifications for each body site. For example, out of all "feces" samples, 235 were correctly predicted as "feces", while 27 were misclassified as "sebum".

### Phylogenetic Attention Maps

These plots visualize which parts of the phylogenetic tree the model focuses on (attends to) when classifying samples into a specific body site. Branches with higher attention weights (typically warmer colors like red/orange) are more influential for that class.

**Example Attention Map for "sebum":**
![Phylogenetic Attention Map: sebum](https://raw.githubusercontent.com/tydymy/phyloGAT/main/data/sebun_att.png)

**Example Attention Map for "saliva":**
![Phylogenetic Attention Map: saliva](https://raw.githubusercontent.com/tydymy/phyloGAT/main/data/saliva_att.png)

**Example Attention Map for "feces":**
![Phylogenetic Attention Map: feces](https://raw.githubusercontent.com/tydymy/phyloGAT/main/data/feces_att.png)

These maps help in identifying specific microbial lineages or clades that are characteristic or discriminative for each body site according to the trained GAT model.
