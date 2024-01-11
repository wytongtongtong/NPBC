# Neural Probabilistic Bounded Confidence (NPBC)

Source code of our paper: A Neural Probabilistic Bounded Confidence Model for Opinion Dynamics on Social Networks 


## Requirements
* Python 3.9
* numpy==1.19.5
* opencv_python==4.5.2.52
* pandas==1.2.0
* python_dateutil==2.8.2
* pytreebank==0.2.7
* scikit_learn==1.0.2
* scipy==1.6.0
* torch==1.10.0
* torchmetrics==0.5.1
* torchtext==0.11.0
* torchvision==0.9.1
* transformers==4.8.2

## Directory description

- src/: The models are stored here. 

  - npbc.py: PyTorch implementation of the proposed method, Neural Probabilistic Bounded Confidence (NPBC). 

- working/: The preprocessed datasets are stored here.

  - posts_final_sample_twitter_Abortion.tsv: Dataset of 100 most active users in Twitter Abortion dataset. 

    Each row has the format: `{user_id},{opinion},{time}`.

    `{opinion}` is manually annotated class label.  

    Due to the privacy concern, we removed tweet text. 

  - posts_final_synthetic_consensus.tsv: Synthetic dataset generated using PBC (stochastic opinion dynamics model) with exponent parameter $\rho=-1.0$.  

  - posts_final_synthetic_clustering.tsv: Synthetic dataset generated using PBC with exponent parameter $\rho=0.1$.  

  - posts_final_synthetic_polarization.tsv: Synthetic dataset generated using PBC with exponent parameter $\rho=1.0$.  

    Each row has the format: `{user_id},{opinion},{time}`.


## Data Generation & Collection

### Data generation

We provide synthetic datasets in working/. But you can also generate these datasets. 

- Generate three synthetic datasets

  ```
  python3 simulate.py
  ```

### Data collection

We provide sample real data in working/. 


### Data preprocessing

- Change data formats 

  ```    
  python3 convert_data.py 
  ```    

## How To Use 

### Example of Usage

- Training: Run ```main_npbc.py``` file to train and evaluate the proposed method with default settings. 
 
  ```
  python3 main_npbc.py 
  ```

- To specify the parameters, run
```

```

  - `method`: str (default=NPBC)

  - `dataset`: str (default=synthetic_consensus)

     Options are "synthetic_consensus", "synthetic_clustering", "synthetic_polarization", "sample_twitter_Abortion"

  - `save_dir` specifies the path to save the trained model. The model path defaults to be "./output" if not specified.

  - `num_hidden_layers` specifies the number of layers $L$ in the neural network.

  - `hidden_features` specifies the number of units per layer $N_u$ in the neural network.

  - `alpha` specifies the trade-off hyperparameter $\alpha$.

  - `lr` specifies learning rate. 

  - `K` specifies dimension of latent space. 

  - `type_odm` specifies the choice of opinion dynamics model.  


     If True, then use profile descriptions of Twitter users as input of the neural network. 



