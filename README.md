# FAIR-UMN-ECAL: Using Neural Networks to Predict Radiation Damage to Lead Tungstate Crystals at the CERN LHC

## Overview

This github repository contains the code for analyzing and modeling the ECAL data. The code is written in Python. We provide some Jupyter notebooks and python scripts with inline descriptions.

Besides this github repository, we have a complementary document that provides background for this problem, describes the purposes of this project, and introduces the dataset and neural network models in more detail. The document can be accessed via [this web page](https://fair-umn.github.io/FAIR-UMN-ECAL/)

This repository includes the following folders and/or files. For *folders*, we provide a respective *Readme* document inside. The structure of folders is shown below: 

```
.
├── data                                  /* The folder for the ECAL datasets
|   ├── interim                           /* The whole ECAL datasets for the Jupyter notebook src/MAIN_multipleXtal.ipynb (to be added by user)
|   ├── df_skimmed_xtal_54000_201*        /* Demo datasets for the Jupyter notebook src/MAIN.ipynb  
|   └──  README.md 
|
├── src_vX.0                              /* All you need to train the deep neural network (Seq2Seq) models. (v2.0 is the latest version)
|   ├── ecal_dataset_prep.py              /* Script to prepare dataset for Seq2Seq training.
|   |── MAIN.ipynb                        /* Jupyter notebook to process data, train the Seq2Seq model (for single xtal), and make prediction.
|   |── MAIN_multipleXtal.ipynb           /* Jupyter notebook to process data, train the Seq2Seq model (for multiple xtals), and make prediction.  
|   ├── Processing_Results                /* Script to collect and analyze the results
|   ├── seq2seq_model.py                  /* Script to define the Seq2Seq model     
|   ├── seq2seq_prediction.py             /* Script to make predictions on data 
|   ├── seq2seq_train.py                  /* Script to train the Seq2Seq model
|   ├── util.py                           /* Script includes additional helper functions
|   └── README.md  
|
|
└── docs                                   /* Source branch to built the documentation page. 
|
|
└── page_src                               /* The source codes used to build the documentation page
|
|
└── fair_gpu.yml                          /* The YML file to create a GPU execution environment.
|
|
└── fair_cpu.yml                          /* The YML file to create a CPU execution environment.
|
|
└── LICENSE                               /* The MIT LICENSE.

```

## Get Started

We provided two options for users to set up the execution environment: 
- we provide the envrionment YML file so that one can set up the execution environment with it directly;
- we provide the detailed steps and commands to install each required package. 

Before starting, be sure to have the [git](https://git-scm.com/) and [Anaconda3](https://www.anaconda.com/products/individual) installed (alternatively, you can also use [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) instead of[Anaconda3](https://www.anaconda.com/products/individual), which has been tested by us and works well for our demo).

### Set up from the YML file

1. Get and clone the github repository:

   `git clone https://github.com/FAIR-UMN/FAIR-UMN-ECAL`

2. Switch to `FAIR-UMN-ECAL` :

   `cd XXX/FAIR-UMN-ECAL`  (*Note*: `XXX` here indicates the upper directory of `FAIR-UMN-ECAL`. For example, if you clone `FAIR-UMN-ECAL` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use [Anaconda3](https://www.anaconda.com/products/individual-d)):

   `conda deactivate`

4. Create a new conda environment with the YML file (choose GPU or CPU version according to your computational resources):

    GPU version run: `conda env create -f fair_gpu.yml`
   
    CPU version run: `conda env create -f fair_cpu.yml`

5.  Activate conda environment:
    
    `conda activate fair_gpu` (If you choose the GPU version in Step4)
    
    `conda activate fair_cpu` (If you choose the CPU version in Step4)

6. You are now ready to explore the codes/models! Please remember to check the *src/MAIN.ipynb* first



### Set up from the source

1. Get and clone the github repository:

   `git clone https://github.com/FAIR-UMN/FAIR-UMN-ECAL/`

2. Switch to `FAIR-UMN-ECAL` :

   `cd XXX/FAIR-UMN-ECAL`  (*Note*: `XXX` here indicates the upper directory of `FAIR-UMN-ECAL`. For example, if you clone `FAIR-UMN-ECAL` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use [Anaconda3](https://www.anaconda.com/products/individual-d)):

   `conda deactivate`

4. Create a new conda environment:

   `conda create -n fair_umn python=3.6`

5.  Activate conda environment:
    
    `conda activate fair_umn`

6. Install Pytorch (choose GPU or CPU version according to your computational resources):

   GPU version run: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
   
   CPU version run: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
   
7. Install scikit-learn/pandas/matplotlib/numpy/seaborn/tqdm/Jupyter notebook

   ```
   pip install scikit-learn
   pip install pandas
   pip install matplotlib
   pip install numpy
   pip install seaborn
   pip install tqdm
   pip install notebook
   ```
   
8. You are now ready to explore the codes/models! Please remember to check the *src/MAIN.ipynb* first

   
*Note*: 
1) To install Anaconda, please follow its [official guideline](https://docs.anaconda.com/anaconda/user-guide/getting-started/). For example, to install Anaconda3 on Linux, check [here](https://docs.anaconda.com/anaconda/install/linux/); to install Anaconda3 on Windows, check [here](https://docs.anaconda.com/anaconda/install/windows/); and to install Anaconda3 on macOS, check [here](https://docs.anaconda.com/anaconda/install/mac-os/).
3) We test our model on Ubuntu, Windows, and macOS.

## V2.0 Update

- **Data preprocessing**: remove data which has small luminosity (i.e., remove luminosity <= some threshold epsilon) so that the calibration recovery parts will be removed.

- **Data visualization**: improve the data visualization code to clearly separate the data in luminosity recovery part from the entire sequence.

## Future Work

- Try different initialization strategy to find a better minimizer.

- Conduct more experiments to determine the best iteration number, window size, and batch size for the v2.0 data (after removing calibration recovery parts).

## Support or Contact

If you need any help, please feel free to contact us:
- [Buyun Liang](https://buyunliang.org/) (*liang664 an_at_symbol umn a_dot_symbol edu*)
- [Bhargav Joshi](https://www.linkedin.com/in/bhargav-joshi-0732152b/?originalSubdomain=in) (*joshib an_at_symbol umn a_dot_symbol edu*)


