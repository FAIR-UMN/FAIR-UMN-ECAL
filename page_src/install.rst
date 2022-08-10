Installation
========================

We provided two options for users to set up the execution environment: 
- we provide the envrionment YML file so that one can set up the execution environment with it directly;
- we provide the detailed steps and commands to install each required package. 

Before starting, be sure to have the `git <https://git-scm.com/>`_ and `Anaconda3 <https://www.anaconda.com/products/individual>`_ installed (alternatively, you can also use `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ instead of `Anaconda3 <https://www.anaconda.com/products/individual>`_, which has been tested by us and works well for our demo).

Set up from the YML file
----------------------------------

1. Get and clone the github repository::

        git clone https://github.com/FAIR-UMN/FAIR-UMN-ECAL

2. Switch to ``FAIR-UMN-ECAL`` (*Note*: ``XXX`` here indicates the upper directory of ``FAIR-UMN-ECAL``. For example, if you clone ``FAIR-UMN-ECAL`` under ``/home/Download``, then you should replace ``XXX`` with ``/home/Download``.)::

        cd XXX/FAIR-UMN-ECAL

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use `Anaconda3 <https://www.anaconda.com/products/individual>`_ )::

        conda deactivate

4. Create a new conda environment with the YML file (choose GPU or CPU version according to your computational resources).

    GPU version run::
        
         conda env create -f fair_gpu.yml
   
    CPU version run::
        
         conda env create -f fair_cpu.yml

5.  Activate conda environment.
    
    If you choose the GPU version in Step4::

        conda activate fair_gpu
    
    If you choose the CPU version in Step4::

        conda activate fair_cpu

6. You are now ready to explore the codes/models! Please remember to check the ``src/MAIN.ipynb`` first.



Set up from the source
----------------------------------


1. Get and clone the github repository::

        git clone https://github.com/FAIR-UMN/FAIR-UMN-ECAL

2. Switch to ``FAIR-UMN-ECAL`` (*Note*: ``XXX`` here indicates the upper directory of ``FAIR-UMN-ECAL``. For example, if you clone ``FAIR-UMN-ECAL`` under ``/home/Download``, then you should replace ``XXX`` with ``/home/Download``.)::

        cd XXX/FAIR-UMN-ECAL

3. Deactivate conda base environment first you are in (otherwise, go to step 4 directly) (We use `Anaconda3 <https://www.anaconda.com/products/individual>`_ )::

        conda deactivate

4. Create a new conda environment::

        conda create -n fair_umn python=3.6

5.  Activate conda environment::
    
        conda activate fair_umn

6. Install Pytorch (choose GPU or CPU version according to your computational resources).

   GPU version run::
        
         conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   
   CPU version run::
        
         conda install pytorch torchvision torchaudio cpuonly -c pytorch
   
7. Install scikit-learn/pandas/matplotlib/numpy/seaborn/tqdm/Jupyter notebook::

   
        pip install scikit-learn
        pip install pandas
        pip install matplotlib
        pip install numpy
        pip install seaborn
        pip install tqdm
        pip install notebook
   
   
8. You are now ready to explore the codes/models! Please remember to check the ``src/MAIN.ipynb`` first.

   
**Note**: 
To install Anaconda, please follow its `official guideline <https://docs.anaconda.com/anaconda/user-guide/getting-started/>`_. For example, to install Anaconda3 on Linux, check `the Linux doc <https://docs.anaconda.com/anaconda/install/linux/>`_; to install Anaconda3 on Windows, check `the Windows doc <https://docs.anaconda.com/anaconda/install/windows/>`_; and to install Anaconda3 on macOS, check `the Mac doc <https://docs.anaconda.com/anaconda/install/mac-os>`_. We test our model on Ubuntu, Windows, and macOS.


Dependencies
-----------------

        pytorch

        scikit-learn

        pandas

        matplotlib

        numpy

        seaborn

        tqdm

        notebook
