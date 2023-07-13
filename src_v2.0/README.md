# DNN Models

## Overview 

This folder contains Python code/Jupyter notebooks for Seq2Seq models, including:

- **MAIN.ipynb**: Jupyter notebook to process data, train the Seq2Seq model (for single xtal), and make prediction.
- **MAIN_multipleXtal.ipynb**: Jupyter notebook to process data, train the Seq2Seq model (for multiple xtals), and make prediction. 
- **ecal_dataset_prep.py**: Script to prepare dataset for Seq2Seq training.
- **Processing_Results.py**: Script to collect and analyze the results.
- **seq2seq_model.py**: Script to define the Seq2Seq model.     
- **seq2seq_prediction.py**: Script to make predictions on data. 
- **seq2seq_train.py**: Script to train the Seq2Seq model.
- **util.py**: Script includes additional helper functions.

## The Seq2Seq Model

We formulate the problem as a [sequence modeling problem](https://towardsdatascience.com/introduction-to-sequence-modeling-problems-665817b7e583) in which we want the Seq2Seq model to learn a function that can map the input sequence to the output sequence. The detailed design of our Seq2Seq model can be found in the [complementary document](https://github.com/FAIR-UMN/FAIR-UMN-ECAL). Here, we merely briefly describe the important hyper-parameters we used in training the Seq2Seq models. 

- *input_len*: the time steps (sequence length) for input data;
- *output_len*: the time steps (sequence length) for output data;
- *stride*: the stide of the sequence/window (default: output_len);
- *learning_rate*: the learning rate for our model;
- *n_epochs*: the maximum epoch to train our model;
- *print_step*: we print the training information per “print_step” epoch;
- *batch_size*: the batch size to train our model;
- *opt_alg*: the name of the optimization function (one should select one from {adam’, ‘sgd’});
- *train_strategy*: different training strategies (one should select one from {‘recursive’, ‘teacher_forcing’, ‘mixed’});
- *teacher_forcing_ratio*: it is a float number in the range of 0-1; it will be ignored when train_strategy=‘recursive’;
- *hidden_size*: the number of features in the hidden state;
- *num_layers*: the number of recurrent layers;
- *gpu_id*: the gpu id;
- *train_file*: the training csv file;
- *val_file*: the validation csv file;
- *test_file*: the test csv file;
- *crystal_id*: the crystal’s id;
- *verbose*:set it to be True if print information is desired; otherwise, set it to False; default (False).

For more details about the neural network model we used and its performance results, please check our [complementary document](https://github.com/FAIR-UMN/FAIR-UMN-ECAL).

## V2.0 Update

- **Data preprocessing**: remove data which has small luminosity (i.e., remove luminosity <= some threshold epsilon) so that the calibration recovery parts will be removed.

- **Data visualization**: improve the data visualization code to clearly separate the data in luminosity recovery part from the entire sequence.

## Future Work

- Try different initialization strategy to find a better minimizer.

- Conduct more experiments to determine the best iteration number, window size, and batch size for the v2.0 data (after removing calibration recovery parts).