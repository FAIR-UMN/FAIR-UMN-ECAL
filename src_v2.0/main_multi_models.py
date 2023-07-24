from seq2seq_model import *
from seq2seq_train import *
from seq2seq_prediction import *
from ecal_dataset_prep import *


'''
This file is used to train models on different xtals: 1 seq2seq model to each xtal. 
'''

input_len = 24 # deafult 24
output_len = 24 # default 24
stride = output_len
learning_rate = 1e-3
n_epochs = 500 # Default:n_epochs = 200. Can be adjusted
print_step = 1
batch_size = 16 # batch_size = 128 if luminosity threshold = 0
if output_len>=48: batch_size = 32
opt_alg = 'adam'
train_strategy = 'teacher_forcing' #"Please select one of them---[recursive, teacher_forcing, mixed]!"
teacher_forcing_ratio = 0.5 # please set it in the range of [0,1]
hidden_size = 1024
num_layers = 2
gpu_id = 0
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
lumi_threshold = 2e7 # Default: lumi_threshold = 2e7. Maximum luminosity is around 1e8. Can be adjusted
verbose = False


for crystal_id in range(54180,54300,15):

    train_file_2016 = '../0_Dataset/interim/df_skimmed_xtal_{}_2016.csv'.format(crystal_id)
    test_file_2017 = '../0_Dataset/interim/df_skimmed_xtal_{}_2017.csv'.format(crystal_id)
    test_file_2018 = '../0_Dataset/interim/df_skimmed_xtal_{}_2018.csv'.format(crystal_id)

    parent_folder = 'LSTM_{}_IW_{}_OW_{}_LR_{}_Epochs_{}_ID_{}_lumi_threshold_{}'.format(hidden_size, input_len, output_len, learning_rate, n_epochs, crystal_id, lumi_threshold)

    # folder to save figures
    save_dir_vis_data = parent_folder + '/vis_data/' 

    # folder to save models
    save_dir_models = parent_folder + '/models/' 

    # folders for case1
    save_dir_case1_fig= parent_folder + '/case1_fig/'  
    save_dir_case1_csv= parent_folder + '/case1_csv/' 

    # folders for case2
    save_dir_case2_fig= parent_folder + '/case2_fig/'  
    save_dir_case2_csv= parent_folder + '/case2_csv/'

    dir_list = [save_dir_vis_data, save_dir_case1_fig, save_dir_case1_csv, save_dir_case2_fig, save_dir_case2_csv,save_dir_models]
    for cur_dir in dir_list:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
            print('>>> {} has been created successfully!'.format(cur_dir))
        else:
            print('>>> {} is exist!'.format(cur_dir))


    # for train_file_2016
    fig_name_cali = os.path.join(save_dir_vis_data, '2016_cali_original.png')
    fig_name_scaled_cali = os.path.join(save_dir_vis_data, '2016_cali_scaled.png')
    ecal_dataset_prep_train_2016 = ECAL_Dataset_Prep(train_file_2016, 
                                                    input_len, 
                                                    output_len, 
                                                    stride, 
                                                    fig_name_cali, 
                                                    fig_name_scaled_cali,
                                                    verbose,
                                                    lumi_threshold)
    ecal_dataset_prep_train_2016.start_processing()

    # for test_file_2017
    fig_name_cali = os.path.join(save_dir_vis_data, '2017_cali_original.png')
    fig_name_scaled_cali = os.path.join(save_dir_vis_data, '2017_cali_scaled.png')
    ecal_dataset_prep_test_2017 = ECAL_Dataset_Prep(test_file_2017, 
                                                    input_len, 
                                                    output_len, 
                                                    stride, 
                                                    fig_name_cali, 
                                                    fig_name_scaled_cali,
                                                    verbose,
                                                    lumi_threshold)
    ecal_dataset_prep_test_2017.start_processing()

    # for test_file_2018
    fig_name_cali = os.path.join(save_dir_vis_data, '2018_cali_original.png')
    fig_name_scaled_cali = os.path.join(save_dir_vis_data, '2018_cali_scaled.png')
    ecal_dataset_prep_test_2018 = ECAL_Dataset_Prep(test_file_2018, 
                                                    input_len, 
                                                    output_len, 
                                                    stride, 
                                                    fig_name_cali, 
                                                    fig_name_scaled_cali,
                                                    verbose,
                                                    lumi_threshold)
    ecal_dataset_prep_test_2018.start_processing()


    X_train = ecal_dataset_prep_train_2016.torch_X
    Y_train = ecal_dataset_prep_train_2016.torch_Y


    lstm_encoder = LSTM_Encoder(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    lstm_decoder = LSTM_Decoder(input_size=Y_train.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    lstm_encoder.to(device)
    lstm_decoder.to(device)
    print(lstm_encoder)
    print(lstm_decoder)


    loss_figure_name = os.path.join(save_dir_vis_data, '0_loss.png')
    target_len = output_len
    seq2seq_train = Seq2Seq_Train(lstm_encoder,
                                lstm_decoder,
                                X_train,
                                Y_train,
                                n_epochs,
                                target_len,
                                batch_size,
                                learning_rate,
                                opt_alg,
                                print_step,
                                train_strategy,
                                teacher_forcing_ratio,
                                device,
                                loss_figure_name)
    seq2seq_train.start_train()

    #after training, we also want to save our models
    model_file_name = os.path.join(save_dir_models, 'lstm_encoder.pt')
    save_model(lstm_encoder.eval(), model_file_name)
    model_file_name = os.path.join(save_dir_models, 'lstm_decoder.pt')
    save_model(lstm_decoder.eval(), model_file_name)


    # ### Case1 Prediction:  do not use predictions as input to help the next-round prediction

    # In[10]:


    # check its prediction on training data
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_train_2016.np_X
    Ytrain = ecal_dataset_prep_train_2016.np_Y
    df = ecal_dataset_prep_train_2016.df_lumi
    scaler_cali = ecal_dataset_prep_train_2016.scaler_cali
    year = '2016'
    test_case = 'case1'
    fig_name_mape = os.path.join(save_dir_case1_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case1_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case1_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()


    # In[11]:


    # check its prediction on test data-2017
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_test_2017.np_X
    Ytrain = ecal_dataset_prep_test_2017.np_Y
    df = ecal_dataset_prep_test_2017.df_lumi
    scaler_cali = ecal_dataset_prep_test_2017.scaler_cali
    year = '2017'
    test_case = 'case1'
    fig_name_mape = os.path.join(save_dir_case1_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case1_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case1_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()


    # In[12]:


    # check its prediction on test data-2018
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_test_2018.np_X
    Ytrain = ecal_dataset_prep_test_2018.np_Y
    df = ecal_dataset_prep_test_2018.df_lumi
    scaler_cali = ecal_dataset_prep_test_2017.scaler_cali
    year = '2018'
    test_case = 'case1'
    fig_name_mape = os.path.join(save_dir_case1_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case1_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case1_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()


    # ### Case2 Prediction: use predictions as input to help the next-round prediction.

    # In[13]:


    # check its prediction on training data
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_train_2016.np_X
    Ytrain = ecal_dataset_prep_train_2016.np_Y
    df = ecal_dataset_prep_train_2016.df_lumi
    scaler_cali = ecal_dataset_prep_train_2016.scaler_cali
    year = '2016'
    test_case = 'case2'
    fig_name_mape = os.path.join(save_dir_case2_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case2_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case2_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()


    # In[14]:


    # check its prediction on test data-2017
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_test_2017.np_X
    Ytrain = ecal_dataset_prep_test_2017.np_Y
    df = ecal_dataset_prep_test_2017.df_lumi
    scaler_cali = ecal_dataset_prep_test_2017.scaler_cali
    year = '2017'
    test_case = 'case2'
    fig_name_mape = os.path.join(save_dir_case2_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case2_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case2_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()


    # In[15]:


    # check its prediction on test data-2018
    # Please note that here, the data are in the numpy format, not the tensor format
    Xtrain = ecal_dataset_prep_test_2018.np_X
    Ytrain = ecal_dataset_prep_test_2018.np_Y
    df = ecal_dataset_prep_test_2018.df_lumi
    scaler_cali = ecal_dataset_prep_test_2017.scaler_cali
    year = '2018'
    test_case = 'case2'
    fig_name_mape = os.path.join(save_dir_case2_fig, '0_MAPE_{}_{}.png'.format(test_case,year))
    fig_name_mse = os.path.join(save_dir_case2_fig, '1_MSE_{}_{}.png'.format(test_case,year))
    metric_file = os.path.join(save_dir_case2_csv, '{}_{}.csv'.format(test_case,year))
    seq2seq_prediction = Seq2Seq_Prediction(lstm_encoder,
                                            lstm_decoder,
                                            Xtrain,
                                            Ytrain,
                                            df,
                                            scaler_cali,
                                            device,
                                            fig_name_mape,
                                            fig_name_mse,
                                            metric_file,
                                            test_case)
    seq2seq_prediction.start_prediction()



