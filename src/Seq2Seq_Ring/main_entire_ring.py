#------ import packages ------#
from seq2seq_model import *
from seq2seq_train import *
from seq2seq_prediction import *
from ecal_dataset_prep import *



input_len = 24
output_len = 24
stride = output_len
learning_rate = 1e-3
n_epochs = 200
print_step = 1
batch_size = 128
if output_len>=48: batch_size = 32
opt_alg = 'adam'
train_strategy = 'teacher_forcing' #"Please select one of them---[recursive, teacher_forcing, mixed]!"
teacher_forcing_ratio = 0.5 # please set it in the range of [0,1]
print("teacher_forcing_ratio = {}".format(teacher_forcing_ratio))
hidden_size = 1024
num_layers = 2
gpu_id = 0
crystal_id_start = 54000
crystal_id_end = 54000
verbose = False

training_year = 2016

device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

folder_name = 'LSTM_{}_IW_{}_OW_{}_LR_{}_ID_{}_{}_train_year_{}'.format(hidden_size, input_len, output_len, learning_rate,crystal_id_start,crystal_id_end,training_year)


# folder to save figures
save_dir_vis_data = '{}/vis_data/'.format(folder_name)

# folder to save models
save_dir_models = '{}/models/'.format(folder_name)

# folders for case1
save_dir_case1_fig= '{}/case1_fig/'.format(folder_name)
save_dir_case1_csv= '{}/case1_csv/'.format(folder_name)

# folders for case2
save_dir_case2_fig= '{}/case2_fig/'.format(folder_name)
save_dir_case2_csv= '{}/case2_csv/'.format(folder_name)

dir_list = [save_dir_vis_data, save_dir_case1_fig, save_dir_case1_csv, save_dir_case2_fig, save_dir_case2_csv,save_dir_models]
for cur_dir in dir_list:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
        print('>>> {} has been created successfully!'.format(cur_dir))
    else:
        print('>>> {} is exist!'.format(cur_dir))


### Visualize the datasets (only show the calibration curve).

# for train_file_2016/2017

X_train_all = None
Y_train_all = None

for crystal_id in range(crystal_id_start,crystal_id_end+1):
    try:
        fig_name_cali = os.path.join(save_dir_vis_data, '{}_cali_original_ID_{}.png'.format(training_year,crystal_id))
        fig_name_scaled_cali = os.path.join(save_dir_vis_data, '{}_cali_scaled_ID_{}.png'.format(training_year,crystal_id))
        train_file_2016 = '../0_Dataset/interim/df_skimmed_xtal_{}_2016.csv'.format(crystal_id)
        test_file_2017 = '../0_Dataset/interim/df_skimmed_xtal_{}_2017.csv'.format(crystal_id)
        test_file_2018 = '../0_Dataset/interim/df_skimmed_xtal_{}_2018.csv'.format(crystal_id)


        if training_year == 2016:
            ecal_dataset_prep_train_2016 = ECAL_Dataset_Prep(train_file_2016, 
                                                            input_len, 
                                                            output_len, 
                                                            stride, 
                                                            fig_name_cali, 
                                                            fig_name_scaled_cali,
                                                            verbose)
            ecal_dataset_prep_train_2016.start_processing()

            X_train = ecal_dataset_prep_train_2016.torch_X
            Y_train = ecal_dataset_prep_train_2016.torch_Y
        elif training_year == 2017:
            ecal_dataset_prep_train_2017 = ECAL_Dataset_Prep(test_file_2017, 
                                                            input_len, 
                                                            output_len, 
                                                            stride, 
                                                            fig_name_cali, 
                                                            fig_name_scaled_cali,
                                                            verbose)
            ecal_dataset_prep_train_2017.start_processing()

            X_train = ecal_dataset_prep_train_2017.torch_X
            Y_train = ecal_dataset_prep_train_2017.torch_Y
        else:
            print("please use 2016 or 2017 to train the model")

        if X_train_all == None:
            X_train_all = X_train
            Y_train_all = Y_train
        else:
            X_train_all = torch.cat( (X_train_all,X_train), dim=1 )
            Y_train_all = torch.cat( (Y_train_all,Y_train), dim=1 )
    except Exception as e:
        print("skip ID {}".format(crystal_id))

lstm_encoder = LSTM_Encoder(input_size=X_train_all.shape[2], hidden_size=hidden_size, num_layers=num_layers)
lstm_decoder = LSTM_Decoder(input_size=Y_train_all.shape[2], hidden_size=hidden_size, num_layers=num_layers)
lstm_encoder.to(device)
lstm_decoder.to(device)
print(lstm_encoder)
print(lstm_decoder)

loss_figure_name = os.path.join(save_dir_vis_data, '0_loss.png')
target_len = output_len
seq2sqe_train = Seq2Seq_Train(lstm_encoder,
                              lstm_decoder,
                              X_train_all,
                              Y_train_all,
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
seq2sqe_train.start_train()

#after training, we also want to save our models
model_file_name = os.path.join(save_dir_models, 'lstm_encoder.pt')
save_model(lstm_encoder.eval(), model_file_name)
model_file_name = os.path.join(save_dir_models, 'lstm_decoder.pt')
save_model(lstm_decoder.eval(), model_file_name)

