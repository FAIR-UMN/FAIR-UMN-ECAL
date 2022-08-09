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
train_strategy = 'recursive' #"Please select one of them---[recursive, teacher_forcing, mixed]!"
teacher_forcing_ratio = 0.5 # please set it in the range of [0,1]
hidden_size = 1024
num_layers = 2
gpu_id = 0
crystal_id_start = 54000
crystal_id_end = 54359
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

dir_list = [save_dir_vis_data, save_dir_case1_fig, save_dir_case1_csv, save_dir_case2_fig, save_dir_case2_csv]
for cur_dir in dir_list:
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
        print('>>> {} has been created successfully!'.format(cur_dir))
    else:
        print('>>> {} is exist!'.format(cur_dir))
    
##################################################################################################################
# load model
lstm_encoder = LSTM_Encoder(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
lstm_decoder = LSTM_Decoder(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
lstm_encoder.to(device)
lstm_decoder.to(device)

model_file_name = os.path.join('LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54000_train_year_2016/models/', 'lstm_encoder.pt')
lstm_encoder.load_state_dict(torch.load(model_file_name))

model_file_name = os.path.join('LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54000_train_year_2016/models/', 'lstm_decoder.pt')
lstm_decoder.load_state_dict(torch.load(model_file_name))
##################################################################################################################

MAPE_dict = {}
# crystal_id_start = 54289
# for test_case in ['case1','case2']:
#     for year in ['2016','2017','2018']:
for test_case in ['case2']:
    for year in ['2018']:
        MAPE_key = test_case + year
        MAPE_dict[MAPE_key] = -1
        MAPEs = []
        lengths = []
        for crystal_id in range(crystal_id_start,crystal_id_end+1):
        
            try:
                print("{}_year_{}_ID_{}".format(test_case,year,crystal_id))
                # for test_file_201X
                fig_name_cali = os.path.join(save_dir_vis_data, '{}_cali_original_ID_{}.png'.format(year,crystal_id))
                fig_name_scaled_cali = os.path.join(save_dir_vis_data, '{}_cali_scaled_ID_{}.png'.format(year,crystal_id))

                test_file_201X = '../0_Dataset/interim/df_skimmed_xtal_{}_{}.csv'.format(crystal_id,year)

                ecal_dataset_prep_test_201X = ECAL_Dataset_Prep(test_file_201X, 
                                                                input_len, 
                                                                output_len, 
                                                                stride, 
                                                                fig_name_cali, 
                                                                fig_name_scaled_cali,
                                                                verbose)
                ecal_dataset_prep_test_201X.start_processing()


                # check its prediction on test data-201X
                # Please note that here, the data are in the numpy format, not the tensor format
                Xtrain = ecal_dataset_prep_test_201X.np_X
                Ytrain = ecal_dataset_prep_test_201X.np_Y
                df = ecal_dataset_prep_test_201X.df_lumi
                scaler_cali = ecal_dataset_prep_test_201X.scaler_cali
                
                
                fig_name_mape = os.path.join(save_dir_case1_fig, '0_MAPE_{}_{}_ID_{}.png'.format(test_case,year,crystal_id))
                fig_name_mse = os.path.join(save_dir_case1_fig, '1_MSE_{}_{}_ID_{}.png'.format(test_case,year,crystal_id))
                metric_file = os.path.join(save_dir_case1_csv, '{}_{}_ID_{}.csv'.format(test_case,year,crystal_id))

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
                [MAPE,length] = seq2seq_prediction.getAPE()
                print("MAPE = {}, length = {}".format(MAPE,length))
                MAPEs.append(MAPE)
                lengths.append(length)

                del seq2seq_prediction
                del ecal_dataset_prep_test_201X

        

            except Exception as e:
                print("skip ID {}".format(crystal_id))

        MAPE_vec = np.array(MAPEs)
        lengths_vec = np.array(lengths)
        final_MAPE = MAPE_vec@lengths_vec/np.sum(lengths_vec)
        print("==============================================================")
        print("{}_year_{}: final MAPE = {}".format(test_case,year,final_MAPE))
        print("==============================================================")
        MAPE_dict[MAPE_key] = final_MAPE

for test_case in ['case1','case2']:
    for year in ['2016','2017','2018']:
        MAPE_key = test_case + year
        print("==============================================================")
        print("{}_year_{}: final MAPE = {}".format(test_case,year,MAPE_dict[MAPE_key]))
        print("==============================================================")