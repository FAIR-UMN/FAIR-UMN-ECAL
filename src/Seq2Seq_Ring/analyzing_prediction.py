#------ import packages ------#
from seq2seq_model import *
from seq2seq_train import *
from seq2seq_prediction import *
from ecal_dataset_prep import *
import csv
import numpy as np
import matplotlib.pyplot as plt



crystal_id_start = 54000
crystal_id_end = 54359



# save_dir_case1 = 'Jul5_recursive_LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54359'
# save_dir_case1 = 'Jul12_teacherforcing_ratio1_LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54359'
# save_dir_case1 = 'Jul12_teacherforcing_ratio_0.5_LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54359'
save_dir_case1 = 'Aug2_trainon_54000_single_xtal_LSTM_1024_IW_24_OW_24_LR_0.001_ID_54000_54359_train_year_2016'

save_dir_case1_csv = save_dir_case1 + '/case1_csv/'

MAPE_dict = {}
for test_case in ['case1','case2']:
    plt.cla()
    plt.clf()
# #     plt.gca().set_prop_cycle(None)
    plt.style.use('seaborn')
    plt.grid(linestyle='dotted')
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=40)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)


    plt.grid(axis="x")
    

# for test_case in ['case1']:
    for year in ['2016','2017','2018']:
        MAPE_key = test_case + year
        MAPE_dict[MAPE_key] = -1
        MAPEs = []
        for crystal_id in range(crystal_id_start,crystal_id_end+1):
            # print("{}_year_{}_ID_{}".format(test_case,year,crystal_id))
            try:
                csv_file_name = os.path.join(save_dir_case1_csv,'{}_{}_ID_{}.csv'.format(test_case,year,crystal_id))
                with open(csv_file_name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 1:
                            # print(f'Column names are {", ".join(row)}')
                            MAPE = float(row[0])
                            # print(MAPE)
                            MAPEs.append(MAPE)
                        line_count += 1
            except Exception as e:
                print("skip ID {}".format(crystal_id))
        MAPE_vec = np.array(MAPEs)

        plt.hist(MAPE_vec,bins=50,alpha=0.5,label=year)
        

        final_MAPE = np.mean(MAPE_vec)
        print("==============================================================")
        print("{}_year_{}: final MAPE = {}".format(test_case,year,final_MAPE))
        print("==============================================================")
        MAPE_dict[MAPE_key] = final_MAPE

    plt.xlabel('MAPE')
    plt.legend()
    
    fig_name = save_dir_case1 + '/prediction_' + test_case
    plt.savefig(fig_name)
    # plt.show()
    # plt.figure().clear()
    # plt.close()
    # plt.cla()
    # plt.clf()
    

    
    pass

for test_case in ['case1','case2']:
    for year in ['2016','2017','2018']:
        MAPE_key = test_case + year
        print("==============================================================")
        print("{}_year_{}: final MAPE = {}".format(test_case,year,MAPE_dict[MAPE_key]))
        print("==============================================================")
