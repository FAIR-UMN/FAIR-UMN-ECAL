# This script is used to prepare the dataset we need
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({'font.size': 17})

# May 8: test remove 0 luminosity data
DEBUG = True

# This class is used to generate the ECAL-Dataset
# And its input is a csv file name
class ECAL_Dataset_Prep:
    def __init__(self, csv_file, input_len, output_len, stride, fig_name_cali, fig_name_scaled_cali, verbose=False, lumi_threshold=0.0):
        self.csv_file = csv_file
        self.il = input_len
        self.ol = output_len
        self.stride = stride  # to ensure there is no overlap in prediction, please set the stride = output_window
        self.fig_name_cali = fig_name_cali
        self.fig_name_scaled_cali = fig_name_scaled_cali
        self.verbose = verbose # True: print information; False: will not print information

        # the scalers
        self.scaler_cali = StandardScaler()
        self.scaler_lumi = StandardScaler()
        # the dataframe version
        self.df_cali = None
        self.df_lumi = None
        # the numpy version
        self.np_cali = None
        self.np_lumi = None
        # the tensor/torch version
        self.torch_cali = None
        self.torch_lumi = None
        # the numpy version of input and target samples
        self.np_X = None
        self.np_Y = None
        # the tensor/torch version of input and target samples
        self.torch_X = None
        self.torch_Y = None
        self.lumi_threshold = lumi_threshold

    def start_processing(self):
        # now, we will call functions one by one to update the values
        self.get_df()
        self.normalize_dataset()
        self.sequence_dataset()
        self.visualize_data_samples()

    # get the dataframe and split the dataset into two dataframes:
    # one for calibration
    # one for luminosity diff
    def get_df(self):
        self.df = pd.read_csv(self.csv_file, index_col=0)
        self.df.index = pd.to_datetime(self.df['laser_datetime'])
        try:
            self.df_cali = self.df[['calibration']].copy()
            self.df_lumi = self.df[['delta_lumi']].copy()
            if DEBUG:
                # TODO: Problem with normalization
                lumi_threshold = self.lumi_threshold # max(self.df_lumi['delta_lumi'])*0.0
                self.df_cali = self.df_cali[(self.df_lumi>lumi_threshold)['delta_lumi']]
                self.df_lumi = self.df_lumi[(self.df_lumi>lumi_threshold)['delta_lumi']]
                pass

        except:
            assert False, "We except the csv should at least include ['calibration', 'delta_lumi'] (even they are empty columns)!"

    # normalize the dataset and also get other numpy and torch version
    def normalize_dataset(self):
        # normalize calibration
        if len(self.df_cali) !=0:
            self.scaler_cali.fit(self.df_cali[['calibration']])
            self.df_cali['calibration_scaled'] = None
            self.df_cali.loc[:,'calibration_scaled'] = self.scaler_cali.transform(self.df_cali[['calibration']])
            if self.verbose:
                print(self.df_cali.describe())
            self.np_cali = self.df_cali['calibration_scaled'].to_numpy()
            self.np_cali = self.np_cali.reshape(-1, 1)

        # normalize luminosity diff
        if len(self.df_lumi) !=0:
            self.scaler_lumi.fit(self.df_lumi[['delta_lumi']])
            self.df_lumi['delta_lumi_scaled'] = None
            self.df_lumi.loc[:,'delta_lumi_scaled'] = self.scaler_lumi.transform(self.df_lumi[['delta_lumi']])
            if self.verbose:
                print(self.df_lumi.describe())
            self.np_lumi = self.df_lumi['delta_lumi_scaled'].to_numpy()
            self.np_lumi = self.np_lumi.reshape(-1, 1)

    def sequence_dataset(self):
        # please note that: we arrange features in this order:
        # the first feature is lumi_diff, which is the "luminosity delta"
        # the second feature is cali, which is the "calibration"
        num_lumi = self.np_lumi.shape[0]
        num_cali = self.np_cali.shape[0]

        # self.seperate_decreasing_parts()

        #case: we have the same number of cali & lumi
        if num_lumi == num_cali and num_lumi!=0:
            num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
            #here, we want to combine the luminosity and calibration as our input
            y1_combined = np.hstack((self.np_lumi, self.np_cali))
            num_features_combined = y1_combined.shape[1]
            X = np.zeros([self.il, num_samples, num_features_combined])
            Y = np.zeros([self.ol, num_samples, num_features_combined])

            #processing X---input samples
            for ii in np.arange(num_samples):
                start_x = self.stride * ii
                end_x = start_x + self.il
                X[:, ii, :] = y1_combined[start_x:end_x, :]
            self.np_X = X
            self.torch_X = torch.from_numpy(X).type(torch.Tensor)

            ### processing Y---target samples
            for ii in np.arange(num_samples):
                start_y = self.stride * ii + self.il
                end_y = start_y + self.ol
                Y[:, ii, :] = y1_combined[start_y:end_y, :]
            self.np_Y = Y
            self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)
            pass

        # elif num_lumi>0 and num_cali==0:
        #     num_samples = (num_lumi- self.il - self.ol) // self.stride + 1
        #     Y = np.zeros([self.ol, num_samples, 1])
        #     ### processing Y---target samples
        #     for ii in np.arange(num_samples):
        #         start_y = self.stride * ii + self.il
        #         end_y = start_y + self.ol
        #         Y[:, ii, :] = self.np_lumi[start_y:end_y, :]
        #     self.np_Y = Y
        #     self.torch_Y = torch.from_numpy(Y).type(torch.Tensor)

        # else:
        #     assert False, 'We only support two cases: 1) Cali and Lumi. have the same length; 2) We only have Lumi.!'


    def visualize_data_samples(self):

        fig_title ='removing 0 luminosity comparison'
        fig_name = self.fig_name_cali[:-4] + 'comparison.png' 
        self.plot_cali_lumi_comparison([self.df[['calibration']],self.df_cali['calibration']], [self.df[['delta_lumi']],self.df_lumi['delta_lumi']], [self.df.index,self.df_lumi.index], fig_name, fig_title)

        fig_title ='After normalization: Mean={}; Std={}'.format(round( self.df_cali['calibration_scaled'].mean(), 3), round( self.df_cali['calibration_scaled'].std(), 3))

        self.plot_cali_lumi(self.df_cali['calibration_scaled'], self.df_lumi['delta_lumi_scaled'], self.df_cali.index, self.fig_name_scaled_cali, fig_title)

        pass

    def plot_cali_lumi_comparison(self, target_arr, lumi_info_arr, time_info_arr, fig_name, fig_title):



        #### double Y figure
        fig, ax1 = plt.subplots(figsize=(16, 9))  # fig, ax1 = plt.subplots(figsize=(25, 5))
        plt.title(fig_title, fontsize=40)
        plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=25)
        plot1 = ax1.plot(time_info_arr[0], target_arr[0], color='b', linewidth=3,linestyle='None', marker = 'o', markersize = 4, label='Calibration (true)')

        #after
        plot2 = ax1.plot(time_info_arr[1], target_arr[1], color='red', linewidth=3, linestyle='None', marker = 'o', markersize = 3, label='Calibration (true) after')

        ax1.set_ylabel('Calibration', fontsize=35)
        ax1.yaxis.label.set_color('b')
        ax1.set_xlabel('Time Info', fontsize=35)
        # ax1.set_ylim(0.7, 1)
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ### now, start the other y plot
        ax2 = ax1.twinx()
        plot3 = ax2.plot(time_info_arr[0], lumi_info_arr[0], label='Luminosity', color='yellow',linestyle='None', marker = 'o', markersize = 2, linewidth=2)

        # after
        plot4 = ax2.plot(time_info_arr[1], lumi_info_arr[1], label='Luminosity after', color='green', linestyle='None', marker = 'o', markersize = 1, linewidth=2)

        ax2.set_ylabel('Luminosity', fontsize=35)
        ax2.yaxis.label.set_color('grey')
        # ax2.set_ylim(0, 0.08)
        # ax2.set_xlim(1966, 2014.15)
        ax2.tick_params(axis='y', labelsize=25)
        for tl in ax2.get_yticklabels():
            tl.set_color('grey')

        # lines = plot1 + plot3
        lines = plot1 + plot2 + plot3 + plot4
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.tight_layout()

        # plt.title(fig_title, fontsize=40)
        # plt.xlabel(fontsize=35)
        # plt.ylabel(fontsize=35)
        # plt.legend(fontsize=28)

        plt.savefig(fig_name, dpi=300)
        # plt.show()
        plt.close()

    def plot_cali_lumi(self, target, lumi_info, time_info, fig_name, fig_title):
        #### double Y figure
        fig, ax1 = plt.subplots(figsize=(16, 9))  # fig, ax1 = plt.subplots(figsize=(25, 5))
        plt.title(fig_title, fontsize=40)
        plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=25)
        plot1 = ax1.plot(time_info, target, color='red', linewidth=3,linestyle='None', marker = 'o', markersize = 4, label='Calibration (true)')
        ax1.set_ylabel('Calibration', fontsize=35)
        ax1.yaxis.label.set_color('b')
        ax1.set_xlabel('Time Info', fontsize=35)
        # ax1.set_ylim(0.7, 1)
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ### now, start the other y plot
        ax2 = ax1.twinx()
        plot3 = ax2.plot(time_info, lumi_info, label='Luminosity', color='green',linestyle='None', marker = 'o', markersize = 2, linewidth=2)
        ax2.set_ylabel('Luminosity', fontsize=35)
        ax2.yaxis.label.set_color('grey')
        # ax2.set_ylim(0, 0.08)
        # ax2.set_xlim(1966, 2014.15)
        ax2.tick_params(axis='y', labelsize=25)
        for tl in ax2.get_yticklabels():
            tl.set_color('grey')

        lines = plot1 + plot3
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.tight_layout()

        # plt.title(fig_title, fontsize=40)
        # plt.xlabel(fontsize=35)
        # plt.ylabel(fontsize=35)
        # plt.legend(fontsize=28)

        plt.savefig(fig_name, dpi=300)
        # plt.show()
        plt.close()