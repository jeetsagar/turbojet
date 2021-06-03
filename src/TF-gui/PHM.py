# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PHM.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import matplotlib.pyplot as plt 

from PyQt5 import QtCore, QtGui, QtWidgets

import h5py
import bisect
import numpy as np

from matplotlib.figure import Figure

import tfdata
from tfdata import get_dataset

import pandas as pd

import tftrain
from tftrain import trainModel

from tftest import InferModel
import matplotlib.image as mpimg

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Ui_TrainTestSelect(object):
    def setupUi(self, TrainTestSelect):
        TrainTestSelect.setObjectName("TrainTestSelect")
        TrainTestSelect.resize(735, 710)
        TrainTestSelect.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.groupBox = QtWidgets.QGroupBox(TrainTestSelect)
        self.groupBox.setGeometry(QtCore.QRect(400, 30, 231, 231))
        self.groupBox.setObjectName("groupBox")

        self.BatchSize = QtWidgets.QLabel(self.groupBox)
        self.BatchSize.setGeometry(QtCore.QRect(10, 30, 71, 31))
        self.BatchSize.setObjectName("BatchSize")
        
        self.Epochs = QtWidgets.QLabel(self.groupBox)
        self.Epochs.setGeometry(QtCore.QRect(10, 70, 71, 31))
        self.Epochs.setObjectName("Epochs")
        
        self.LR = QtWidgets.QLabel(self.groupBox)
        self.LR.setGeometry(QtCore.QRect(10, 110, 101, 31))
        self.LR.setObjectName("LR")
        
        self.Dropout = QtWidgets.QLabel(self.groupBox)
        self.Dropout.setGeometry(QtCore.QRect(10, 150, 101, 31))
        self.Dropout.setObjectName("Dropout")
        
        self.BatchSizeHolder = QtWidgets.QTextEdit(self.groupBox)
        self.BatchSizeHolder.setGeometry(QtCore.QRect(110, 30, 71, 31))
        self.BatchSizeHolder.setObjectName("BatchSizeHolder")
        
        self.EpochsHolder = QtWidgets.QTextEdit(self.groupBox)
        self.EpochsHolder.setGeometry(QtCore.QRect(110, 70, 71, 31))
        self.EpochsHolder.setObjectName("EpochsHolder")
        
        self.LRHolder = QtWidgets.QTextEdit(self.groupBox)
        self.LRHolder.setGeometry(QtCore.QRect(110, 110, 71, 31))
        self.LRHolder.setObjectName("LRHolder")
        
        self.DropoutHolder = QtWidgets.QTextEdit(self.groupBox)
        self.DropoutHolder.setGeometry(QtCore.QRect(110, 150, 71, 31))
        self.DropoutHolder.setObjectName("DropoutHolder")
        
        self.groupBox_2 = QtWidgets.QGroupBox(TrainTestSelect)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 30, 271, 231))
        self.groupBox_2.setObjectName("groupBox_2")
        
        self.AugmentPhysics = QtWidgets.QCheckBox(self.groupBox_2)
        self.AugmentPhysics.setGeometry(QtCore.QRect(60, 30, 161, 21))
        self.AugmentPhysics.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.AugmentPhysics.setTristate(False)
        self.AugmentPhysics.setObjectName("AugmentPhysics")
        
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setGeometry(QtCore.QRect(60, 130, 111, 23))
        self.radioButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton.setObjectName("radioButton")
        
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_2.setGeometry(QtCore.QRect(60, 160, 111, 23))
        self.radioButton_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton_2.setChecked(True)
        self.radioButton_2.setObjectName("radioButton_2")
        
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(60, 190, 111, 23))
        self.radioButton_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton_3.setObjectName("radioButton_3")
        
        self.AugmentPhysics_2 = QtWidgets.QCheckBox(self.groupBox_2)
        self.AugmentPhysics_2.setGeometry(QtCore.QRect(60, 60, 161, 21))
        self.AugmentPhysics_2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.AugmentPhysics_2.setTristate(False)
        self.AugmentPhysics_2.setObjectName("AugmentPhysics_2")
        
        self.AugmentPhysics_3 = QtWidgets.QCheckBox(self.groupBox_2)
        self.AugmentPhysics_3.setGeometry(QtCore.QRect(60, 90, 161, 21))
        self.AugmentPhysics_3.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.AugmentPhysics_3.setTristate(False)
        self.AugmentPhysics_3.setObjectName("AugmentPhysics_3")
        
        self.pushButton_3 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_3.setGeometry(QtCore.QRect(520, 320, 111, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.InferClicked)
        
        self.pushButton_4 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_4.setGeometry(QtCore.QRect(520, 360, 111, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.TrainTestEvalClicked)
        
        self.pushButton_5 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_5.setGeometry(QtCore.QRect(520, 440, 111, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.LoadRealTimeClicked)
        
        self.pushButton_6 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_6.setGeometry(QtCore.QRect(200, 270, 111, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.LoadLotClicked)
        
        self.pushButton_7 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_7.setGeometry(QtCore.QRect(520, 270, 111, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(self.TrainInsightsClicked)
        
        self.pushButton_8 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_8.setGeometry(QtCore.QRect(520, 400, 111, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.clicked.connect(self.DeploymentClicked)
        
        self.pushButton_9 = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton_9.setGeometry(QtCore.QRect(520, 480, 111, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(self.ExitClicked)
        
        self.tableWidget = QtWidgets.QTableWidget(TrainTestSelect)
        self.tableWidget.setGeometry(QtCore.QRect(40, 320, 471, 360))
        self.tableWidget.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setHighlightSections(False)

        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setObjectName("tableWidget")

        self.tableWidget.horizontalHeader().setDefaultSectionSize(90)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(57)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setHighlightSections(False)
        self.tableWidget.verticalHeader().setStretchLastSection(True)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0,item)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1,item)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2,item)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3,item)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4,item)



        self.pushButton = QtWidgets.QPushButton(TrainTestSelect)
        self.pushButton.setGeometry(QtCore.QRect(400, 270, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.TrainClicked)

        self.retranslateUi(TrainTestSelect)
        QtCore.QMetaObject.connectSlotsByName(TrainTestSelect)

    def retranslateUi(self, TrainTestSelect):
        _translate = QtCore.QCoreApplication.translate
        TrainTestSelect.setWindowTitle(_translate("TrainTestSelect", "NASA-Turbojet-PHM"))
        self.groupBox.setTitle(_translate("TrainTestSelect", "Training Hyperparameters"))
        self.BatchSize.setText(_translate("TrainTestSelect", "Batch Size"))
        self.Epochs.setText(_translate("TrainTestSelect", "Epochs"))
        self.LR.setText(_translate("TrainTestSelect", "Learning Rate"))
        self.Dropout.setText(_translate("TrainTestSelect", "Dropout"))
        self.BatchSizeHolder.setHtml(_translate("TrainTestSelect", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1024</p></body></html>"))
        self.EpochsHolder.setHtml(_translate("TrainTestSelect", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">60</p></body></html>"))
        self.LRHolder.setHtml(_translate("TrainTestSelect", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.0001</p></body></html>"))
        self.DropoutHolder.setHtml(_translate("TrainTestSelect", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.2</p></body></html>"))
        self.groupBox_2.setTitle(_translate("TrainTestSelect", "Training Configuration"))
        self.AugmentPhysics.setText(_translate("TrainTestSelect", "Augment Physics"))
        self.radioButton.setText(_translate("TrainTestSelect", "BaseCNN      "))
        self.radioButton_2.setText(_translate("TrainTestSelect", "CNN-LSTM    "))
        self.radioButton_3.setText(_translate("TrainTestSelect", "CNN-2LSTM "))
        self.AugmentPhysics_2.setText(_translate("TrainTestSelect", "Batch Normalisation"))
        self.AugmentPhysics_3.setText(_translate("TrainTestSelect", "Resume Traning"))
        self.pushButton_3.setText(_translate("TrainTestSelect", "Infer"))
        self.pushButton_4.setText(_translate("TrainTestSelect", "Train-Test Eval"))
        self.pushButton_5.setText(_translate("TrainTestSelect", "Load RealTime"))
        self.pushButton_6.setText(_translate("TrainTestSelect", "Load Data"))
        self.pushButton_7.setText(_translate("TrainTestSelect", "Train Insights"))
        self.pushButton_8.setText(_translate("TrainTestSelect", "Deployment"))
        self.pushButton_9.setText(_translate("TrainTestSelect", "Exit"))
        self.pushButton.setText(_translate("TrainTestSelect", "Train"))


        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog","Unit"))

        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog","IsTrain"))

        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog","RMSE"))

        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog","s x 10^6"))

        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog","Deploy Cycle"))

    def LoadLotClicked(self):
        
        if self.AugmentPhysics.checkState():
            augmentPhy = 1
            fpath = "./data/DS02-KalmanNew.h5"
            self.attrs = 49
        else:
            augmentPhy = 0
            fpath = "./data/N-CMAPSS_DS02-006.h5"
            self.attrs = 35

        self.fpath = fpath

        with h5py.File(fpath,'r') as hdf:
            A_dev = np.array(hdf.get("A_dev"))
            A_test = np.array(hdf.get("A_test"))
            self.y_dev = np.array(hdf.get("Y_dev"))
            self.y_test = np.array(hdf.get("Y_test"))

        unit_devarray = np.array(A_dev[:, 0], dtype=np.int32)
        unit_testarray = np.array(A_test[:, 0], dtype=np.int32)

        self.unit_devarray = unit_devarray
        self.unit_testarray = unit_testarray

        self.unit_devunique = list(np.unique(unit_devarray))
        self.unit_testunique = list(np.unique(unit_testarray))
        print(self.unit_testunique)
        print(self.unit_devunique)


        self.tftrain_ds = get_dataset(fpath, [], augmentPhy, 1)
        self.tftest_ds = get_dataset(fpath, [], augmentPhy, 0)

        self.updateTable()

    def LoadRealTimeClicked(self):
        suffix = "test"
        fpath = "./data/N-CMAPSS_DS02-006.h5"
        with h5py.File(fpath,'r') as hdf:
             W_in = np.array(hdf.get("W_"+suffix))
             X_s_in = np.array(hdf.get("X_s_"+suffix))
             X_v_in = np.array(hdf.get("X_v_"+suffix))
             T_in = np.array(hdf.get("T_"+suffix))


        fig, ax = plt.subplots(4,4)
        fig.suptitle("Real Time Sensor Measurements",fontsize=20)
        fig.text(0.5, 0.04, 'Time (seconds)', ha='center',fontsize = 20)
        fig.text(0.04, 0.5, 'Sensor readings (units)', va='center', rotation='vertical',fontsize = 20)

        ypred_test = self.ypred_test

        for t in range(10000):
            W_in_curr = np.array(W_in[t:t+50,:])
            X_s_in_curr = np.array(X_s_in[t:t+50,:])
            X_v_in_curr = np.array(X_v_in[t:t+50,:])
            T_in_curr = np.array(T_in[t:t+50])
            Ypred_in = np.array(ypred_test[t:t+50])



            ax[0,0].set_title("Altitude (ft)")
            ax[0,0].plot(W_in_curr[:,0])
            ax[0,0].set_xticklabels([])

            ax[0,1].set_title("Mach Number (-)")
            ax[0,1].plot(W_in_curr[:,1])
            ax[0,1].set_xticklabels([])

            ax[0,2].set_title("Throttle-Resolver Angle (%)")
            ax[0,2].plot(W_in_curr[:,2])
            ax[0,2].set_xticklabels([])

            ax[0,3].set_title("Temp. Fan Inlet (R)")
            ax[0,3].plot(W_in_curr[:,3])
            ax[0,3].set_xticklabels([])

            ax[1,0].set_title("Fuel Flow (pps)")
            ax[1,0].plot(X_s_in_curr[:,0])
            ax[1,0].set_xticklabels([])

            ax[1,1].set_title("Fan Speed (rpm)")
            ax[1,1].plot(X_s_in_curr[:,1])
            ax[1,1].set_xticklabels([])

            ax[1,2].set_title("Core Speed (rpm)")
            ax[1,2].plot(X_s_in_curr[:,2])
            ax[1,2].set_xticklabels([])

            ax[1,3].set_title("LPC outlet Temp (R)")
            ax[1,3].plot(X_s_in_curr[:,3])
            ax[1,3].set_xticklabels([])
            
            ax[2,0].set_title("HPC outlet Temp (R)")
            ax[2,0].plot(X_s_in_curr[:,4])
            ax[2,0].set_xticklabels([])
            
            
            ax[2,1].set_title("HPT outlet Temp (R)")
            ax[2,1].plot(X_s_in_curr[:,5])
            ax[2,1].set_xticklabels([])
            
            ax[2,2].set_title("LPT outlet Temp (R)")
            ax[2,2].plot(X_s_in_curr[:,6])
            ax[2,2].set_xticklabels([])

            ax[2,3].set_title("Bypass-duct Pressure (psia)")
            ax[2,3].plot(X_s_in_curr[:,7])
            ax[2,3].set_xticklabels([])

            ax[3,0].set_title("Fan outlet Pressure (psia)")
            ax[3,0].plot(X_s_in_curr[:,8])
            
            ax[3,1].set_title("LPC outlet Pressure (psia)")
            ax[3,1].plot(X_s_in_curr[:,9])

            ax[3,2].set_title("HPC outlet Pressure (psia)")
            ax[3,2].plot(X_s_in_curr[:,10])


            ax[3,3].set_title("RUL (cycles) current = " + "{0:.2f}".format(Ypred_in[-1]) )
            if(Ypred_in[0]>15):
                ax[3,3].plot(Ypred_in,"g")
            else:
                ax[3,3].plot(Ypred_in,"r")
            

            
 
            plt.pause(0.00001)
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()    
            for i in range(4):
                for j in range(4):
                    ax[i,j].cla()
        
        plt.show()
        return 
        for i in list_i:
            y = np.random.random()
            plt.plot(i)
            plt.pause(2)
            plt.cla()

        plt.show()

    def TrainInsightsClicked(self):
        fig = Figure()
        df_in = pd.read_csv("./losses/lossTrend.csv")
        plt.title("Loss Trend For Training",fontsize=20)
        plt.plot(list(df_in["Epochs"]),list(df_in["Loss"]))
        plt.xlabel("Epochs",fontsize=20)
        plt.ylabel("Loss",fontsize=20)
        plt.show()

    def __rmse__(self,pred,true):
        return np.sqrt(np.mean((pred-true)**2))

    def __nasafn__(self,pred,true):
        sum_in = 0
        for i in range(len(pred)):
            if pred[i]<true[i]:
                sum_in += np.exp((1/13)*(np.abs(pred[i]-true[i])))
            else:
                sum_in += np.exp((1/10)*(np.abs(pred[i]-true[i])))
        return sum_in/(10**6)

    def TrainTestEvalClicked(self):


        units = self.unit_devunique+self.unit_testunique
        arrys = list(self.unit_devarray)+list(self.unit_testarray)

        indexes = [arrys.index(x)-i*50  for i,x in enumerate(list(set(self.unit_devarray))+list(set(self.unit_testarray)))]

        indexes = indexes+[len(arrys)-len(units)*50]
        print(indexes)
        self.indexes = indexes

        c = 0
        ypred_train = np.loadtxt("./output/y_predtrain.out")
        ypred_test = np.loadtxt("./output/y_predtest.out")

        ytrue_train = np.loadtxt("./output/y_truetrain.out")
        ytrue_test = np.loadtxt("./output/y_truetest.out")


        ytrue = list(ytrue_train)+list(ytrue_test)
        ypred = list(ypred_train)+list(ypred_test)

        self.ytrue_test = list(ytrue_test)
        self.ypred_test = list(ypred_test)

        # ytrue = list(self.y_dev) + list(self.y_test)
        rmse_train = self.__rmse__(ypred_train,ytrue_train)
        nasa_train = self.__nasafn__(ypred_train,ypred_train)

        rmse_test = self.__rmse__(ypred_test,ytrue_test)
        nasa_test = self.__nasafn__(ypred_test,ytrue_test)

        fig, ax = plt.subplots(3,3)
        fig.suptitle("Unit Wise RUL Analysis",fontsize=20)
        fig.text(0.5, 0.04, 'Time (seconds)', ha='center',fontsize = 20)
        fig.text(0.04, 0.5, 'RUL (cycles)', va='center', rotation='vertical',fontsize = 20)

        rmses = []
        nasas = []

        for i in range(3):
            for j in range(3):
                if c>5:
                    title = "Unit " + str(units[c])+ " (Test) " 
                else:
                    title = "Unit " + str(units[c])+ " (Train) " 
                ax[i,j].set_title(title)
                ax[i,j].plot(ytrue[indexes[c]:indexes[c+1]],label="True",c="Green")
                ax[i,j].plot(ypred[indexes[c]:indexes[c+1]],label="Predicted",c="Blue")
                ax[i,j].legend()
                ax[i,j].ticklabel_format(scilimits=(0,5),useOffset=True)
                rmses.append(self.__rmse__(np.array(ypred[indexes[c]:indexes[c+1]]),np.array(ytrue[indexes[c]:indexes[c+1]])))
                nasas.append(self.__nasafn__(np.array(ypred[indexes[c]:indexes[c+1]]),np.array(ytrue[indexes[c]:indexes[c+1]])))
                c += 1

        self.rmse_final = list(rmses[:3]) + [rmse_test] + list(rmses[3:]) + [rmse_train]
        self.nasa_final = list(nasas[:3]) + [nasa_test] + list(nasas[3:]) + [nasa_train]

        for i in range(len(units)+2):
            self.tableWidget.setItem(i,2,QtWidgets.QTableWidgetItem("{0:.2f}".format(self.rmse_final[i])))
            self.tableWidget.setItem(i,3,QtWidgets.QTableWidgetItem("{0:.2f}".format(self.nasa_final[i])))


        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()



        X = pd.DataFrame()
        X["RMSE"] = self.rmse_final
        X["Nasa"] = self.nasa_final
        X.to_csv("./losses/Evals.csv",index=False)



    def DeploymentClicked(self):

        units = self.unit_testunique
        arrys = list(self.unit_testarray)


        indexes = [arrys.index(x)-i*50  for i,x in enumerate(list(set(self.unit_testarray)))]

        indexes.append(len(arrys)-len(units)*50)

        fig,ax = plt.subplots(3,1)
        fig.suptitle("Deployment Analysis of Test Units",fontsize=20)
        fig.text(0.5, 0.04, 'Cycles', ha='center',fontsize = 20)
        fig.text(0.04, 0.5, 'RUL Deviation (cycles)', va='center', rotation='vertical',fontsize = 20)

        ypred_test = np.loadtxt("./output/y_predtest.out")
        ytrue_test = np.loadtxt("./output/y_truetest.out")

        self.ypred_test = list(ypred_test)

        c = 0
        deploy_cycle = []
        for i in range(3):
            true_in = ytrue_test[indexes[c]:indexes[c+1]]
            pred_in = ypred_test[indexes[c]:indexes[c+1]]

            inds = [list(true_in).index(x) for x in ((set(list(true_in))))]
            inds.append(len(true_in))
            uniques = list(set(true_in))[::-1]


            ycap = []
            y = []
            # t = 0
            err = []
            err_max = []
            err_min = []
            for j in uniques:

                arr_p = pred_in[true_in==j]

                y_max = np.max(arr_p)
                y_min = np.min(arr_p)

                ycap_in = np.mean(arr_p)
                y_in = j

                err.append(y_in-ycap_in)
                err_max.append(y_in-y_max)
                err_min.append(y_in-y_min)

                y.append(y_in)
                ycap.append(ycap_in)

            
            cycle_in = 0
            for k in range(len(err)-1):
                if np.abs(err[-k-1])>8:
                    cycle_in = len(y)-k
                    break
            deploy_cycle.append(cycle_in)

            ax[i].set_title("Unit "+str(units[c]))
            ax[i].plot(err,'bo-')
            ax[i].plot(err_max,'g-')
            ax[i].plot(err_min,'g-')
            ax[i].axhline(y = 8, color = 'r', linestyle = '--')
            ax[i].axhline(y = -8, color = 'r', linestyle = '--')
            ax[i].axvline(x = cycle_in, color = 'black', linestyle = '--')
            c += 1


        
        for p,d in enumerate(deploy_cycle):
            self.tableWidget.setItem(p,4,QtWidgets.QTableWidgetItem(str(d)))
        self.tableWidget.setItem(len(deploy_cycle),4,QtWidgets.QTableWidgetItem("{0:.2f}".format(np.mean(deploy_cycle))))

        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        plt.show()




    def ExitClicked(self):
        exit()

    def InferClicked(self):

        if self.radioButton.isChecked():
            Architecture = "BaseCNN"
        elif self.radioButton_2.isChecked():
            Architecture = "CNN-LSTM"
        elif self.radioButton_3.isChecked():
            Architecture = "CNN-2LSTM"

        InferModel(self.tftest_ds,Architecture+str(self.attrs),0)
        InferModel(self.tftrain_ds,Architecture+str(self.attrs),1)






        

    def updateTable(self):
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(len(self.unit_devunique)+len(self.unit_testunique)+2)

        row_in = 0

        for i in range(len(self.unit_testunique)):
            self.tableWidget.setItem(row_in,0,QtWidgets.QTableWidgetItem(str(self.unit_testunique[i])))

            chkBoxItem = QtWidgets.QTableWidgetItem()
            chkBoxItem.setFlags(QtCore.Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(QtCore.Qt.Unchecked)
            self.tableWidget.setItem(row_in,1,chkBoxItem)

            row_in += 1

        self.tableWidget.setItem(row_in,0,QtWidgets.QTableWidgetItem("TestUnits"))
        self.tableWidget.setItem(row_in,1,QtWidgets.QTableWidgetItem("---"))

        
        row_in += 1

        for i in range(len(self.unit_devunique)):
            self.tableWidget.setItem(row_in,0,QtWidgets.QTableWidgetItem(str(self.unit_devunique[i])))
            chkBoxItem = QtWidgets.QTableWidgetItem()
            chkBoxItem.setFlags(QtCore.Qt.ItemIsEnabled)
            chkBoxItem.setCheckState(QtCore.Qt.Checked)
            self.tableWidget.setItem(row_in,1,chkBoxItem)

            row_in += 1
        
        self.tableWidget.setItem(row_in,0,QtWidgets.QTableWidgetItem("TrainUnits"))
        self.tableWidget.setItem(row_in,1,QtWidgets.QTableWidgetItem("---"))

    def TrainClicked(self):
        if self.radioButton.isChecked():
            Architecture = "BaseCNN"
        elif self.radioButton_2.isChecked():
            Architecture = "CNN-LSTM"
        elif self.radioButton_3.isChecked():
            Architecture = "CNN-2LSTM"

        if self.AugmentPhysics_3.checkState():
            resume_train  = 1
        else:
            resume_train = 0

        self.arch = Architecture

        params = {"Epochs":int(self.EpochsHolder.toPlainText()), "BatchSize":int(self.BatchSizeHolder.toPlainText()), "LearningRate":float(self.LRHolder.toPlainText()),"Dropout":float(self.DropoutHolder.toPlainText()),"Architecture":Architecture,"BatchNorm":self.AugmentPhysics_2.checkState(),"Attrs":self.attrs,"ResumeTraining":resume_train}
        print(params)
        trainModel(self.tftrain_ds,params)









if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TrainTestSelect = QtWidgets.QDialog()
    ui = Ui_TrainTestSelect()
    ui.setupUi(TrainTestSelect)
    TrainTestSelect.show()
    sys.exit(app.exec_())

