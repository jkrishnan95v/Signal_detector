# RF Source_detection using 1D CNN

This project uses a CNN to perform Radio Signal source detection. A 20 element array is used to estimate the number of signals impinging it. The upper triangle of the received auto-correlation matrix is flattened into a one dimensional vector and fed as the input. The output is a one hot encoded vector which denotes the number of sources present.      

Estimation of the number of sources is an important step in various array processing technologies. It is a prerequisite in most DoA algorithms. This model was succesfull in recognisng upto 6 sources at 20dB of SNR. Output of the code includes plots of the training and validation error, stats(precision,sensitivity, and specificity), a confusion matrix, and the ROC plot of the test data.


Work ongoing - testing at various SNR and try to resolve more number of signals using alternate architectures. 
Work submitted for APS-URSI 2021 and for IEEE-TAP special issue on ML 

# Architecture 

![cnn](https://user-images.githubusercontent.com/20204692/132585407-62829f0a-67ff-4f49-8a07-6cc5ea4d5458.png)

# Training and Validation 

![accuracy_plots](https://user-images.githubusercontent.com/20204692/132585454-f3f2237c-0653-46bf-a915-103e76e97ac1.jpg)

# Results 

![conf_20db](https://user-images.githubusercontent.com/20204692/132585583-ef567dd6-ec27-4d19-8d3a-707343d8e87a.png)

