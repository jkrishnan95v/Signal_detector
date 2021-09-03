# RF Source_detection using 1D CNN

This project uses a CNN to perform Radio Signal source detection. A 20 element array is used to estimate the number of signals impinging it. The upper triangle of the received auto-correlation matrix is flattened into a one dimensional vector and fed as the input. The output is a one hot encoded vector which denotes the number of sources present.      

Estimation of the number of sources is an important step in various array processing technologies. It is a prerequisite in most DoA algorithms. This model was succesfull in recognisng upto 6 sources at 20dB of SNR. Output of the code includes plots of the training and validation error, stats(precision,sensitivity, and specificity), a confusion matrix, and the ROC plot of the test data.


Work ongoing - testing at various SNR and try to resolve more number of signals using alternate architectures. 
Work submitted for APS-URSI 2021 and for IEEE-TAP special issue on ML 
