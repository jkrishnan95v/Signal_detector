#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:59:19 2020

@author: arjun
"""

import random
import numpy as np
from datagen import UpperTriFeatureExtract# D	#importing helper functions 
from datagen import DiscreteTargetSpectrum 
from numpy import asarray
from numpy import savetxt
random.seed(10)                                             #set seed for reproducibility

CARRIER_FREQUENCY = 1e9                                     #frequency of operation
CARRIER_WAVELENGTH = 3e8/CARRIER_FREQUENCY                  #wavelength
NUMBER_OF_SENSORS = 20                                      #
NOISE_VARIANCE = 0.01 #20db, can vary this                                       #noise variance 
DEVIATION_FACTOR = 4                                        #uncertainty in randomness of nonuniform spacinG
SNAPSHOTS_PER_REALIZATION = 256 
#NUMBER_OF_SOURCES = 10                                       #Number of sources
SPECTRUM_WIDTH = 60
ITERATIONS = 110000 


spectrum_view = np.arange(-90,90,1)
inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
array_length = (NUMBER_OF_SENSORS-1) * inter_elemental_distance 
X_columns = 2 * NUMBER_OF_SENSORS
X_D = np.zeros(([ITERATIONS, SNAPSHOTS_PER_REALIZATION, X_columns]))
Y_data = np.zeros([ITERATIONS,1])
number_of_sources = np.zeros([ITERATIONS, 1], dtype = np.int)
uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
uniform_sensor_position = uniform_sensor_position.reshape(NUMBER_OF_SENSORS, 1) #uniform positions for ULA
X_in = np.zeros(NUMBER_OF_SENSORS *(NUMBER_OF_SENSORS+1))
X_data = np.zeros([ITERATIONS, int(len(X_in))])
Train = np.zeros([ITERATIONS, int(len(X_in))+1])


MIN_NUMBER_OF_SIGNALS = 0
MAX_NUMBER_OF_SIGNALS = 6

data_gen = UpperTriFeatureExtract()
discrete_spectrum = DiscreteTargetSpectrum()
H = data_gen.hermitian
l=1
count = 0
for k in range(ITERATIONS): 
    #az_theta_D = np.array([5,20,-15]) 
    #m = np.random.randint(1,5)
    #print(m)
    count = count+ 1
    if count % 1000 == 0:
        print(count)
    
    number_of_sources[k] = data_gen.generate_random_number(MIN_NUMBER_OF_SIGNALS, MAX_NUMBER_OF_SIGNALS )
    signal_dir = data_gen.generate_random_directions(-SPECTRUM_WIDTH, SPECTRUM_WIDTH, number_of_sources[k])
    signal_dir = np.sort(signal_dir)
    signal_dir_rad = data_gen.convert_to_rad(signal_dir) #DOA to estimate
    number_s = np.asscalar(number_of_sources[k])
    x_u = data_gen.generate_qpsk_signals(signal_dir_rad=signal_dir_rad, sensor_position=uniform_sensor_position, NUMBER_OF_SENSORS=NUMBER_OF_SENSORS, NOISE_VARIANCE=NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS=SNAPSHOTS_PER_REALIZATION, NUMBER_OF_SOURCES=number_s, CARRIER_WAVELENGTH=CARRIER_WAVELENGTH) 
    r_xx = x_u.dot(H(x_u))
    x_data = data_gen.get_upper_tri(r_xx, NUMBER_OF_SENSORS)
    x_data_r = x_data.real
    #x_data_r = x_data_r/x_data_r.max()
    x_data_i = x_data.imag
    #x_data_i = x_data_i/x_data_i.max()
    parsed_input = data_gen.get_parse_upper_tri(x_data_r,x_data_i) 
    norm_x = data_gen.get_normalized_input(input_data=parsed_input)
    
    #X_u = norm_x.T
    
    X_data[k,:] = norm_x
    Y_data[k,:] = number_of_sources[k] # here it should be number of sources, keep it as a variable inside loop. 
    Train[k,0] = number_of_sources[k]
    Train[k,1:] = norm_x 
    l+= 1

 #
#
#Y_data = discrete_spectrum.get_integize_target(target=Y_data)  
#y_data = discrete_spectrum.get_encoded_target(spectrum=spectrum_view,target=Y_data)
#savetxt('data.csv', X_data, delimiter=',')
#savetxt('label.csv', Y_data, delimiter=',')
# Write hdf files    

import h5py
hf = h5py.File('0-6source_20snr_100k.h5','w')#change this 
hf.create_dataset('extracted_x', data=X_data)
hf.create_dataset('target_y', data=Y_data)
hf.close()



#estimated_angles = data_gen.get_rootmusic_estimate(input_data=x_u, frequency=CARRIER_FREQUENCY, sources=NUMBER_OF_SOURCES)
#print(np.sort(signal_dir))
#print(estimated_angles)
# start with 4 signals. Use one hot encoding, 0000, 0001, 0010, 0100, 1000. 
# Use CNN
#Things to do - Generate dataset of different number of signals(0 to 4), and their covariance matrixed.
#Design a  CNN such that input is covariance samples and output is encoded number of signals. 

