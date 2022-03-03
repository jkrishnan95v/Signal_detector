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
NUMBER_OF_SENSORS = 10                                      #
NOISE_VARIANCE = 0.1 #20db, can vary this                                       #noise variance 
DEVIATION_FACTOR = 4                                        #uncertainty in randomness of nonuniform spacinG
SNAPSHOTS_PER_REALIZATION = 20
#NUMBER_OF_SOURCES = 10                                       #Number of sources
SPECTRUM_WIDTH = 60
ITERATIONS = 1000

number_of_subrray = 5



spectrum_view = np.arange(-90,90,1)
inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
array_length = (NUMBER_OF_SENSORS-1) * inter_elemental_distance 
X_columns = 2 * NUMBER_OF_SENSORS
X_D = np.zeros(([ITERATIONS, SNAPSHOTS_PER_REALIZATION, X_columns]))
Y_data = np.zeros([ITERATIONS,1])
#number_of_sources = np.zeros([ITERATIONS, 1], dtype = np.int)
number_of_correlatedsources = np.zeros([ITERATIONS, 1], dtype = np.int)

number_of_coherentsignals = np.zeros([ITERATIONS, 1], dtype = np.int)
uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
uniform_sensor_position = uniform_sensor_position.reshape(NUMBER_OF_SENSORS, 1) #uniform positions for ULA
X_in = np.zeros(number_of_subrray *(number_of_subrray+1))
X_data = np.zeros([ITERATIONS, int(len(X_in))])
#X_data = np.zeros([ITERATIONS, NUMBER_OF_SENSORS])

Train = np.zeros([ITERATIONS, int(len(X_in))+1])


NUMBER_OF_SIGNALS = 2
number_of_correlatedsources = 3
#MAX_NUMBER_OF_SIGNALS = 9
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
    if count % 100 == 0:
        print(count)
    
    #number_of_sources[k] = data_gen.generate_random_number(MIN_NUMBER_OF_SIGNALS, MAX_NUMBER_OF_SIGNALS )
   # if number_of_sources[k] == 1:
  #   else:
#        number_of_correlatedsources[k] = data_gen.generate_random_number(MIN_NUMBER_OF_SIGNALS, number_of_sources[k]-1 )
    #number_of_coherentsignals[k] = number_of_sources[k] - number_of_correlatedsources[k]
    signal_dir = data_gen.generate_random_directions(-SPECTRUM_WIDTH, SPECTRUM_WIDTH, NUMBER_OF_SIGNALS)
    signal_dir = np.sort(signal_dir)
    signal_dir_rad = data_gen.convert_to_rad(signal_dir) #DOA to estimate
   # number_s = np.asscalar(NUMBER_OF_SIGNALS)
    x_u = data_gen.generate_correlated_qpsk_signals(signal_dir_rad=signal_dir_rad, sensor_position=uniform_sensor_position, NUMBER_OF_SENSORS=NUMBER_OF_SENSORS, NOISE_VARIANCE=NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS=SNAPSHOTS_PER_REALIZATION, NUMBER_OF_SOURCES=NUMBER_OF_SIGNALS, CARRIER_WAVELENGTH=CARRIER_WAVELENGTH, number_of_correlatedsources=number_of_correlatedsources) 
    
    pho = np.zeros(number_of_correlatedsources)
    deltaphi = np.zeros( number_of_correlatedsources)
    
    
    r_f = [
        np.zeros([5, 5], dtype=int)
        for _ in range(5)
    ]
    
    r_fwd = [
        np.zeros([5, 5], dtype=int)
    ]
    
    
    
    r_b = [
        np.zeros([5, 5], dtype=int)
        for _ in range(5)
    ]
    
    r_bwd  = [
        np.zeros([5, 5], dtype=int)
    ]
    
    
    
    j=0
    for i in range(number_of_subrray,NUMBER_OF_SENSORS-1):
        x_f = x_u[j:i,:]
        r_f[j] = x_f.dot(H(x_f)) 
        j = j+1
    
    j=NUMBER_OF_SENSORS
    a=0
    for i in range(number_of_subrray,0,-1):
        x_b = x_u[i:j,:]
        r_b[a] = x_b.dot(H(x_b)) 
        j -= 1
        a +=1 


    for i in range(0,number_of_subrray-1):
        r_fwd += r_f[i]
        r_bwd += r_b[i]
    
    r_fwd = r_fwd/number_of_subrray
    r_bwd = r_bwd/number_of_subrray

    r_fbss = r_fwd + r_bwd/2  
    r_fbss = r_fbss.reshape(number_of_subrray, number_of_subrray)      
    r_xx = x_u.dot(H(x_u))  
    
    
    w = np.linalg.eigvalsh(r_xx) # W are the eigen values and v the eigen vectors
    
    
    x_data = data_gen.get_upper_tri(r_fbss, number_of_subrray)
    x_data_r = x_data.real
    #x_data_r = x_data_r/x_data_r.max()
    x_data_i = x_data.imag
    #x_data_i = x_data_i/x_data_i.max()
    parsed_input = data_gen.get_parse_upper_tri(x_data_r,x_data_i) 
    norm_x = data_gen.get_normalized_input(input_data=parsed_input)
    
    #X_u = norm_x.T
    
    X_data[k,:] = norm_x
    Y_data[k,:] = NUMBER_OF_SIGNALS # here it should be number of sources, keep it as a variable inside loop. 
    #Train[k,0] = number_of_sources[k]
    #Train[k,1:] = w
    l+= 1

 #
#
#Y_data = discrete_spectrum.get_integize_target(target=Y_data)  
#y_data = discrete_spectrum.get_encoded_target(spectrum=spectrum_view,target=Y_data)
#savetxt('data.csv', X_data, delimiter=',')
#savetxt('label.csv', Y_data, delimiter=',')
# Write hdf files    

import h5py
hf = h5py.File('1source_10snr_10k_1correlated.h5','w')#change this 
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

