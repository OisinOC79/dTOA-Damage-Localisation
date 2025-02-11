# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:56:33 2025

@author: oconn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split as tts
import kernel_cookbook as kern
import gaussian_models as GM

class likelihood_estimation:
    
    #This will be applied across multiple models
    
    #Init will generate the coordinates required, nothing else. Hyps will need to be called in the likelihood estimation argument
    def __init__(self,feature,dtoa,test_value,n_coordinates_x,n_coordinates_y): 
        self.n_coordinates = int(n_coordinates_x)*int(n_coordinates_y)
        #feature = Xy_normalised,dtoa=dtoa standardised, test_value = random number of point being tested, n_coordinates = how many coordinates per space in grid
        self.feature = feature
        self.target = dtoa
        #Definition of test point and associated dTOA value
        x_coordinate_test_J = self.feature[test_value]
        #Ensuring test point doesn't appear in any training data
        self.feature = np.delete(self.feature, test_value, axis=0)

                
        #Corresponding dTOA test value for selected coordinates
        self.matching_indices = np.where((self.feature[:, 0] == x_coordinate_test_J[0]) & (self.feature[:, 1] == x_coordinate_test_J[1]))
        self.dtoa_test_value = self.target[test_value]
        #Making sure test point doesn't appear in any training data
        
        self.target = np.delete(self.target,[test_value])
              
        x_normalised = self.feature[:,0]
        y_normalised = self.feature[:,1]

        #Creation of alll possible candidate locations (x,y) in a (100,100) grid 
        candidate_x = np.linspace(np.min(x_normalised),np.max(x_normalised), n_coordinates_x)
        candidate_y = np.linspace(np.min(y_normalised), np.max(y_normalised), n_coordinates_y)
        
        #Full array of candidate coordinates
        self.candidate_coordinates = np.array([],[])
        self.candidate_coordinates = np.column_stack((np.tile(candidate_x, len(candidate_y)), np.repeat(candidate_y, len(candidate_x))))
        self.candidate_locations_rows, self.candidate_locations_columns = np.shape(self.candidate_coordinates)
       
        #establishment of arrays to be used in likelihood assessment
        self.likelihood_array = np.zeros(((self.n_coordinates),1))
        self.mu_array = []
        self.sigma_f_array = []
        self.test_coordinate = self.feature[test_value]

    def GaussianModel(self,sample_type, test_split,r_state,beta):
    
        
        if sample_type == 'bayesian':
        
            #Setting up datasets for bayesian sampling- this array has 36 values linearly sampled across the testpiece
            train_index = [11, 89,164,231,288,376,458,498,552,623,717,781,867,961,1003,1069,1107,1160, 1203,1234,1251,1281,1334,1412,1482,1556,1651,1715,1773,1824,1909,1964,2068,2112,2167,2253]
           
            X_sample = self.feature[train_index]
            y_sample = self.target[train_index]
            
            self.feature = np.delete(self.feature,train_index, axis = 0)
            self.target = np.delete(self.target,train_index)
           
            kernel = kern.Matern32
            hyps0 = np.array([1,1.2,1e-6])
            bounds = [(1e-1,2),(1,1.5),(1e-6,1.0005)]
            hyps_opt, nlml = GM.train(hyps0,(X_sample, y_sample),kernel,bounds,opt_type='min')

            
            #Generation of test/train arrays for bayesian approach
            X_train, X_test, y_train, y_test = tts(self.feature, self.target, test_size=0.001, random_state=r_state)
            
            #Definition of model
            model = CholeskyGPR(length_scale=hyps_opt[1] ,var = hyps_opt[0], sigma_n = hyps_opt[-1])
            
        #Generation of training sample using bayesian Upper Confidence Bound acquisition function
            for i in range(600):
                UCB =[]
                mu_pred_array = []
                #selecting coordinate to train model
                UCB, mu_pred_array, std_average, mu_abs_average, mu_pred_adjusted= acquisition_function(X_sample,y_sample, X_train, model ,beta)
                max_index = np.argmax(np.abs(UCB))
                
                Next_X_coordinate = X_train[max_index].reshape(1,-1)
                Next_Y_coordinate = y_train[max_index].reshape(-1)
                
                X_sample = np.vstack([X_sample, Next_X_coordinate])
                y_sample = np.append(y_sample,Next_Y_coordinate)
                
                #removing selected coordinates from array containing all potential coordinates
                X_train = np.delete(X_train, max_index, axis=0)
                y_train = np.delete(y_train, max_index) 
           
            self.X_train = X_sample
            self.y_train = y_sample

            
        if sample_type == 'random':
            #random generation of sample
            self.X_train, self.X_test, self.y_train, self.y_test = tts(self.feature,self.target,test_size = test_split, random_state = r_state)
        
        
        #Final model fitting using training data
        kernel = kern.Matern32
        hyps0 = np.array([1,1.2,1e-6])
        bounds = [(1e-2,20),(1e-3,3),(1e-6,1.0005)]
        self.hyps_opt, nlml = GM.train(hyps0,(self.X_train, self.y_train),kernel,bounds,opt_type='min')
        model = CholeskyGPR(length_scale=np.log(self.hyps_opt[1]) ,var = np.log(self.hyps_opt[1]), sigma_n = np.log(self.hyps_opt[-1]))  # Adjust length scale accordingly// Was 0.35 for the testing index above but for 50/50 split has been changed to 0.025
        model.fit(self.X_train, self.y_train)
        return model
    
    def source_localisation(self,model):
        counter = int(0)
    
        for i in range(self.candidate_locations_rows):
             
                #The location being assessed for a given loop iteration
                candidate_location = (self.candidate_coordinates[i,:]).reshape(1,2)
                
                #The prediction of the dTOA value for the candidate location
                mu_j, sigma_f_j = model.predict(candidate_location,self.X_train) #0.45
        
                #Array containing all predicted dTOA values
                self.mu_array.append(mu_j.item())
                
                #Array containing all standard deviations for predictions 
                self.sigma_f_array.append(sigma_f_j.item())
                
                #Likelihood assessment, broken down into 3 parts for readibility
                likeli_part_1 = (-0.5*np.log(sigma_f_j.item()))
                likeli_part_2 = -((self.dtoa_test_value - mu_j.item())**2)/(2*sigma_f_j.item())    
                likeli_part_3 = -0.5*np.log(np.pi)
                
                #Final likelihood sum
                likelihood = np.exp(likeli_part_1 + likeli_part_2 + likeli_part_3)
              
                self.likelihood_array[counter,0] = likelihood
                
                counter+=1
                
        return self.likelihood_array
            
    def plt_source_localisation(self,sensor_coordinates_1, sensor_coordinates_2, sensor_1, sensor_2):
        plt.figure(1, figsize = (8,12))
        plt.xlim(0,0.5405405405405406)
        plt.ylim(0,1)
        plt.scatter(self.candidate_coordinates[:,0],self.candidate_coordinates[:,1], c= self.likelihood_array, cmap = 'plasma')
        plt.colorbar(label = 'likelihood')
        plt.scatter(self.test_coordinate[0], self.test_coordinate[1],color='g', marker='o', s=150, label='AE Source location')
        plt.scatter(self.candidate_coordinates[np.argmax(self.likelihood_array),0], self.candidate_coordinates[np.argmax(self.likelihood_array), 1], color='r', marker='x', s=125, linewidth = 3, label='Predicted AE source location')
        plt.scatter(sensor_coordinates_1[0], sensor_coordinates_1[1],color = 'black' , marker = '>' , s=50, label = sensor_1)
        plt.scatter(sensor_coordinates_2[0], sensor_coordinates_2[1],color = 'black' , marker = '>' , s=50, label = sensor_2)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Likelihood evaluatation for sensor pair' + sensor_1 +  sensor_2)
        plt.legend(loc=1)
        plt.show()

#Used for final likelihood summation
def sum_arrays(*arrays):
    n = len(arrays)
    return sum((1/n) * np.array(arr) for arr in arrays)        
        
class CholeskyGPR:
     def __init__(self, length_scale=1, var = 1, sigma_n =1):
         self.length_scale = length_scale
         self.var = var
         self.sigma_n = sigma_n
         self.L = None
         self.alpha = None
         self.normalised_spatial_distance = None
         self.predicted_normalised_spatial_distance = None
         
         #Function to normalise the model input (x/y coordinates based on max/min values)
         #Not being used currently
     def input_normalise(input_array):
         normalised_input_array = (input_array - 0)/(370-0)
         return normalised_input_array
     
     def input_normalise_hy(input_array):
         normalised_input_array = ((input_array - np.min(input_array)) / (np.max(input_array - np.min(input_array))))
         return normalised_input_array

         #Function to normalise the model output (target) dtoa values 
         #Done by subtracting the mean and dividing by the standard deviation
     def output_standardise(output_array):
         standardised_output_array = (output_array-np.average(output_array))/(np.std(output_array))
         return standardised_output_array


     #Definition of kernel. Kernel calculates spatial distances 
     #Between X_train points, predict method does this differently to advise predictions
     def matern32_kernel(self, X_train):
         self.normalised_spatial_distance = distance.cdist(X_train, X_train, 'euclidean')
         
         matern32_kernel_value = self.var*((1 + np.sqrt(3) * (self.normalised_spatial_distance/self.length_scale)) * np.exp(-np.sqrt(3.0) * (self.normalised_spatial_distance/self.length_scale))) 
         return matern32_kernel_value
     
       
     #Determination of fit method
     def fit(self, X_train, y_train):
         self.kernel = self.matern32_kernel(X_train)
         I_matrix = np.eye(len(X_train))
         I_matrix *= self.sigma_n
         #Calculation of cholesky decomposition - L = cholesky(K + σn)
         self.L = np.linalg.cholesky(self.kernel + I_matrix)
         l_inv_y = np.linalg.solve(self.L,y_train)
         #Calculation of vector for further use in prediction - α = L.T \ (L\y)
         inverse_Y = np.linalg.solve(self.L, y_train)
         self.alpha = np.linalg.solve(self.L.T, inverse_Y)
         #Covariance matrix is determined by the euclidean distance bewteen x_train and x_train (K,K)
         self.covariance_matrix = self.kernel
         
         
     #Calculation of predictions
     def predict(self, X_test, X_train): 
         
         #Calculation of spatial distance bewteen X train and X test
         self.predicted_normalised_spatial_distance = (distance.cdist(X_train, X_test)/self.length_scale)
         
         #Calculation of spatial distance between X test and X test
         self.testing_spatial_distance = (distance.cdist(X_test, X_test)/self.length_scale)
         
         #Kernel (X_train, X_test) (K,K*)
         K_s = self.var*((1 + np.sqrt(3) * self.predicted_normalised_spatial_distance) * np.exp(-np.sqrt(3.0) * self.predicted_normalised_spatial_distance))
        
         #Kernel (X_test, X_test) (K*,K*)
         K_ss = self.var*((1 + np.sqrt(3) * self.testing_spatial_distance)* np.exp(-np.sqrt(3.0) * self.testing_spatial_distance))

         #Predictive Mean
         mu = np.dot(K_s.T, self.alpha)  
         
         #Predictive variance calculation
         v = np.linalg.solve(self.L, K_s)
         vTv = np.dot(v.T,v)

         #Calculation of covariance matrix using X_test, X_test [K_ss = (K*,K*)]
         sigma_f = K_ss - vTv 
                
         
         return mu, sigma_f
     
     
     #Calculation of Negative Log Marginal Likelihood
     def NLML(self, X_train, y_train):
         self.fit(X_train, y_train)
         covariance_matrix_xx = self.covariance_matrix
         alpha = self.alpha
         L = self.L
         n = len(X_train)    
         noise = 1e-6* np.eye(n)
         

         inverse_a_y = np.dot(y_train.T,alpha)
         
         #part_1 = -0.5*(np.dot(y_train.T, {(np.linalg.solve((cholesky(k+σn).T), [(np.linalg.solve(cholesky(k+σn),y_Train))]))}))
         
         #determination of -0.5y.T*(a)^-1*y 
         part_1 = -0.5 *inverse_a_y
         
         #L = cholesky(K + σn)
         part_2 = -0.5*np.sum(np.log(np.diag(L)))
         
         #Part 3 is a fixed scalar
         part_3 = -0.5*n*np.log(2*np.pi)
              
         nlml = part_1+part_2+part_3
         
         print('NLML' , nlml)
         return nlml
         
     #Mean Squared Error Function
     def MSE(self,y_test,y_pred):

         n_mse = len(y_test)

         mse = np.sqrt(np.sum((y_test-y_pred)**2)/n_mse)
         
         return mse
     
     def MAE(self, y_test, y_pred):
         n_mae = len(y_test)
         mae = np.sum(np.abs(y_test - y_pred)) / n_mae
         return mae
         

 #Bayesian UCB acquisition function
def acquisition_function( X_sample, Y_sample, X_train, Surrogate_Model, beta):
     mu_pred_adjusted = np.array([])
     
     mu_pred_array = np.array([])
     Surrogate_Model.fit(X_sample, Y_sample)
     mu_pred, variance = Surrogate_Model.predict(X_train, X_sample)
     std = np.sqrt(np.diag(variance))
     mu_pred_array = np.append(mu_pred_array,mu_pred)

     mu_pred_abs = np.abs(mu_pred)
     mu_pred_abs_mean = np.mean(mu_pred_abs)
     #Calculation of UCB

     UCB = mu_pred_abs + (beta * std)
     return UCB, mu_pred_array, std, mu_pred_abs_mean, mu_pred_adjusted
 