from __future__ import print_function
%matplotlib inline
#import scipy.io as sio
import numpy as np
import math 
import matplotlib.pyplot as plt 
###
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy import signal
import numpy as np
from numpy import linalg as LA
from scipy import signal

def g(u, switch = 2):
    '''
    Inputs:
    u - as a input vector
    switch - 1:tanh(a1u)
             2:u*exp(-u^2/2)
    reference: Independent Component Analysis: Algorithms and Applications By Aapo Hyvärinen
    '''
    if (switch == 2):
            return np.multiply(u, np.exp(-u**2/2))
    else:
            a = 1 # 1 <= a <= 2
            return np.tanh(np.multiply(a, u))

def diff_g(u, switch = 2):
    if (switch == 2):
        diff = np.exp(-u**2 / 2) - u**2 * np.exp(-u**2 / 2)
    else:
        print('Error! No such function exists!')
    return diff

def preprocess(X):
    ### --- centering ---
    # EX is get by sum over all rows to form a single row
    EX = np.mean(X, axis=0)
    # subtract EX from every row of X
    X = X - np.asarray([EX] * n_samples)
    # now you can check EX is indeed 0 
    EX = np.mean(X, axis=0)
    #print EX

    ### --- whitening ---
    # compute EXX matrix
    EXX = np.dot(X.T, X)/n_samples
    # get eigenvalues and eigVectors (i.e. transform matrix (E))
    eigValues, eigVectors = LA.eig(EXX)
    # D_insqrt = (DiagMatrix)^(-1/2)
    D_insqrt = np.diag(1.0/np.sqrt(eigValues))
    # tempM = E*D_insqrt*(E^T)
    tempM = np.dot(np.dot(eigVectors, D_insqrt), eigVectors.T)
    # get normalized X_n = X*tempM
    X = np.dot(X, tempM)
    return X

def ICA(x):
    number_of_independent_source = np.shape(x)[1]
    number_of_source_we_get = 0
    number_of_time_frame = np.shape(x)[0] # row is time frame, column is source number
    all_w = []
    while number_of_source_we_get < number_of_independent_source:
        if number_of_source_we_get == 0:
            # This is the first fitting vector, no Schmidt decomposition
            # Initialize w vector
            w = np.random.rand(number_of_independent_source)
            small_error = 0.01
            # Normalize w
            w = w / np.sqrt(np.dot(w, w))
            temp_mean_1 = 0
            temp_mean_2 = 0
            for i in range(0, number_of_time_frame):
                temp_mean_1 = temp_mean_1 + (x[i, :] * g(np.dot(w, x[i, :]), switch = 2)) / number_of_time_frame
                temp_mean_2 = temp_mean_2 + diff_g(np.dot(w, x[i, :]) , switch = 2) / number_of_time_frame
            w_prime = temp_mean_1 - temp_mean_2 * w
            #print('w_prime:', w_prime)
            # Normalize w
            print('break point 1')
            new_w = (w_prime / np.sqrt(np.dot(w_prime, w_prime))).copy()
            estimate_error = abs(1 - np.dot(new_w, w))
            print('error: ', estimate_error)
            iter = 0
            while estimate_error > small_error:
                print('iter: ', iter)
                w = new_w.copy()
                temp_mean_1 = 0
                temp_mean_2 = 0
                for i in range(0, number_of_time_frame):
                    temp_mean_1 = temp_mean_1 + (x[i, :] * g(np.dot(w, x[i, :]), switch = 2)) / number_of_time_frame 
                    temp_mean_2 = temp_mean_2 + diff_g(np.dot(w, x[i, :]) , switch = 2) / number_of_time_frame
                w_prime = temp_mean_1 - temp_mean_2 * w
                new_w = (w_prime / np.sqrt(np.dot(w_prime, w_prime))).copy()
                estimate_error = abs(1 - abs(np.dot(new_w, w)))
                print('error: ', estimate_error)
                iter = iter + 1
            all_w.append(new_w)
            number_of_source_we_get = number_of_source_we_get + 1
        else:
            # Need to impose Schmidt decomposition
            # Initialize w vector
            w = np.random.rand(number_of_independent_source)
            small_error = 0.01
            # Normalize w
            w = w / np.sqrt(np.dot(w, w))
            temp_mean_1 = 0
            temp_mean_2 = 0
            for i in range(0, number_of_time_frame):
                temp_mean_1 = temp_mean_1 + (x[i, :] * g(np.dot(w, x[i, :]), switch = 2)) / number_of_time_frame
                temp_mean_2 = temp_mean_2 + diff_g(np.dot(w, x[i, :]) , switch = 2) / number_of_time_frame
            w_prime = temp_mean_1 - temp_mean_2 * w
            # Do Schmidt decomposition
            parallel_w_part = 0
            for j in range(0, number_of_source_we_get):
                parallel_w_part = parallel_w_part + np.dot(w_prime, all_w[j]) * all_w[j]
            w_prime = w_prime - parallel_w_part
            new_w = (w_prime / np.sqrt(np.dot(w_prime, w_prime))).copy()
            estimate_error = abs(1 - abs(np.dot(new_w, w)))
            print('break point 2')
            print(estimate_error)
            while estimate_error > small_error:
                
                w = new_w.copy()
                temp_mean_1 = 0
                temp_mean_2 = 0
                for i in range(0, number_of_time_frame):
                    temp_mean_1 = temp_mean_1 + (x[i, :] * g(np.dot(w, x[i, :]), switch = 2)) / number_of_time_frame
                    temp_mean_2 = temp_mean_2 + diff_g(np.dot(w, x[i, :]) , switch = 2) / number_of_time_frame
                w_prime = temp_mean_1 - temp_mean_2 * w
                # Do Schmidt decomposition
                parallel_w_part = 0
                for j in range(0, number_of_source_we_get):
                    parallel_w_part = parallel_w_part + np.dot(w_prime, all_w[j]) * all_w[j]
                w_prime = w_prime - parallel_w_part
                new_w = (w_prime / np.sqrt(np.dot(w_prime, w_prime))).copy()
                estimate_error = abs(1 - abs(np.dot(new_w, w)))
                print(estimate_error)
            all_w.append(new_w)
            number_of_source_we_get = number_of_source_we_get + 1
    #Done! all_w get!    
    return all_w
            
def get_decompose(W, X):
    number_of_time_frame = np.shape(X)[0]
    number_of_sound_track = np.shape(X)[1]
    new_S = np.zeros((number_of_time_frame, number_of_sound_track))
    for i in range(0, number_of_time_frame):
        for j in range(0, number_of_sound_track):
            new_S[i, j] = np.dot(X[i, :], W[j])
    return new_S



### generate demo-data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

### preform ICA and reconstruct original sound track

X = preprocess(X)
W = ICA(X)
cal_S = get_decompose(W, X)

### print result

models = [X, S, new_S]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.004, 0.9, 0.94, 0.26, 0.46)
plt.show()



