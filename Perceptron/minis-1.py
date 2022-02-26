import numpy as np 
import pandas as pd 
import math
import matplotlib.cm as cm 
import matplotlib.pyplot as plt



train_data = pd.read_csv("train.csv")
test_data= pd.read_csv("test.csv")
#oddzielenie wynikow od pikseli 
train_labels=np.array(train_data.loc[:,'label'])
test_labels=np.array(train_data.loc[:,'label'])
train_data=np.array(train_data.loc[:,train_data.columns!='label'])
test_data=np.array(test_data.loc[:,test_data.columns!='label'])
#train_data=train_data/train_data.max()
print(train_data.shape)
#kod zeby pokazac 30 przykladowych liczb
for i in range(2):
    index=i;
    plt.title(f"No. {index}")
    plt.imshow(test_data[index].reshape(28,28), cmap=cm.binary)
    plt.show()

data = pd.read_csv('submission1.csv', encoding= 'unicode_escape')
print(data.head(10))
plt.hist(train_labels,edgecolor='black', linewidth=1.2)
plt.title("Histogram danych")
plt.axis([0,9,0,5000])
plt.xlabel("Cyfra")
plt.ylabel("Liczba wystąpień w zbiorze")
plt.show()
#albo histogram albo to nizej
print("train data")
y_value=np.zeros((1,10))
for i in range (10):
    print("Liczba wystąpień cyfry ",i,"=",np.count_nonzero(train_labels==i))
    y_value[0,i-1]= np.count_nonzero(train_labels==i)


train_data=np.reshape(train_data,[784,42000])
train_label=np.zeros((10,42000))
for col in range (42000):
    val=train_labels[col]
    for row in range (10):
        if (val==row):
            train_label[val,col]=1
print("train_data shape="+str(np.shape(train_data)))
print("train_label shape="+str(np.shape(train_label)))
print("test shape ="+str(np.shape(test_data)))
print("test shape ="+str(np.shape(test_data)))
test_data=np.reshape(test_data,[784,28000])
test_label=np.zeros((10,28000))
for col in range (28000):
    val=train_labels[col]
    for row in range (10):
        if (val==row):
            train_label[val,col]=1
print("test_data shape="+str(np.shape(test_data)))
print("test_label shape="+str(np.shape(test_label)))
#definiowanie funkcji aktywacji
def relu(Z):
   A = np.maximum(0,Z)    
   cache = Z 
   return A, cache

def softmax(Z):
    e_x = np.exp(Z)
    A= e_x / np.sum(np.exp(Z))  
    cache=Z
    return A,cache

    
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def softmax_backward(Z,cache):
    Z=cache
    dZ=np.zeros((42000,10))
    Z=np.transpose(Z)
    for row in range (0,42000):
            den=(np.sum(np.exp(Z[row,:])))*(np.sum(np.exp(Z[row,:])))
            for col in range (0,10):
                sums=0
                for j in range (0,10):
                    if (j!=col):
                        sums=sums+(math.exp(Z[row,j]))
                
                dZ[row,col]=(math.exp(Z[row,col])*sums)/den           
    dZ=np.transpose(dZ)
    Z=np.transpose(Z)

    assert (dZ.shape == Z.shape)
    return dZ   

    
#inicjalizacja wag i błędów
def initialize_parameters_deep(layer_dims):   
    parameters = {}
    L = len(layer_dims)            # ilosc warstw 
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

#propagacja w przód
def linear_forward(A, W, b):
    Z = np.dot(W,A) +b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        
        A, activation_cache = relu(Z) 
    elif activation == "softmax":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # ilość warstw w sieci
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)               
    return AL, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    return cost
#propagacja wsteczna
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)  
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True);
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)  
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    M=len(layers_dims)
    current_cache = caches[M-2]
    grads["dA"+str(M-1)], grads["dW"+str(M-1)], grads["db"+str(M-1)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    for l in range(len_update-1):
        parameters["W" + str(l+1)] =parameters["W" + str(l+1)] - (learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db" + str(l+1)])
    return parameters

def plot_graph(cost_plot):
       
    x_value=list(range(1,len(cost_plot)+1))
    print(x_value)
    print(cost_plot)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(x_value,cost_plot,0.,color='g')
    plt.show()


layers_dims = [784,700,600,500,400,300,200,100,50,10] #  n-warstwowy model (n=6 włączając warstwę wyjścia i wejścia)
len_update=len(layers_dims) 


def L_layer_model(X, Y, layers_dims, learning_rate , num_iterations , print_cost=False):
    print("training...")
    costs = []  
    cost_plot=np.zeros(num_iterations)
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost =compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate) 
        cost_plot[i]=cost;
    
    plot_graph(cost_plot)
    return parameters

parameters1 = L_layer_model(train_data, train_label, layers_dims,learning_rate = 0.0005, num_iterations =22 , print_cost = True) 
print("training done")



def check_test(X,params,num_iterations=1):
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, params) 
        
        return AL,caches

ez1, ez2 = check_test(test_data,parameters1)        

results =[]

for i in range(28000):
    tmp = {}
    for j in range(10):        
        tmp[j] = ez1[j,i]
    max_key = max(tmp, key=tmp.get)
    results.append(max_key)

plt.hist(results,edgecolor='black', linewidth=1.2)
plt.title("Histogram danych")
plt.axis([0,9,0,5000])
plt.xlabel("Cyfra")
plt.ylabel("Liczba wystąpień w zbiorze testowym")
plt.show()
#odpowiedz
print("test done")
np.savetxt('submission.csv', 
           np.c_[range(1,28001),results], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
