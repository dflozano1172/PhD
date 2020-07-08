
import numpy as np

# weights      -> List of number of nodes per layer with input and output layer. 
# list_weights -> List of inter-layer matrices of weights
# list_bias    -> List of inter-layer bias term 
def initialize_weights(weights):
  list_weights = []
  list_bias    = []
  for i in range(len(weights) - 1):
    w = np.random.rand(weights[i],weights[i+1])
    b = np.random.rand(weights[i+1],1)
    #b = np.random.rand()

    list_weights.append(w)
    list_bias.append(b)
  return list_weights, list_bias
# Returns the value of the sigmoid function evaluated in z
def sigmoid(z):
  sig = 1 / (1 + np.exp(-z))
  return sig
# Returns the value of the derivative of sigmoid function evaluated in z
def der_sigmoid(z):
  der_sig = sigmoid(z) * (1 - sigmoid(z))
  return der_sig
# Return the list of matrices with the values of the nodes in each layer for all the samples, the last element of this list is the prediction
def prediction_FeedForward(X,list_weights, list_bias):  
  a_l = []
  a_l.append(X)
  a = X
  for i in range(len(list_weights)):
    z = np.dot(a,list_weights[i]) + list_bias[i].T
    #z = np.dot(a,list_weights[i]) + list_bias[i]
    a = sigmoid(z)
    a_l.append(a)
  return  a_l
# Find the cost of of given predictiopn with given real values.
def find_cost(y_hat,y):
  m = y.shape[0]
  total_cost = (1/m) * np.sum((y_hat-y)**2)
  return total_cost
def derivates_BackPropagation(y,a_l, list_weights, list_bias):
  m = y.shape[0]
  der_weights = []
  der_bias    = []
  y_hat       = a_l[2]

  dJ_dtheta1 = (1/m) * np.dot(a_l[1].T,(y_hat - y) * der_sigmoid(np.dot(a_l[1],list_weights[1]) + list_bias[1].T))
  dJ_db1     = (1/m) *          np.sum((y_hat - y) * der_sigmoid(np.dot(a_l[1],list_weights[1]) + list_bias[1].T),axis=0)
  dJ_db1     = dJ_db1.reshape(dJ_db1.shape[0],1) 
  der_weights.insert(0,dJ_dtheta1)
  der_bias.insert(0,dJ_db1)

  dJ_dtheta0 = (1/m) * np.dot(a_l[0].T,(y_hat - y) * der_sigmoid(np.dot(a_l[1],list_weights[1]) + list_bias[1].T) * der_sigmoid(np.dot(a_l[0],list_weights[0]) + list_bias[0].T) * list_weights[1].T)
  dJ_db0     = (1/m) *          np.sum((y_hat - y) * der_sigmoid(np.dot(a_l[1],list_weights[1]) + list_bias[1].T) * der_sigmoid(np.dot(a_l[0],list_weights[0]) + list_bias[0].T) * list_weights[1].T,axis = 0)
  dJ_db0     = dJ_db0.reshape(dJ_db0.shape[0],1) 
  der_weights.insert(0,dJ_dtheta0)
  der_bias.insert(0,dJ_db0)
  return der_weights, der_bias


def  Update_weights(list_weights, list_bias, der_weights, der_bias, lr):
  new_list_weights = []
  new_list_bias = []
  for i in range(len(list_weights)):
    weights_i = list_weights[i] - lr * der_weights[i]
    new_list_weights.append(weights_i)

  for i in range(len(list_bias)):
    bias_i = list_bias[i] - lr * der_bias[i]
    new_list_bias.append(bias_i)
  return new_list_weights, new_list_bias

def neural_network(X,y,h,lr,epochs):
  error_list = []
  list_weights, list_bias = initialize_weights([X.shape[1],h,1])
  
  for i in range(epochs):
    a_l                     = prediction_FeedForward(X,list_weights, list_bias)
    y_hat                   = a_l[2]
    cost                    = find_cost(y_hat,y)
    error_list.append(cost)
    der_weights, der_bias   = derivates_BackPropagation(y, a_l, list_weights, list_bias)
    #der_weights, der_bias   = derder(X, list_weights, list_bias)
    list_weights, list_bias = Update_weights(list_weights, list_bias, der_weights, der_bias, lr)
    if i % (epochs / 10) == 0:
      print(cost)
  return list_weights, list_bias, error_list