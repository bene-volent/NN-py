
# np provides arrays and useful functions for working with them
import numpy as np
import random
import json
# neural network class definition
class NN:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate = 0.1):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda X: np.tanh(X)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # query the neural network
    def predict(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs.flatten()
    def copy(self):
        temp = NN(self.inodes,self.hnodes,self.onodes)
        temp.who = self.who.copy()
        temp.wih = self.wih.copy()
        return temp


    def mutate(self,prob):
        mut_rate = prob
        def func(val):
            if random.uniform(0,1)<mut_rate:
                return val+random.gauss(0,0.2)
            else:
                return val
        hoshape = self.who.shape
        ihshape = self.wih.shape

        self.wih = np.array(list(map(func,self.wih.flatten()))).reshape(ihshape)
        self.who = np.array(list(map(func,self.who.flatten()))).reshape(hoshape)



    def save(self,name = ""):
        data = {
            'nodes':[self.inodes,self.hnodes,self.onodes],
            'wih':self.wih.tolist(),
            'who':self.who.tolist(),
        }
        if (len(name)==0):
            from datetime import date

            today = date.today()
            d1 = today.strftime("%d-%m-%Y")
            with open(f"/NN_data_{d1}.json",'w') as f:
                json.dump(data,f)
        else:
            with open(f"{name}.json",'w') as f:
                json.dump(data,f)
    @staticmethod
    def load(name):
        data = ''
        with open(f"{name}.json") as json_data_file:
            data = json.load(json_data_file)
        tmp = NN(*data['nodes'])
        tmp.who = np.array(data['who'])
        tmp.wih = np.array(data['wih'])
        return tmp
    @staticmethod
    def crossover(n1,n2):
        ihshape = n1.wih.shape
        hoshape = n1.who.shape

        


        n1wih = n1.wih.copy().flatten()
        n2wih = n2.wih.copy().flatten()
        n1who = n1.who.copy().flatten()
        n2who = n2.who.copy().flatten()

        ihrand = random.randint((len(n1who)*40)//100,len(n1wih)-1)
        ih = np.array(n1wih[0:ihrand].tolist() + n2wih[ihrand:len(n2wih)].tolist())
        ohrand = random.randint((len(n1who)*40)//100,len(n1who)-1)
        oh = np.array(n1who[0:ohrand].tolist() + n2who[ohrand:len(n2who)].tolist())

        child= NN(n1.inodes,n1.hnodes,n1.onodes)
        child.who = oh.reshape(hoshape)
        child.wih = ih.reshape(ihshape)
        return child




