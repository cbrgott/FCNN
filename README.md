# FCNN
Full connected neural network with plain linear algebra:
Forward propagation :

####### PARAMETERS #############
X = Input layer
H1_preact = hidden layer (with no activation)
Y_preact = output layer (with no activation)
H1_act = hidden layer (with activation)
Y_act = output layer (with activation)
B1 = Bias of hidden layer
B2 = Bias of output layer
######## FORWARD PROPAGATIONN ##########
H1_preact = W1X+B1
H1_act = activation_funct(H1_preact)
Y_preact = W2H1_act+B2
Y_act = activation_funct(Y_preact)
########## BACKWARD PROPAGATION ########
Notation dEvar =dE/dvar , tHe chain rule is applied.
E =  SSE (sum of square error)
#### ouput layer #####
dEY_act = dE/dEY_act
dEY_preact = (dE/dY_act)(dY_act/dY_preact)
dEW2 = (dE/dY_preact)(dY_preact/dW2) = dEY_preact.H1
dEB2 = (dE/dY_preact)(dY_preact/dB2) = dEY_preact.1 = dEY_preact
#### input layer ######
dEH1_act = (dE/dY_preact)(dY_preact/dH1_act) = dEY_preact.W2
dEH1_preact = (dE/dEH1_act)(dEH1_act/dEH1_preact) = dEY_preact.(dEH1_act/dEH1_preact)
dEW1 = (dE/H1_preact)(dH1_preact/dW1)=dEH1_preact.X1
dEB1 = (dE/H1_preact)(dH1_preact/dB1) = dEH1_preact.1 = dEH1_preact
############### GRADIENTE DESCENT ########
B1 = B1-learning_rate.dEB1
B2 = B2-learning_rate.dEB2
W1 = B1-learning_rate.dEW1
W2 = B1-learning_rate.dEW2
