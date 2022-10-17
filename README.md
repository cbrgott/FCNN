# FCNN
Fully connected neural network using plain linear algebra.

####### PARAMETERS #############

X = Input layer, (k,1)

H1_preact = hidden layer (with no activation), (h,1)

Y_preact = output layer (with no activation), (m,1)

H1_act = hidden layer (with activation), (h,1)

Y_act = output layer (with activation), (m,1)

B1 = Bias of hidden layer, (h,1)

B2 = Bias of output layer, (m,1)

W1 = weight of hidden layer, (h,k)

W2 = weight of output layer , (n,h)

######## FORWARD PROPAGATIONN ##########

H1_preact = W1X+B1

H1_act = activation_funct(H1_preact)

Y_preact = W2H1_act+B2

Y_act = activation_funct(Y_preact)

########## BACKWARD PROPAGATION ########

The chain ruled is applied here. Notation: dEvar =dE/dvar  

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

W1 = W1-learning_rate.dEW1

W2 = W2-learning_rate.dEW2
