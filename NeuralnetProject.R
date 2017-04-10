# We are going to be using the Boston dataset in the MASS package

set.seed(500)
library(MASS)
data <- Boston 

# The Boston dataset contains data about the housing values in the suburbs of Boston.
# The goal is to predict the median value (medv) of occupied homes

# checking for gaps in the data 
apply(data, 2, function(x) sum(is.na(x)))

# since there are no gaps, we can proceed. If there were gaps or other problems,
# we would have to fix the dataset. Otherwise our network would perform poorly.

# We randomly split the data into a train set and a test set. 

# random splitting of data 
index <- sample(1:nrow(data), round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

# Here we fit a linear regression model and test it on the test set.
# Since we are dealing with a regression problem, we are going to use 
# the mean squared error (MSE) as a measure of how much our predictions 
# are far away from the real data.

# Linear regression on training data
linreg.fit <- glm(medv~., data=train)
summary(linreg.fit)

# Prediction of testing data based on learning 
predict.linreg <- predict(linreg.fit, test)
mean_squared_error.linreg <- sum((predict.linreg - test$medv)^2)/nrow(test)

# As a first step, we are going to address data preprocessing.
# It is good practice to normalize your data before training a neural network.
# If this step is ommited, your neural models may become useless.

# normalizing data
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, 
scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

# Now that our data is ready, we create a neural netwrok with the configuration 13:5:3:1.
# this means there will be a 13-node input layer (corresponding to the 13 variables in
# Boston dataset) two hidden layers, one with 5 neurons and one with 3 nuerons, and
# an output layer of one neuron. 

library(grid)
library(neuralnet)

nnames <- names(train_)
f <- as.formula(paste("medv ~", paste(nnames[!nnames %in%
"medv"], collapse = " + ")))

nn <- neuralnet(f, data=train_, hidden=c(5,3), linear.output=T)

# The hidden argument accepts a vector with the number of neurons for each hidden layer,
# while the argument linear.output is used to specify whether we want to do regression 
# Currently, there are no hard and fast rules for the configuration of Neural Networks. 

# We can visualize the network by plottong it:
#plot(nn)

# This is the graphical representation of the model with the weights on each 
# connection. The black lines show the connections between each layer and 
# the weights on each connection while the blue lines show the bias term
# added in each step. The bias can be thought as the intercept of a linear model.

# Predict medv using test set and our neural network
predict.nn <- compute(nn, test_[,1:13])
predict.nn_ <- predict.nn$net.result*(max(data$medv)-
min(data$medv))+min(data$medv)
test.r <- (test_$medv)*(max(data$medv)-
min(data$medv))+min(data$medv)

# Now we can try to predict the values for the test set and calculate the MSE. 
# Remember that the net will output a normalized prediction, 
# so we need to scale it back to make a meaningful comparison. 

# We then compare the two MSEs.
mean_squared_error.nn <- sum((test.r - predict.nn_)^2)/nrow(test_)
print(paste(mean_squared_error.linreg, mean_squared_error.nn))

# Apparently, the net is doing a better work than the linear model at predicting medv.

# A visual comparison of the performance of the network and the linear model 

par(mfrow=c(1,2))

plot(test$medv,predict.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,predict.linreg,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

# These graphs confirm that our neural network 
# is performing better than the linear regression model. 
