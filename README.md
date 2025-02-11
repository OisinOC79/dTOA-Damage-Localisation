## A MACHINE LEARNING BASED APPROACH TO DAMAGE LOCALISATION

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
The purpose of this project is to develop a model capable of localising damage within a complex structure, by using data that has been experimentally obtained from a network of sensors across a test-piece within a lab. In doing so, this model aims to provide maintenance operators and engineer's with a proof of concept for a machine learning based approach to maintenance and damage localisation, which could reduce the time and cost associated with maintenance activities such as visual inspections. Furthermore, by leveraging Bayesian Optimisation for model training, this model also attempts to reduce the volume of training data required for such modules, an obstacle that frequently prevents the application of machine learning in real engineering environments.

## DATA
The data used in this project was experimentally obtained during my time at university. Experiments where conducted whereby artificial Acoustic Emission events were initiated across a complex test-piece, and the difference in time of arrival values were recorded from the multiple acoustic emissions sensors across the test-piece, alongside the X,Y coordinates of where the artificial event took place across the structure. 

## MODEL 
This is a forward model, using an initial gaussian process regression model to learn the relationship between acoustic emission source location (X,Y coordinate - model input) and the difference in time taken (dTOA values) for acoustic emissions to arrive between two sensors. To predict damage source locations, this forward model is then used to predict a much denser grid of dTOA values across the test-piece. This reduces the amount of data required for model training, and allows unseen dTOA values to be placed according to where they are most likely to lie on this map. 

Each sensor pair must have its own seperate model. In combining multiple models on the same plot, it is then possible to triangulate a source location with much more accuracy than it would be possible with the data from just one sensor pair. 

In this instance, gaussian processes have been chosen because of their basis in Bayesian statistics. This allows for uncertainty to be propagated throughout the model. This means that predictions are not taken as definitive, which will provide operators with more use in practical applications.

Lastly, I have chosen to use a gaussian process regression model as these models are compatible within acquisition functions of bayesian optimisation, often acting as surrogate models. This project concludes by implementing bayesian optimisation to selectively sample points for model training, which aims to reduce the amount of training data required by models whilst informing operators of structural regions where their training data collection should be more targeted.

## HYPERPARAMETER OPTIMSATION
This model relies on using Gaussian Process Regression to allow for predictions of the damage source location to be made. I have chosen the Matern kernel for the implementation, which required for the tuning of 3 primary hyper-parameters. Namely, these are the signal variance, the length-scale and the noise variance. Using an approach discussed by Rasumussen and Williams, to tune these hyperparamters I have implemented a gradient based approach, aiming to find the combination of hyperparameters which minimise the model's Negative Log Marginal Likelihood. This was done through a combination of my scipy.minimise, using the NLML as the loss function argument in this case.

## RESULTS
This approach to damage localisation is hugely successful and can make predictions with minimal amount of error, this was assessed using the Root Mean Square Error, comparing the predicted location and the source location. One criticism of this approach is that it is quite computationally expensive, but the approach is reliant on developing multiple Gaussian Process Regression models. Alas, this is unavoidable. With that being said, the computational expense is not wasted, as predictions are made with good accuracy.
