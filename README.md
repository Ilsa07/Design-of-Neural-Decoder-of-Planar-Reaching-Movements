# BMI_first_place_team

**To do**
- Experiment more with the velocity feature. Look in detail how it affects performance
- Build a better classifier: Increase the number of trials used for training it, but also reduce the number of segments used from each trial. Identify the segments that have the best predictive performance for the classifier.
- Build a better regression model. RSVM is probably not the way. Look at Kalman filters, Wiener filters, and feedforward NN.
- There seets to be overshooting at the end of the trajectories. Figure out what causes this. Potentially clip the prediction range for each iteration to prevent overshooting.
- Explore ultra fast method with very low dimensionality (through PCA).
- Explore more dimensionality reduction techniques for this type of dataset.
