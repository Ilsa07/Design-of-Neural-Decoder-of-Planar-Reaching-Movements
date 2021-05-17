# Design of Neural Decoder of Planar Reaching Movements

This project was done as part of the Brain Machine Interfaces course at Imperial College London and was carried out in a group of four students, which I was a member of. It follows the structure of an IEEE paper, which can be read in the **.pdf** within the directory.

## Abstract
The following paper outlines a trajectory estimator neural decoder, which was trained and evaluated on spike recordings of a monkey's brain while it was performing reaching movements in eight different directions. The developed algorithm performs data pre-processing and classification of neural signals corresponding to the planning phase, before the reaching movement is executed, to estimate the movement direction of the arm. Once movement direction is classified, regression is performed with a Binary Decision Tree to continuously estimate the position of the monkey's hand and post-processing to increase the regression accuracy. The methods in this paper focus on both the accuracy and the computing time of the algorithm. The accuracy of the reaching angle classification was 99.3\%, and the Root Mean Square Error (RMSE) of the decoder was 8.99, with a run time of 17.18 seconds.
