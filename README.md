This is a mini-experiment that analysis the findings given in the research paper below using the MNIST fashion dataset.

[Understanding Deep Learning Requires Re-thinking Generalization](https://arxiv.org/pdf/1611.03530.pdf)

For a detailed understanding of the experiment look at the [Jupiter notebook](https://github.com/eshika289/Overfitting-and-Regularization-Analysis-of-Neural-Networks/blob/main/Overfitting_and_Regularization_Analysis_of_Neural_Networks.ipynb). The notebook also includes various charts/graphs like precision-recall curves, confusion matrices, loss/accuracy per iteration graphs.


# Results from Training A Model on the Images (no changes)

```
[train loss, train accuracy]: [0.24070537090301514, 0.9119833111763]
[test loss, test accuracy]: [0.3524151146411896, 0.8784999847412109]
Generalization Error 0.6475848853588104
```

The model performs well on both the test and training datasets. The accuracy and loss of the test and training sets are not too far apart indicating that the model generalizes well (generalization error of about 0.65)

# Results from Training A Model on Image with Shuffled Pixels

### Original Picture vs Shuffled Pixel Picture 
![download](https://github.com/eshika289/Overfitting-and-Regularization-Analysis-of-Neural-Networks/assets/102436075/3461f977-e39c-4e9b-9a4a-f0bbf1782133)![download](https://github.com/eshika289/Overfitting-and-Regularization-Analysis-of-Neural-Networks/assets/102436075/c6b5b4b0-ab8f-44b1-ab73-f8592bee8cb3)

## Without Early Stopping
```
1875/1875 [==============================] - 5s 3ms/step - loss: 0.2723 - accuracy: 0.9109
313/313 [==============================] - 1s 2ms/step - loss: 6.6578 - accuracy: 0.2287
[train loss, train accuracy]: [0.2723132371902466, 0.9109166860580444]
[test loss, test accuracy]: [6.65777587890625, 0.22869999706745148]
Generalization Error 5.65777587890625
```
Here we see that the training loss and accuracy are very similar to loss and accuracy of the model trained on the unaltered images. This indicates that the model can learn the label of an image even if the pixels of the image are randomly shuffled.

One thing to note here is that it took the model about 30 epochs to reach about 88% accuracy with the shuffled pixels, but only took about 10 epochs to reach about 88% accuracy with the original image.  This indicates that the target function(relationship between images and labels) of the shuffled pixel problem may be more complex than the target function of the original problem.

However, the test loss and accuracy indicate that the model performs poorly. It does not generalize well as it overfits the training dataset. The generalization error is about 5.6.

## With Early Stopping
```
1875/1875 [==============================] - 4s 2ms/step - loss: 1.4310 - accuracy: 0.4725
313/313 [==============================] - 1s 2ms/step - loss: 2.0043 - accuracy: 0.2379
[train loss, train accuracy]: [1.4309738874435425, 0.47253334522247314]
[test loss, test accuracy]: [2.004284143447876, 0.2379000037908554]
Generalization Error 1.004284143447876
```
The loss and accuracy of the training and test sets indicate that the model does not perform well. This is because there is not a strong relationship between the shuffled pixel images and the labels.

With early stopping and a validation set, we reduced the overfitting of the training set indicated by the lower generalization error.

# Results from Training A Model on a Set with random labels

The labels were shuffled in the training set.  Therefore, there is no "real" relationship between the images (x) and their labels (y).


## Without Early Stopping

```
1875/1875 [==============================] - 4s 2ms/step - loss: 1.0849 - accuracy: 0.5929
313/313 [==============================] - 1s 2ms/step - loss: 11.7956 - accuracy: 0.1205
[train loss, train accuracy]: [1.0849181413650513, 0.5929333567619324]
[test loss, test accuracy]: [11.795574188232422, 0.12049999833106995]
Generalization Error 10.795574188232422
```
Here we see that the training loss and accuracy make it seem like the model has a decent performance with an accuracy of about 60%. This indicates that the model can learn the label of an image even if the labels are completely random.

One thing to note here is that model took about 300 epochs and has an additional hidden layer. This all represents the fact that the target/ideal function is more complex than the ones above. This is because the labels are randomized and there is no "real" relationship between the images and the labels. Despite the fact that there is no real relationship, the model has a 60% accuracy on the train test. 

How is this possible? The paper explains that the neural network is essentially just memorizing the training data exactly. This showcases both the extreme overfitting that is happening here and the extreme complexities that neural networks are able to model (high effective capacity as the paper puts it).

The test loss and accuracy indicate that the model performs poorly. It does not generalize well as it overfits the training dataset. The generalization error is about 10.8.

## With Early Stopping
```
1875/1875 [==============================] - 6s 3ms/step - loss: 2.3159 - accuracy: 0.1000
313/313 [==============================] - 1s 3ms/step - loss: 2.3160 - accuracy: 0.1000
[train loss, train accuracy]: [2.3159115314483643, 0.10001666843891144]
[test loss, test accuracy]: [2.315950632095337, 0.10000000149011612]
Generalization Error 3.910064697265625e-05
```
The loss and accuracy of the training and test sets indicate that the model does not perform well. This is because there is no "real" relationship between the images and the random labels.

With early stopping and a validation set, we reduced the overfitting of the training set indicated by the lower generalization error.

# Regularization (Early Stopping & Validation Set) for Neural Networks

The high loss and low accuracy of the training and test sets for the shuffled pixels model indicate that the model does not generalize well. This is because there is not a strong relationship between the shuffled pixel images and the labels.
The high loss and low accuracy of the training and test sets for the shuffled labels model indicate that the model does not generalize well. This is because there is no "real" relationship between the images and the shuffled labels.

With early stopping and a validation set, we reduced the overfitting of the training set indicated by the lower generalization errors.

The paper discusses that we may be able to improve the generalization error using early stopping. This experiment aims to showcase that finding through the examples above.

# Hyperparameter Optimization Strategy

The hyperparameter optimization strategy I am using is Hyberband. This uses the Random Search algorithm with some additional features like resource allocation and successive halving to help speed up the search process. It can also use the early stopping method where it stops testing a hyperparameter set if a certain metric (I used validation loss) has not improved over the past few epochs.

The Random Search algorithm searches a random combination of the sets of hyperparameters and chooses the best performing hyperparameters among the ones tested. This method doesn't guarantee that we get the best hyperparameters, but it runs much faster than the exhaustive Grid Search algorithm.

The hyperband hyperparameter tuner uses the `model_builder` function to build each model with a different set of hyperparameters. The objective function in this tuner is the validation loss meaning it will choose a set of hyperparameters that results in a model that minimizes the validation loss.

I also use an early stopping callback which is monitoring the validation accuracy across epoch runs and if the validation accuracy has not improved by much during the past 5 epochs, it will stop searching that specific set of hyperparameters.

Once the hyperparameters that minimize the validation loss are found, I run a model with the optimized hyperparameters for 50 epochs and chooses the epoch number that minimizes the validation loss. Essentially, it is treating the number of epochs as a hyperparameter. The aim of doing this is to avoid running the model for too long which could lead to overfitting the training set.
