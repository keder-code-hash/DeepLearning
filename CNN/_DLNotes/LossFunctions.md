https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
https://keras.io/api/losses/

## Neural Network are genrally being trained by **Stochestic Gradiant Descent** and weights are being updated by using backpropagation algorithm.And we can choose a loss function for our model.
## In simple words, loss functions are required to calculate the gradiant and Gradients are used to update weights in the Neural Network.
<br>
<br>

# Loss Functions :
> The function we want to minimize or maximize is called the objective function or criterion. When we are minimizing it, we may also call it the cost function, loss function, or error function.
Now, in of CNN we object functions are actually being minimized by loss function or cost function.If
we choose the loss function incorrectly then then the parameter or features which are being extracted
for the minimizations of obejctive function may ybe wrong.Models are pretty much effected by this.

<br>
<br>
<br>
 
:tent:

<dl>
  <dt>Regression Loss Function</dt>
  <dt></dt>
  <dd>Mean Squared Error Loss</dd>
  <dd>Mean Squared Logarithmic Error Loss</dd>
  <dd>Mean absolute Error Loss</dd>
  <dd>Huber Loss</dd>
</dl>

:fire:

<dl>
  <dt>Binary Classification Loss Functions</dt>
  <dt></dt>
  <dd>Binary Cross-Entropy</dd>
  <dd>Hinge Loss</dd>
  <dd>SquaredHinge Loss</dd>
</dl>

:joy:

<dl>
  <dt>Multi-class Classification Loss Functions</dt>
  <dt></dt>
  <dd>Multi-Class Cross-Entropy Loss</dd>
  <dd>Sparse Multiclass Cross-Entropy Loss</dd>
  <dd>Kullback Leibler Divergence Loss</dd>
</dl>


# Regression Loss Function :

```
regression actually deals with a dependet variable(Y) and some independet variables(X1,X2,...,Xn)
```

## 1 .Mean Squared Error Loss :

```
Squared Error loss for each training example, also known as L2 Loss, is the square of the difference
between the actual and the predicted values.MSE denotes the mean of all these values.
```
### Expression : 

L=$(y-f(x))^2$
<br>
MSE=($\sum_{i=1}^{n} L_i$)/n

### Use when?
```md

Mean squared error is calculated as the average of the squared differences between the predicted and
actual values. The result is always positive regardless of the sign of the predicted and actual values 
and a perfect value is 0.0. The squaring means that larger mistakes result in more error than smaller 
mistakes, meaning that the model is punished for making larger mistakes.
```
**This types of loss function sholud not be used if data are scattered(outliers).**

## 2. Mean Squared Logerithmic Error Loss :

### Use when?
```md
There may be regression problems in which the target value has a spread of values and when predicting a 
large value, one may not want to penalize the model too much for predicting unscaled quantities 
directly. Relaxing the penalty on huge differences can be done with the help of Mean Squared Logarithmic 
Error.
As a loss measure, it may be more appropriate when the model is predicting unscaled quantities directly. 
Nevertheless, we can demonstrate this loss function using our simple regression problem.
```
### expression: 
![](https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/mean-squared-loss-error.PNG)

## 3. Mean Absolute Error Loss :

### Use when?
```
On some regression problems, the distribution of the target variable may be mostly Gaussian, but may 
have outliers, e.g. large or small values far from the mean value.
The Mean Absolute Error, or MAE, loss is an appropriate loss function in this case as it is more robust 
to outliers. It is calculated as the average of the absolute difference between the actual and predicted 
values.
```

## 4. Huber Loss :

In case of regression we have to deal with the outliers i.e. values far from the mean values of the data 
points. MSE are actully pretty good to handle outliers and MAE are good to avoid then to take a mean 
positive value. So we are going to take a middle between then and it can be calculated.



Using the MAE for larger loss values mitigates the weight that we put on outliers so that we still get a 
well-rounded model. At the same time we use the MSE for the smaller loss values to maintain a quadratic 
function near the centre.

**A comparison between L1 and L2 loss yields the following results:**

L1 loss is more robust than its counterpart.
On taking a closer look at the formulas, one can observe that if the difference between the predicted 
and the actual value is high, L2 loss magnifies the effect when compared to L1. Since L2 succumbs to 
outliers, L1 loss function is the more robust loss function.

L1 loss is less stable than L2 loss.
Since L1 loss deals with the difference in distances, a small horizontal change can lead to the 
regression line jumping a large amount. Such an effect taking place across multiple iterations would 
lead to a significant change in the slope between iterations.

On the other hand, MSE ensures the regression line moves lightly for a small adjustment in the data 
point.

Huber Loss combines the robustness of L1 with the stability of L2, essentially the best of L1 and L2 
losses. For huge errors, it is linear and for small errors, it is quadratic in nature.

Huber Loss is characterized by the parameter delta (𝛿). For a prediction f(x) of the data point y, with 
the characterizing parameter 𝛿, Huber Loss is formulated as:

![](https://miro.medium.com/proxy/1*0eoiZGyddDqltzzjoyfRzA.png)

![](https://miro.medium.com/max/960/1*qk317yM8Dfvg5FHaLkie4w.png)
|:--:| 
| *MAE (red), MSE (blue), and Huber (green) loss functions* |

# Binary Classification Loss Functions:

**classifications probelm involves predicting a discrete clas ouputs based upon some paramenters.So that a new unseen class can be easily classified.**

## 1. Binary Cross-Entropy:

Entropy is measured the randomness in the information and cross-entropy means the difference of 
randomness between the two random variable.




https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/