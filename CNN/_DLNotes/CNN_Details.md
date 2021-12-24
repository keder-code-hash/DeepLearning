<style>
.markdown-body{
    font-family: hack;
    font-size : 18px ;
    line-height : 1.8;
}
</style>

# <u>Convolutional Layer</u> : 

This layer is the first layer is used to extract various feature form input images.
In this layer,the **mathematical operation of convolution** is performed between the input image and 
filter of a particular size MXM.
By sliding the filter over the input image, the dot product is taken between **the filter and the parts of input image**
w.r.t. the size of the filter(MXM).

## Output : 
Output of this layer is a <i>**featured map(gives information about corners and edges).**</i>

### Function Used : 

```python
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```
# <u>Pooling Layer</u> :

1. A convolution layer is basically followed by a **Pooling Layer.**
2. It is used to reduce the convolved map in the network to redeuce the computatioal cost.
    It decreases the connections between the layer.
3. There are different type of pooling operations :-
    1. MaxPooling : - In this, largest element is taken from the map.
    2. AveragePooling : - Calculates the average.

<br/>
<hr>
<br/>

**Function** :

``` python
tf.keras.layers.MaxPool2D(
pool_size=(2, 2), strides=None, padding='valid', data_format=None,
**kwargs
)
```       

<hr>
<br/>
<br/>

### **pool_size** : 
A window of pool_size if defined.pool_size[1,2] indicates length of window row wise is 1 and
columnwise is 2.

### **Strides**  : 
Input window defined by the pool_size, gets shifted by the strides argument(strides=[1,2] 
means, widow get shifted by 1 according to row and 3 by column). 

### **padding** : 
Sometimes it is required to add some padding to the actual matrix.
It has two parameter.
1. **same** : no padding.
2. **valid** : generate auto padding.

    **Consider a situation**,
        Actual Mtrix size: 4X4
        pool_size: 2X2
        strides: 1X2

Now , In this case the last column is always considerd one time only but the others are considered multiple
    time.For this we introduce som epadding to row then last coulumn will be evaluated multiple time and result
    will be more effective.

### **data_format** : 
A string, one of channels_last (default) or channels_first. The ordering of the dimensions in 
the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while 
channels_first corresponds to inputs with shape (batch, channels, height, width). 




# <u>Dropout</u> :
Sometimes models return pretty accurate results on training dataset,Then there is a chance of overfit.So we have
to drop some neurons from our network.We can use the **dropout** in this context.

### Function Used :

```python
tf.keras.layers.Dropout(rate,noise_shape=None, seed=None,**kwargs)
```


## rate : 
The Dropout layer randomly sets input units **to 0 with a frequency of rate at each step during training time**,
which helps prevent overfitting. **Inputs not set to 0 are scaled up by 1/(1 - rate)** such that the sum over
all inputs is unchanged.
<br/>
<br/>
<br/>
<hr>

**Note** :
<pre> 
Note that the Dropout layer only applies when training is set to True such that no values are dropped during
inference. When using model.fit, training will be appropriately set to True automatically, and in other
contexts, you can set the kwarg explicitly to True when calling the layer.
</pre>
<hr>

## noise_shape :





# <u>Fully Connected Layer</u> : 

This layer consists of the weights and biases along with the neurons and used to connect diffeent layers.This
type of layers is usually placed before the output layer.

classification is also began form this layer.

### Function Used :

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

## <u> Activation Function </u> : 
![activation function](file:///D:/DeepLearning/CNN/_DLNotes/activation.png)

### <i>Softmax</i> :

The softmax function is used as the <mark>activation function in the output layer</mark> of neural 
network models that predict **a multinomial probability distribution.**

 
# <u> Some Important Terms </u>: 

## 1. Batch : 
The batch size is a hyperparameter that defines **the number of samples to work through**
    **before updating the internal model parameters**.
## 2. Epoch :
The number of epochs is a hyperparameter that defines the number times that the learning algorithm
    will work through the entire training dataset.
 


https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
http://neuralnetworksanddeeplearning.com/chap2.html

# BatchNormalizer :
