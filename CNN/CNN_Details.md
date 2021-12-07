# Pooling Layer :

1. A convolution layer is basically followed by a **Pooling Layer**
2. It is used to reduce the convolved map in the network to redeuce the computatioal cost.
    It decreases the connections between the layer.
3. There are different type of pooling operations :-
    1. MaxPooling : - In this, largest element is taken from the map.
    2. AveragePooling : - Calculates the average.
<br/>
<hr>
<br/>

**Function** :
<code>
    tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None,
    **kwargs
    )
</code>

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

