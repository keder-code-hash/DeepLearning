dataset :
training : https://storage.googleapis.com/kagglesdsdata/competitions/3362/31148/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1640149257&Signature=Nu7l7S4GpQIc2vLRDe9pAm8HiEidGS2QYC5ZXyGdDllPDADPuUocT7h877hJQvlJbkJViarPHIzLZ8mY4BAgAV8DR2FHbyum9NyOI6H03HgC7%2FDGD2C3WdfHWTjwGi6YhKBKJqQgOm5PqYFZr1kpJVWHq6pxQk%2BfVVFWj0hiTmvQgfB1iIwfRwe8uyXk5GQwln8V4qV3%2BI3AoXQSMJ5U324ExTROnX%2BC2JK2yhiRkazMN10juJfBc5nL1HDfe7p40GCuXIIzfY%2BCuaxByBrLOTLo%2FaXJ9kW9YiP%2F9NMpoXV6d3JJZwGKeVaa7KDOY8uPwRAZeffZufuXdKyZKltJeg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.zip
testing :
https://storage.googleapis.com/kagglesdsdata/competitions/3362/31148/test1.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1640149532&Signature=Qf2jWQycOlSLwK1fmuPT9evBIjSyBaW3acbiLoalvbtPd5cfSc2WYFKwgjkR0Xtl%2BzzjP0OCbkU9T0ICNNjxHyRKPOVkg7485OpOdmM207h1owo8smbRZBayEWpgDDRGwuXCWE%2B2MDbr08CmPdTqijHsz%2FAS9QyHsMgL8NLrdtoLuzfZ%2Br4sb6SUuZ4P%2BnmqGlgaCqSpGXIoibnAn9fZQ5cMuUJ%2FpfG0EIgQuM6kpXZ%2FdTpBcE3CP1mK6bxrr1T4AX0pc%2FZKlfJj4rHb4XQgOYwp5%2FYq38hnuNAvpPhgdwD4DjjjygVsw9hQoxiPUyOpVQI3I8xsAbmZRV373H0rZQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest1.zip


### channel : 
In case B/W image we can represent in a 2D matrix where each of the cell contains a specific value form 
0-255.But in case of a color image the representation should be different,**(height*width*channel)**.
here channel depends on the RGB here cnhannel should be 3.It actually gives a multidimenssional 
representation to each one of the cell.The channel no can vary.
We could think of the hidden representations as comprising a number of two-dimensional grids stacked on
top of each other. As in the inputs, these are sometimes called channels.

Now when the input data conatins multiple channels,the kernel also shold conatins as the same no of  
channels to extract the features.
Whenever the **channel_no > 1** we require total channel_no of kernel of size height*weight.As the each 
input image and the convolutional kernel conatins the same no of channel we can preform 
**cross-correlation** operation to each of the two-dimensional tensor of the convolutional kernel and 
two-dimensional tensor of the input image for the each of channel>adding up all the results for we can 
get a 2D matrix again.

```md
1. cross-correlation:
cross-correlation is a measure of similarity of two series as a function of the displacement of one relative to the other. This is also known as a sliding dot product or sliding inner-product.
details : https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
```  
![explain](https://d2l.ai/_images/conv-multi-in.svg)

### reference :
https://d2l.ai/chapter_convolutional-neural-networks/channels.html

---

### BatchData :
When the training dataset is to large then this tyof data is used.It will select the specific amount on 
data based upon the parameters(batch_size) and maintain the whole thing in a proper datastructure.
In case of keras **image_dataset_from_directory** will give a BatcahDataset in **tf**.
**Batch Size** denotes the no. of data to select from original to generate a single data on batch.
```
[1,2,3,4,5,6,12,7,8,9,11]
here batch_size=3
[[1,2,3],[4,5,6],[12,7,8],[9,11]] will be returned as a batch dataset.
```

---

``` Python
import matplotlib.pyplot as plt
train_img_directory="data/traindata/"
# interpolation : https://www.tensorflow.org/api_docs/python/tf/image/resize
# seed : Optional random seed for shuffling and transformations.
image_dataset=keras.utils.image_dataset_from_directory(
    directory=train_img_directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
print(image_dataset)
plt.figure(figsize=(10, 10))
class_names = image_dataset.class_names
for images, labels in image_dataset.take(1):
    for i in range(32):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```