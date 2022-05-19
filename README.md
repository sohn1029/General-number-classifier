# General-number-classifier
When learning deep learning, we need a dataset, training code, and a deep learning model. In this process, we have used MNIST(Modified National Institute of Standards and Technologh database), the most accessible dataset, and a model that matches it. However, the goal of this term project is to train a neural network with a great genralization performance. So to do this, I had to come up with a new dataset, training code, and model. Effort for this are as follows.

## Dataset
At first, I thought I would need some kind of dataset other than MNIST to achieve what I wanted for this project. So I found a dataset called SVHN(Street View House Numbers), and got images and mat file. Through this, I can get MNIST and SVHN. Next, image augmentation was attempted to obtain general results for these data. The attemped augmentation technique is as follows.
-	Rotate
-	Zoom
-	Move
-	Various Background(Simple, Pretty, Noisy, Figure)

### **figure 1**

 ![make_rgb](https://user-images.githubusercontent.com/31722713/169290251-2c695c5f-2291-4293-bf68-b3f8274cbea4.png)

The above code is an augmentation code for MNIST, but we can make SVHN augmentation dataset with a similar technique. With the code above, we can get over 500000 data. A part of the dataset is as follows.

### **figure 2**
 ![fig2](https://user-images.githubusercontent.com/31722713/169290813-5231346b-5a2f-4aaa-ae02-e1b90bcfb160.png)

In addition to this dataset, a small amount of unique fonts are added to the training data.

## Model
The figure showing the overall appearance of the model is shown in <figure 3>.

### **figure 3**
 ![fig3](https://user-images.githubusercontent.com/31722713/169291034-bfd7dabf-fd91-47da-adde-090fa7a245c2.png)

A lenet-like structure leading to fully connected layer past the existing convolution layer not only increases the size of the model, but also shows that the accuracy is quite low. And since the model to be implemented has to deal with more general situations, I need to change the model more innovatively. So the model I came up with is <figure 3>. When I was designing the model, I wanted to be able to ensemble the results of each convolution layer whenever the input data passed through the convolutional layer. This is a model that expresses this wish well. It was also good to use the results of each convolution layer without wasting because the size of the parameters had to be less than 5MB.

## Train
The training process is divided into training phase, validation phase and test phase. During the epoch, the model is stored in model.py, and when the validation cost is the smallest, it is stored in best_model.pt. Prior to training, the model, criterion, and optimizer were defined as shown in figure 4.

### **figure 4**
 ![fig4](https://user-images.githubusercontent.com/31722713/169291121-3d658044-4827-4c34-b6c7-ed98fa4beb05.png)

The model was given the cuda option so that it could be trained with the gpu, the optimizer was set to adam and L2 regularization was applied.

## Result
As we saw in the process of building the dataset, this model performs prediction using a grayscale image with channel 1. For this purpose, like the dataset for training, the image used for testing must be preprocessed in grayscale. Learning was conducted with a learning rate of 0.0009 and a batch size of 32. The training and test results are shown in <figure 5>.

### **figure 5**
![fig5](https://user-images.githubusercontent.com/31722713/169291215-ac64e0a6-0d85-4ea1-8d31-ddb6c42261d3.png)

If you look at figure 5, you can see that the dev cost is lower than the train cost, which I think is because training and testing give different results due to the dropout layer. In order for the two costs to become similar or for the train to have a low cost, the train had to be continued for a long time, more than 100 epochs, but it was difficult in terms of time. If I could spend more time training, I think I would get better results.
