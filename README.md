# Self Driving Car
This project aims to clone a driving style of a person into a self-driving car on a simulation track using deep learning, as can be seen on the following two animations:
![alt img](https://cdn-images-1.medium.com/max/868/0*7dReqQXElneHBWUr.jpg)<br>
Refer the [Self Driving Car Notebook](https://github.com/Samyakb50/Self-driving-car-using-deep-neural-network-technique/blob/master/Self_driving_car_using_simulator.ipynb) for complete Information <br>

* Design an autonomous car (AC),a vehicle that is adept at sensing its surroundings and navigating without human involvement and aim at training a model which would enable a car to run autonomously using behavioral cloning technique and convolutional neural network. 
* Used convolutional neural networks (CNNs) to map the raw pixels from a front-facing camera to the steering commands for a self-driving car. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings, on both local roads and highways. The system can also operate in areas with unclear visual guidance such as parking lots or unpaved roads.
* The system is trained to automatically learn the internal representations of necessary processing steps, such as detecting useful road features, with only the human steering angle as the training signal. We do not need to explicitly trained it to detect, for example, the outline of roads.

### Nvidia Model
* The architecture is a convolutional neural network (CNN) that uses 3 layers with 5x5 convolution and max pooling with 2x2 block and 2x2 downsizing, 3 layers with 3x3 convolution and max pooling with 2x2 block and no downsizing, flattening layer and 5 dense fully-connected layers. Mostly ReLU activation is used though last 3 layers use linear activation as it was providing better results. There is also one drop-out layer that similarly to max pooling reduces overfitting. 

### Why use elu Activation function
* Gradient of relu function in negative region is zero. Gradient tend to get smaller and smaller as we keep on moving backward. Backpropogation uses gradient value to change value of weight. Weight will not change.  So network refuse to learn or learn drastically slow. But elu activation give non zero value in negative region. It will decrease error. We always have change to learn and recover.

### Training
The training itself takes random batch of images from all three cameras and their corresponding steering values:

Both left and right camera images adjust the steering angle value a bit - the assumption is that when the front camera sees an image similar to our training left/right image, its steering angle must be corrected as it would otherwise run into the risk of escaping the road. This however causes a tricky situation where car might be driving from left to right and back on straights due to this enforced steering intervention. From the experience, this tends to get better with more training epochs.

### Other Larger Datasets you can train on
(1) Udacity: https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5<br>
70 minutes of data ~ 223GB<br>
Format: Image, latitude, longitude, gear, brake, throttle, steering angles and speed<br>
(2) Udacity Dataset: https://github.com/udacity/self-driving-car/tree/master/datasets [Datsets ranging from 40 to 183 GB in different conditions]<br>

### Conclusion
* A small amount of training data from less than a hundred hours of driving was sufficient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions. 
* The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).
* More work is needed to improve the robustness of the network, to find methods to verify the robustness, and to improve visualization of the network-internal processing steps.

### What's next?
* It would be great to train model with speed, acceleration, gyroscope details etc. Also, driving a real car in the world outside and using the footage instead of simulator. However, as photorealistic rendering is getting widespread, it might be sufficient to use some new simulator as well.

* Results of this project could be also applied to real world, e.g. building an own Jetson-based racecar like the Racecar from MIT, collect training data by driving it around, use home supercomputer to train Dave-2 network and then let it drive autonomously and observe what could be improved.

* Another interesting idea would be to extend this model to 3D for controlling a drone, for example a delivery drone that can have its flights restricted to a well-known area and where a smaller neural network is sufficient.

### Credits & Inspired By
(1) https://github.com/SullyChen/Autopilot-TensorFlow<br>
(2) Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]<br>
(3) Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ <br>
(4) https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/
