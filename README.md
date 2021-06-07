# Self Driving Car
![alt img](https://cdn-images-1.medium.com/max/868/0*7dReqQXElneHBWUr.jpg)<br>
Refer the [Self Driving Car Notebook](https://github.com/Samyakb50/Self-driving-car-using-deep-neural-network-technique/blob/master/Self_driving_car_using_simulator.ipynb) for complete Information <br>

* Design an autonomous car (AC),a vehicle that is adept at sensing its surroundings and navigating without human involvement and aim at training a model which would enable a car to run autonomously using behavioral cloning technique and convolutional neural network. 
* Used convolutional neural networks (CNNs) to map the raw pixels from a front-facing camera to the steering commands for a self-driving car. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings, on both local roads and highways. The system can also operate in areas with unclear visual guidance such as parking lots or unpaved roads.
* The system is trained to automatically learn the internal representations of necessary processing steps, such as detecting useful road features, with only the human steering angle as the training signal. We do not need to explicitly trained it to detect, for example, the outline of roads.

### Conclusions from the paper
* A small amount of training data from less than a hundred hours of driving was sufficient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions. 
* The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).
* More work is needed to improve the robustness of the network, to find methods to verify the robustness, and to improve visualization of the network-internal processing steps.
