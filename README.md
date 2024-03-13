# Neural Network Project
This project is developed by an MSC EE student at the Budapest University of Technology and Economics (BME). It is inspired by 3Blue1Brown's insightful article on backpropagation calculus, which can be found [here](https://www.3blue1brown.com/lessons/backpropagation-calculus). And the amazing article of Michael Nielsen  "Neural Networks and Deep Learning" - http://neuralnetworksanddeeplearning.com/chap2.html

## Project Overview
This repository houses a comprehensive C++ implementation of a neural network, designed and developed with a focus on flexibility, performance, and ease of use. Inspired by the foundational concepts outlined in Michael Nielsen's "Neural Networks and Deep Learning", this project seeks to provide a hands-on approach to understanding and applying neural networks in various domains, including but not limited to image recognition, data analysis, and natural language processing. The classes used throughout the project draw significant inspiration from the aforementioned article. 

### Features
 - Configurable Architecture: Design your neural network architecture with ease, specifying the number of layers, neurons per layer, and activation functions.
 - Diverse Activation Functions: Choose from a variety of activation functions, including Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and Swish, to best fit your application.
 - Advanced Optimization Algorithms: Leverage sophisticated optimization techniques such as Adam, RMSProp, Adagrad, Adadelta, NAG, and Adamax for efficient training.
 - Regularization Techniques: Implement L1 and L2 regularization to combat overfitting and improve generalization.
 - Intelligent Weight Initialization: Utilize He and Xavier initialization methods to start training with strategically chosen weights.
 - OpenMP Parallelization: Benefit from enhanced performance on multicore processors through OpenMP-based parallelization of key operations.
 - Comprehensive Input Conversion: Prepare image, audio, and text data for training with versatile input conversion utilities.
	- Folders of images (supported formats: .JPG, .PNG, .BMP, .TGA, .GIF)
	- Single images
	- Boolean vectors
	- Vectors of strings
	- Text data (.TXT) with customizable separators (e.g., "dog,dog,cat,dog,cat...")
	- Chopped frames from an .MP4 video file (Note: Requires proper inclusion of OpenCV, defining `_VIDEO_`, and setup of compiler and linker)
	- Numerical vectors of any type, convertible to other numerical vectors
	- Time series data with selected window size determining the number of input neurons
	- Audio files (.MP3)
	- Image reconstruction to .BMP

### Prerequisites
A C++ compiler supporting C++17 and OpenMP (e.g., GCC, Clang).
  
## Project Structure
The implementation is modular, organized into several key components to ensure clarity and maintainability:
- Neuron: Defines the fundamental building block of the network, encapsulating the activation function, bias, gradient, and delta calculations, and other optimization required variables.
- Layer: Manages collections of neurons, representing discrete layers within the neural network architecture.
- Edge: Represents connections between neurons, including weight management and optimization-related properties.
- Network: Orchestrates the neural network, managing layers, forwarding inputs, backpropagation, and training.
- inputConverter: Offers utility functions for converting diverse data types (images, audio, text) into a standardized format for network training.
  
### Brief example
- string `animalFolder` = "\path\animalFolder"; // Animal folder = {dogPic1.jpg, dogPic2.jpg, dogPic3.png, catPic1.jpg, dogPic4.jpg ... }
- string `expectedAnimalsTxt` = "\path\expectedAnimals.txt"; // txt = {"dog,dog,dog,cat,dog..."}
- vector<vector<float> `inputFrames` = convertInput(animalFolder);
- vector<vector<float>> `expValues` = convertInput(expectedAnimalsTxt);
- Other initializations
- `Network network();` // Initialize with corresponding parameters (SEEN BELOW)
- `network.trainNetwork(inputFrames, expValues);`

#### Basic and Example Usage
```cpp
vector inputs = convertInput(your_path); // Converts the specified data to neural network compatible inputs
vector expecteds = convertExpecteds(your_path); // Converts specified data to expected outputs
vector layerSizes = {L1, L2, ..., Ln}; // Define the number of neurons in each layer
vector activationFunctions = {"Relu", "Sigmoid", ..., "LeakyRelu"}; // Define activation functions for each layer

// Initialize the network with various parameters
Network network(layerSizes, "OptimizationMethod", "RegularizationMethod", "InitializationMethod", inputs, outputs, optionalDesiredPrecision, optionalDropoutRate);

// Train the network
network.trainNetwork("TrainingMethod", inputs, outputs, optionalEchoNumber);

// Test the network
unseenInputs = convertInput(your_path_to_other_inputs);
OPTIONAL: unseenExpecteds = convertExpecteds(your_path_to_other_expecteds);
network.testNetwork(unseenInputs, unseenExpecteds);
```cpp

// This is an example of how you could use this repository
int main()
{
	try
	{
		//-----------------------------------//

		// This is an artifically created input array, you have to bring your input 
		// to some numerical type array, then pass it to the network.

		// Creating artificial input frames
		vector<unsigned> firstFrame =  {1, 2, 3, 4};
		vector<unsigned> secondFrame = {2, 3, 4, 5};
		vector<unsigned> thirdFrame =  {10, 9, 8, 11};
		vector<unsigned> fourthFrame = {1, 0, 12, 3};
		vector<unsigned> fifthFrame = { 4, 4, 4, 4 };
		vector<unsigned> sixthFrame = { 0, 0, 0, 0 };
		vector<unsigned> seventhFrame = { 0, 3, 6, 9 };
		vector<unsigned> eightFrame = { 0, 2, 3, 4 };

		// Creating input vector of frames
		vector<vector<unsigned>> inputArr;

		// Adding frames to input array
		inputArr.push_back(firstFrame);
		inputArr.push_back(secondFrame);
		inputArr.push_back(thirdFrame);
		inputArr.push_back(fourthFrame);
		inputArr.push_back(fifthFrame);
		inputArr.push_back(sixthFrame);
		inputArr.push_back(seventhFrame);
		inputArr.push_back(eightFrame);

		// Or optionally you can convert input from filesystem
		//inputArr = inputConverter<unsigned>("your_file_path");		// You can provide a folder path or simple .type files. USABLE: [.JPG, .PNG, .BMP, .TGA, .GIF, .MP3, .TXT, (.MP4)]

		//-----------------------------------//

		// Creating artificial expected array
		vector<unsigned> firstExpected =  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		vector<unsigned> secondExpected = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
		vector<unsigned> thirdExpected =  {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
		vector<unsigned> fourthExpected = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
		vector<unsigned> fifthExpected =  {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
		vector<unsigned> sixthExpected =  {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
		vector<unsigned> seventhExpected ={0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
		vector<unsigned> eightExpected =  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		vector<vector<unsigned>> expArr;

		expArr.push_back(firstExpected);
		expArr.push_back(secondExpected);
		expArr.push_back(thirdExpected);
		expArr.push_back(fourthExpected);
		expArr.push_back(fifthExpected);
		expArr.push_back(sixthExpected);
		expArr.push_back(seventhExpected);
		expArr.push_back(eightExpected);

		// Or optionally you can convert expected from filesystem
		// expArr = inputConverter<unsigned>("your_file_path");		// You can provide a folder path or simple .type files. USABLE: [.JPG, .PNG, .BMP, .TGA, .GIF, .MP3, .TXT, (.MP4)]

		//-----------------------------------//

		// First define the number of neurons in each layer
		// each layer should have a numerical value greater then 0
		vector<unsigned> layerSizes = { INPUT_LAYER_SIZE, 5, OUTPUT_LAYER_SIZE };

		// Create network object, with the specified optimization and regularization and initialization techniques
		Network network(layerSizes,		// Layer hidden sizes
						"none",			// Optimization technique
						"none",			// Regularization technique
						"Xavier",		// Initialization technique
						inputArr,		// Vector of input frames
						expArr			// Vector of expected output frames
		);

		// You can randinit all edges and biases, in the network, although it is not
		// necessary, the constructor already does it, but if for some reason you want
		// to reinit the network you have the possibility
		//network.randInitNetwork();

		// You can set different activation functions to the layers
		// you cannot pass more strings then the number of layers
		vector<string> activationFunctions = { "Relu", "Sigmoid", "Sigmoid" };

		// You can set all the network to 1 specific activation function
		network.setAllActivationFunctions("Relu");

		// Or by passing the vector as an argument, you can set different
		// activation functions to different layers
		//network.setLayerActivationFunctions(activationFunctions);

		//-----------------------------------//

		// Training network, with the provided data
		network.trainNetwork("sgd", inputArr, expArr, 100);

		//-----------------------------------//

		// Testing the network
		const int inSize = 4;
		const int expectedSize = 10;

		// Creating artificial test frames
		unsigned testFrame1[inSize] = { 1, 2, 3, 4 };
		unsigned testFrame2[inSize] = { 1, 0, 11, 3 };

		vector<unsigned*> testFrames;

		testFrames.push_back(testFrame1);
		testFrames.push_back(testFrame2);

		// Creating artificial expected arrays, for error calculation
		unsigned testExpected1[expectedSize] = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		unsigned testExpected2[expectedSize] = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };

		vector<unsigned*> testExpArr;

		testExpArr.push_back(testExpected1);
		testExpArr.push_back(testExpected2);

		// Start test, function will provide error data on
		// Standard output
		network.testNetwork(testFrames, inSize, testExpArr, expectedSize);
	}
	catch (std::out_of_range)
	{

	}
	catch(...)
	{

	}



	return 0;
}
