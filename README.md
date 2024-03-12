# Neural Network Project
This project is developed by an MSC EE student at the Budapest University of Technology and Economics (BME). It is inspired by 3Blue1Brown's insightful article on backpropagation calculus, which can be found [here](https://www.3blue1brown.com/lessons/backpropagation-calculus).

## Project Overview
The implementation offers a fresh take on the backpropagation algorithm. The classes used throughout the project draw significant inspiration from the aforementioned article. 

### Features
The project includes versatile input and output vector generator functions to accommodate a wide range of data types, enhancing the neural network's applicability. Currently supported data types include:
- Folders of images (supported formats: .JPG, .PNG, .BMP, .TGA, .GIF)
- Single images
- Boolean vectors
- Vectors of strings
- Text data (.TXT) with customizable separators (e.g., "dog,dog,cat,dog,cat...")
- Chopped frames from an .MP4 video file (Note: Requires proper inclusion of OpenCV, defining `_VIDEO_`, and setup of compiler and linker)
- Numerical vectors of any type, convertible to other numerical vectors
- Time series data with selected window size determining the number of input neurons
- Audio files (.MP3)

### Basic and Example Usage
```cpp
vector inputs = convertInput(your_path); // Converts the specified data to neural network compatible inputs
vector expecteds = convertExpecteds(your_path); // Converts specified data to expected outputs
vector layerSizes = {L1, L2, ..., Ln}; // Define the number of neurons in each layer
vector activationFunctions = {"Relu", "Sigmoid", ..., "LeakyRelu"}; // Define activation functions for each layer

// Initialize the network with various parameters
Network network(layerSizes, "OptimizationMethod", "RegularizationMethod", "InitializationMethod", optionalDesiredPrecision, optionalDropoutRate);

// Train the network
network.trainNetwork("TrainingMethod", inputs, outputs, optionalEchoNumber);

// Test the network
unseenInputs = convertInput(your_path_to_other_inputs);
OPTIONAL: unseenExpecteds = convertExpecteds(your_path_to_other_expecteds);
network.testNetwork(unseenInputs, unseenExpecteds);


// This is an example of how you could use this repository
int main()
{
	try
	{
		// Input and output array's sizes should sta constant
		const unsigned inSize = 4; 
		const unsigned expectedSize = 10;

		// First define the number of neurons in each layer
		// each layer should have a numerical value greater then 0
		vector<unsigned> layerSizes = { inSize, 10, expectedSize };

		// Create network object, with the specified optimization and regularization and initialization techniques
		Network network(layerSizes, "none", "none", "Xavier");

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

		// Training network, with the provided data
		network.trainNetwork("sgd", inputArr, expArr, 100);

		//-----------------------------------//

		// Testing the network

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
