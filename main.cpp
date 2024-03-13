#include <iostream>
#include "Network.h"

template<typename T>
extern vector<vector<T>> inputConverter(const std::string& inputPath);

// Provided by an MSC EE student of Budapest University of Technology and Economics

// The backpropagation algorithm's implementation is highly based on this arcticle:
// http://neuralnetworksanddeeplearning.com/chap2.html

// Currently this project is in this state: WORK IN PROGRESS!
// Neuron, Edge, Layer and Network classes are implemented, but 
// they are not tested and not complete!
// This project is done by 1 person, if anybody has any suggestins, bugs
// and bug fix ideas, please feel free to conact me through the github repo

// The main concept is that the user only has to do 2 things:
// -Bring their input to some kind of numerical array
// -Bring their expected values to some kind of numerical array
// It is NOT necessary that your expected and input arrays to be 
// the same numerical type!
// Then train the network. Hopefully it will learn the pattern provided by your data
// The training process should only be done once, after that you can use the 
// loadFromFile() member function, then you can use your network.

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
		// Do NOT change INPUT_LAYER_SIZE and OUTPUT_LAYER SIZE or it's position
		vector<unsigned> hiddenLayerSizes = { INPUT_LAYER_SIZE, 10, 10, OUTPUT_LAYER_SIZE };

		// Create network object, with the specified optimization and regularization and initialization techniques
		Network network(hiddenLayerSizes,	// Hidden layer sizes
						"adam",				// Optimization technique
						"L1",				// Regularization technique
						"Xavier",			// Initialization technique
						inputArr,			// Vector of input frames
						expArr				// Vector of expected output frames
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
		network.trainNetwork("GD", inputArr, expArr, 10);

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