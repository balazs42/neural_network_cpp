#include <iostream>
#include "Network.h"

// Provided by an MSC EE student of Budapest University of Technology and Economics

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
		// Input and output array's sizes should sta constant
		const unsigned inSize = 4; 
		const unsigned expectedSize = 10;

		// First define the number of neurons in each layer
		// each layer should have a numerical value greater then 0
		vector<unsigned> layerSizes = { inSize, 100, 100, expectedSize };

		// Create network object, with the specified optimization and regularization and initialization techniques
		Network network(layerSizes, "adam", "L1", "Xavier");

		// You can randinit all edges and biases, in the network, although it is not
		// necessary, the constructor already does it, but if for some reason you want
		// to reinit the network you have the possibility
		network.randInitNetwork();

		// You can set different activation functions to the layers
		// you cannot pass more strings then the number of layers
		vector<string> activationFunctions = { "Sigmoid", "Relu", "Sigmoid", "Tanh" };

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

		// Creating input vector of frames
		vector<vector<unsigned>> inputArr;

		// Adding frames to input array
		inputArr.push_back(firstFrame);
		inputArr.push_back(secondFrame);
		inputArr.push_back(thirdFrame);
		inputArr.push_back(fourthFrame);

		//-----------------------------------//

		// Creating artificial expected array
		vector<unsigned> firstExpected =  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		vector<unsigned> secondExpected = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
		vector<unsigned> thirdExpected =  {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
		vector<unsigned> fourthExpected = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};

		vector<vector<unsigned>> expArr;

		expArr.push_back(firstExpected);
		expArr.push_back(secondExpected);
		expArr.push_back(thirdExpected);
		expArr.push_back(fourthExpected);

		//-----------------------------------//

		// Training network, with the provided data
		network.trainNetwork("GD", inputArr, expArr, 100);
	}
	catch (std::out_of_range)
	{

	}
	catch(...)
	{

	}



	return 0;
}