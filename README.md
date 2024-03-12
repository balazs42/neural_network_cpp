Provided by an MSC EE student of Budapest University of Technology and Economics (BME)

The implementation is based on 3Blue1Brown's article on the topic: https://www.3blue1brown.com/lessons/backpropagation-calculus 

My take on the backpropagation algorithm, classes used for the project are highly based on the article.

Provided input and output vector generator functions, currently you can use:
 - Folder of images of types: .MP3, .PNG, .BMP, .TGA, .GIF
 - Single images 
 - Boolean vectors
 - Vector of strings
 - From a .TXT file reading every entry separated by a specific separator, for example tmp.txt = "dog,dog,cat,dog,cat..."
 - Provide an .MP4 video file and chop it up into frames, then use that as the input (openCV must be included to the project properly, and you have to define _VIDEO_, and setup compiler and linker accordingly)
 - Any type of numerical vector, and they can be converted to any specific other numerical vector
 - Your time series data, with a selected window, basically that will be the frame size, and this determines the number of input neurons

Basic usage pseudocode:
 - inputs = convertInput(your_path);
 - expecteds = convertExpecteds(your_path);
 - vector layerSizes = {x1, x2, ... , xn);		// xn is the number of neurons in each layer, by this you define how many layers there will be 
 - vector activationFunctions = {"Relu", "Sigmoid", ... , "LeakyRelu"}; // NOTE: You have to provide as many strings here as many layers you defined, since this will set layer activations

 - Network network(layerSizes, "OptimizationMethod", "RegulariaztionMethod", "InitializationMethod", optionalDesiredPrecision, optionalDropoutRate);	// These parameters should be provided to create the netwrok. You can set optimization(Adam, Adamax, etc.), regularization(L1, L2) and initialization(We, Xavier, Rand) methods to "none", by default it will use then no opt., reg. and will set initial weights and biases randomly. Optional variables can be left out.

 - network.trainNetwork("TrainginMethod", inputs, outputs, optionalEchoNumber);	// Training method can be: GradientDescent(gd), StochasticGradienDescent(sgd), BatchGradientDescent(bgd), MinibatchGradientDescent(mgd).

 Currently this project is in this state: WORK IN PROGRESS!
 Neuron, Edge, Layer and Network classes are implemented, but 
 they are not tested and not complete!
 This project is done by 1 person, if anybody has any suggestins, bugs
 and bug fix ideas, please feel free to conact me through the github repo

 The main concept is that the user only has to do 2 things:
 
 -Bring their input to some kind of numerical array.
 
 -Bring their expected values to some kind of numerical array.
 
 It is NOT necessary that your expected and input arrays to be 
 the same numerical type!
 Then train the network. Hopefully it will learn the pattern provided by your data
 The training process should only be done once, after that you can use the 
 loadFromFile() member function, then you can use your network.

 This is an example of how you could use this repository

<code>
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
		vector<unsigned> layerSizes = { inSize, 50, 50, expectedSize };

		// Create network object, with the specified optimization and regularization techniques
		Network network(layerSizes, "adam", "none", "none");

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
		network.setLayerActivationFunctions(activationFunctions);

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
		network.trainNetwork("SGD", inputArr, expArr, 100);
	}
	catch (std::out_of_range)
	{

	}
	catch(...)
	{

	}



	return 0;
}
</code>
