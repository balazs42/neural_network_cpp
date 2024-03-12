Provided by an MSC EE student of Budapest University of Technology and Economics (BME)

The implementation is based on 3Blue1Brown's article on the topic: https://www.3blue1brown.com/lessons/backpropagation-calculus 

My take on the backpropagation algorithm, classes used for the project are highly based on the article.

Provided input and output vector generator functions, currently you can use:
 - Folder of images of types: .JPG, .PNG, .BMP, .TGA, .GIF
 - Single images 
 - Boolean vectors
 - Vector of strings
 - From a .TXT file reading every entry separated by a specific separator, for example tmp.txt = "dog,dog,cat,dog,cat..."
 - Provide an .MP4 video file and chop it up into frames, then use that as the input (openCV must be included to the project properly, and you have to define _VIDEO_, and setup compiler and linker accordingly)
 - Any type of numerical vector, and they can be converted to any specific other numerical vector
 - Your time series data, with a selected window, basically that will be the frame size, and this determines the number of input neurons
 - Audio files (.MP3)

Basic usage pseudocode:
<code>

 - vector inputs = convertInput(your_path);	// NOTE: Converts the provided folder/.TXT/.MP3/.JPG/.PNG/.BMP/.TGA/.GIF to Neural Network compatible input or output
 - vector expecteds = convertExpecteds(your_path);	// NOTE: Does the same for expected outputs
 - vector layerSizes = {L1, L2, ... , Ln);		// NOTE: Ln is the number of neurons in each layer, by this you define how many layers there will be as N and the number of neurons in each layer L
 - vector activationFunctions = {"Relu", "Sigmoid", ... , "LeakyRelu"}; // NOTE: You have to provide as many strings here as many layers (n) you defined, since this will set each layer's activations correspongingly

 - Network network(layerSizes, "OptimizationMethod", "RegulariaztionMethod", "InitializationMethod", optionalDesiredPrecision, optionalDropoutRate);	// NOTE: These parameters should be provided to create the network. You can set optimization(Adam, Adamax, etc.), regularization(L1, L2) and initialization(We, Xavier, Rand) methods to "none", by default it will use then no opt., reg. and will set initial weights and biases randomly. Optional variables can be left out.

 - network.trainNetwork("TrainginMethod", inputs, outputs, optionalEchoNumber);	// NOTE: Training method can be: GradientDescent(gd), StochasticGradienDescent(sgd), BatchGradientDescent(bgd), MinibatchGradientDescent(mgd), inputs is the created input vector, outputs is the created output vector, echo number could be set to a higher (~1000) value, so more training cycles will be done.

After training you can check the performance of the network like this:
 - unseenInputs = convertInput(your_path_to_other_inputs);	// NOTE: You can use previously "seen" and "unseen" inputs
 - OPTIONAL: unseedExpecteds = convertExpecteds(yout_path_to_other_expecteds);	// NOTE: Set the expected values this step is OPTIONAL
 - network.testNetwork(unseenInputs, unseenExpecteds);	// This will perform a feedforward process on the network, and provided by the expected array it will calulate the error rate for the new inputs, if no expected will be provided, you will be returned the output activations array for every input, so you will have to decode, if the NN perfromed as you'd like.

<code>
 Brief example for the repo:
 - string Input_file: \path\animalFolder: {dogPic1.jpg, dogPic2.jpg, catPic1.jpg, dogPic3.jpg ....}
 - string Expected_file:\path\animals.txt: {"dog,dog,cat,dog ..."}
 - vector<vector<float>> inputs = convertInput(Input_file);
 - vector<vector<float>> expecteds = convertInput(Expected_file);
 - // Initializations
 - network.train(inputs, outputs);
 </code>
   
</code>
 Currently this project is in this state: WORK IN PROGRESS!
 Neuron, Edge, Layer and Network classes are implemented, but 
 they are not tested and not complete!
 This project is done by 1 person, if anybody has any suggestins, bugs
 and bug fix ideas, please feel free to conact me through the github repo

 It is NOT necessary that your expected and input arrays to be the same numerical type, the training should work easily!
 As discussed above, you will not need to do any type of conversations, you can easily paste your input output source's path to the specific  function, it will handle it for you.

 This is a brief example of how you could use this repository, with artificially created input and expected values.

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
