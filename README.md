Provided by an MSC EE student of Budapest University of Technology and Economics

The implementation is based on 3Blue1Brown's article on the topic: https://www.3blue1brown.com/lessons/backpropagation-calculus
My take on the backpropagation algorithm, classes used for the project are highly based on the article.

 Currently this project is in this state: WORK IN PROGRESS!
 Neuron, Edge, Layer and Network classes are implemented, but 
 they are not tested and not complete!
 This project is done by 1 person, if anybody has any suggestins, bugs
 and bug fix ideas, please feel free to conact me through the github repo

 The main concept is that the user only has to do 2 things:
 -Bring their input to some kind of numerical array
 -Bring their expected values to some kind of numerical array
 It is NOT necessary that your expected and input arrays to be 
 the same numerical type!
 Then train the network. Hopefully it will learn the pattern provided by your data
 The training process should only be done once, after that you can use the 
 loadFromFile() member function, then you can use your network.

 This is an example of how you could use this repository
int main(void)
{
	try
	{
		// Input and output array's sizes should sta constant
		const unsigned inSize = 4; 
		const unsigned expectedSize = 10;

		// First define the number of neurons in each layer
		// each layer should have a numerical value greater then 0
		vector<unsigned> layerSizes = { inSize, 50, 50, expectedSize };

		// Create network object, with the specified 
		Network network(layerSizes, false);

		// You can randinit all edges and biases, in the network, although it is not
		// necessary, the constructor already does it, but if for some reason you want
		// to reinit the network you have the possibility
		network.randInitNetwork();

		//-----------------------------------//

		// This is an artifically created input array, you have to bring your input 
		// to some numerical type array, then pass it to the network.

		// Creating artificial input frames
		unsigned* firstFrame = new unsigned[inSize] {1, 2, 3, 4};
		unsigned* secondFrame = new unsigned[inSize] {2, 3, 4, 5};
		unsigned* thirdFrame = new unsigned[inSize] {10, 9, 8, 11};
		unsigned* fourthFrame = new unsigned[inSize] {1, 0, 12, 3};

		// Creating input vector of frames
		vector<unsigned*> inputArr;

		// Adding frames to input array
		inputArr.push_back(firstFrame);
		inputArr.push_back(secondFrame);
		inputArr.push_back(thirdFrame);
		inputArr.push_back(fourthFrame);

		//-----------------------------------//

		// Creating artificial expected array
		unsigned* firstExpected = new unsigned[expectedSize]  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		unsigned* secondExpected = new unsigned[expectedSize] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
		unsigned* thirdExpected = new unsigned[expectedSize]  {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
		unsigned* fourthExpected = new unsigned[expectedSize] {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};

		vector<unsigned*> expArr;

		expArr.push_back(firstExpected);
		expArr.push_back(secondExpected);
		expArr.push_back(thirdExpected);
		expArr.push_back(fourthExpected);

		//-----------------------------------//

		// Training network, with the provided data
		network.trainNetwork("GradientDescent",												// Training method type
							 inputArr,														// Your created input vector of arrays
							 { inSize, inSize, inSize, inSize },							// Number of elements in each input array
							 expArr,														// Your expected vector of arrays
							 { expectedSize, expectedSize, expectedSize, expectedSize },	// Number of elements in the expected vector of arrays
							 10000															// Number of epochs
		);
}
