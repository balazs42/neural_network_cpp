#pragma once

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>	// Include for vector container
#include <algorithm>// Include for random shuffle and etc.
#include <random>
#include <iostream>
#include <string>

#include "Layer.h"	// Include for Layer class
#include "Edge.h"	// Include for Edge class

#include <cstdlib>	// Include <cstdlib> for rand() and srand()
#include <ctime>	// Include <ctime> for time()

#include <omp.h>	// Include for OMP paralellization

using std::vector;
using std::string;

class Network
{
private:
	vector<Layer> layers;	// Layers in the network
	vector<Edge**> edges;	// Edges in the network

	// Optimization techniques
	bool useRmspropOptimization;
	bool useAdagradOptimization;
	bool useAdadeltaOptimization;
	bool useNagOptimization;
	bool useAdamaxOptimization;
	bool useAdamOptimization;

	// Random number generation using the modern <random> library
	std::mt19937 gen; // Standard mersenne_twister_engine seeded with time()
	std::uniform_real_distribution<> dis; // Uniform distribution between -1.0 and 1.0

public:
	Network() : layers(), edges(), gen(std::random_device{}()), dis(-1.0, 1.0),
		useRmspropOptimization(false), useAdagradOptimization(false), useAdadeltaOptimization(false),
		useNagOptimization(false), useAdamaxOptimization(false), useAdamOptimization(false) {}

	// Vector of numbers of neurons in each layer, from left to right should be passed, and if adam optimization should be used
	Network(vector<unsigned> numNeurons, const string& opt)
	{
		// Choosing optimization technique
		if (opt == "RMS" || opt == "rms")
		{
			useRmspropOptimization = true;
		}
		else if (opt == "Adagrad" || opt == "adargad")
		{
			useAdagradOptimization = true;
		}
		else if (opt == "Adadelta" || opt == "adadelta")
		{
			useAdadeltaOptimization = true;
		}
		else if (opt == "NAG" || opt == "nag")
		{
			useNagOptimization = true;
		}
		else if (opt == "Adamax" || opt == "adamax")
		{
			useAdamaxOptimization = true;
		}
		else if (opt == "Adam" || opt == "adam")
		{
			useAdamOptimization = true;
		}
		else if (opt == "none" || opt == "NONE" || opt == "None" || opt == "n")
		{
			useRmspropOptimization = false;
			useAdagradOptimization = false;
			useAdadeltaOptimization = false;
			useNagOptimization = false;
			useAdamaxOptimization = false;
			useAdamOptimization = false;
		}
		else
		{
			throw out_of_range("Invalid optimization technique, none will be used, check code if invalid.");
			useRmspropOptimization = false;
			useAdagradOptimization = false;
			useAdadeltaOptimization = false;
			useNagOptimization = false;
			useAdamaxOptimization = false;
			useAdamOptimization = false;
		}


		// Checking to see if at least we have at least 2 layers
		if (numNeurons.size() < 2)
		{
			throw out_of_range("Invalid number of layers, check code!");
			exit(-5);
		}

		// Checking if there is a 0 size layer
		for (unsigned i = 0; i < numNeurons.size(); i++)
		{
			if (numNeurons[i] <= 0)
			{
				throw out_of_range("Cannout init a layer with 0 neurons, check code!");
				exit(-6);
			}
		}

		// Clearing layers vector
		layers.clear();

		// Creating the layers of neurons
		for (unsigned i = 0; i < numNeurons.size(); i++)
		{
			std::cout << numNeurons[i] << "\n";
			layers.push_back(Layer(numNeurons[i]));
		}

		// Creating the edges between layers
		for (unsigned i = 0; i < numNeurons.size() - 1; i++)
		{
			// Getting the number of neurons between the 2 layers where we want to create the edges
			unsigned leftLayerNumNeurons = numNeurons[i];
			unsigned rightLayerNumNeurons = numNeurons[i + 1];

			// Creating first dimension of the egdes
			Edge** tmp = new Edge*[leftLayerNumNeurons];

			// Adding 2nd dimension to edges
			for (unsigned j = 0; j < leftLayerNumNeurons; j++)
				tmp[j] = new Edge[rightLayerNumNeurons];

			// Adding allocated 2D edge array to the network
			edges.push_back(tmp);
		}

		// Randomly initializing each weight and bias
		randInitNetwork();
	}

	// Setting the activation function to be the same for the whole network
	void setAllActivationFunctions(const string& s) { for (unsigned i = 0; i < layers.size(); i++) layers[i].setLayerActivationFunction(s); }

	// Setting the activation functions of the layers, based on the string of vectors, v should look like: v = {"Sigmoid", "Relu", "Tanh", "Swish", "Relu"}
	void setLayerActivationFunctions(const vector<string>& v)
	{
		// Checking if all layers have an activation function given in the string vector
		if (v.size() != layers.size())
		{
			throw out_of_range("Invalid number of strings, check code!");
			exit(-7);
		}

		// Setting activation functions
		for (unsigned i = 0; i < v.size(); i++)
			layers[i].setLayerActivationFunction(v[i]);
	}
public:
	// Getter functions
	vector<Edge**> getEdges() { return edges; }
	vector<Layer> getLayers() { return layers; }
	string getOptimization() 
	{ 
		string opt = "None";
		if (useAdadeltaOptimization)
			opt = "Adadelta";
		else if (useAdagradOptimization)
			opt = "Adagrad";
		else if (useAdamaxOptimization)
			opt = "Adamax";
		else if (useNagOptimization)
			opt = "Nag";
		else if (useAdamOptimization)
			opt = "Adam";
		return opt;
	}

	/***************************************/
	/******* Initializing functions ********/
	/***************************************/
private:
	// Function to generate a random number (you can customize this as needed)
	double getRandomNumber();
	void randInitEdges();
	void randInitNeurons();
public:
	void randInitNetwork() { randInitEdges(); randInitNeurons(); }

	/***************************************/
	/****** File handling functions ********/
	/***************************************/


private:

	/***************************************/
	/********* Training functions **********/
	/***************************************/


private:
	// Normalizing the given data to 1 - 0 range
	template<typename T>
	void normalizeData(T* arr, unsigned num, double* retArr)
	{
		// Initialize min and max with the extreme values of the data type
		T max = std::numeric_limits<T>::lowest();
		T min = std::numeric_limits<T>::max();

		// Find the min and max values in the array
		for (unsigned i = 0; i < num; i++)
		{
			if (arr[i] > max)
				max = arr[i];

			if (arr[i] < min)
				min = arr[i];
		}

		// TODO: handle if all elements are the same
		if (min == max)
		{
			throw out_of_range("Division by zero, check arrays, at noramlizing input or output!");
			exit(-8);
		}
		double dMin = (double)min;
		double dMax = (double)max;

		// Normalize the data
		for (unsigned i = 0; i < num; i++)
			retArr[i] = (double)((arr[i] - dMin) / (dMax - dMin));

		// Debug print:
		std::cout << "Normalized array\n";
		for (unsigned i = 0; i < num; i++)
			std::cout << retArr[i] << " ";
		std::cout << "\n";
	}

	// Feedforward process on network
	template<typename T>
	void feedForwardNetwork(T* inputArr, unsigned num)
	{
		// Check if input array size is correct
		if (num > layers[0].getNumberOfNeurons())
		{
			// Input data will be lost, if input data is hierarchly ordered, and last portion of data is not necessary this could be ignored
			throw out_of_range("Input array is larger then input layer, data will be lost, check code if this is a problem!");
		}

		// Debug print input array
		std::cout << "Input array\n";
		for (unsigned i = 0; i < num; i++)
			std::cout << inputArr[i] << " ";
		std::cout << "\n";

		// 1st step is normalizing input array, to 0-1 values, and setting first layers activations as these values


		// Normalizing data
		double* normalizedInput = new double[num];
		normalizeData(inputArr, num, normalizedInput);

		// Setting first layer's activations
		Neuron* firstLayer = layers[0].getThisLayer();
		unsigned numberOfNeurons = layers[0].getNumberOfNeurons();

		for (unsigned i = 0; i < num; i++)
			firstLayer[i].setActivation(normalizedInput[i]);

		// Debug print in input layer
		for (unsigned i = 0; i < num; i++)
			std::cout << "Input: " << normalizedInput[i] << " Activation: " << firstLayer[i].getActivation() << "\n";

		// Starting feedforward process
		for (int i = 0; i < layers.size() - 1; i++)
		{
			// Iterationg through each neuron in the right layer
			for (int j = 0; j < layers[i + 1].getNumberOfNeurons(); j++)
			{
				// Neuron array on the right side
				Neuron* rightLayer = layers[i + 1].getThisLayer();

				// Neuron array on the left side
				Neuron* leftLayer = layers[i].getThisLayer();

				double z = 0.0f;
				// Iterating through each neuron in left layer
#pragma omp parallel for reduction(+:z)
				for (int k = 0; k < layers[i].getNumberOfNeurons(); k++)
				{
					// calc    =        left activation       *       edge between         
					z += leftLayer[k].getActivation() * edges[i][k][j].getWeight();
				}
				// Adding bias to the activation
				z += rightLayer[j].getBias();

				// Saving z value
				rightLayer[j].setZ(z);

				// Setting current neurons activation as: activation = activationFunction(z); [for example: a = sigmoid(z)]
				z = rightLayer[j].activateNeuron(rightLayer[j].getZ());
				rightLayer[j].setActivation(z);
			}
		}

		// Free allocated memory
		delete[] normalizedInput;
		
		// Debug print
		std::cout << "Activations in the network\n";
		for (unsigned i = 0; i < layers.size(); i++)
		{
			Neuron* thisLayer = layers[i].getThisLayer();
			for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
			{
				std::cout << thisLayer[j].getActivation() << " ";
			}
			std::cout << "\n";
		}
	}

	// Calculating network's error based on the provided expected array
	template<typename T>
	void calculateError(T* expectedArray, unsigned num)
	{
		// Checking if expected array is same size as output layer
		if (num != layers[layers.size() - 1].getNumberOfNeurons())
		{
			throw out_of_range("Expected array not same size as output layer, check code!");
			exit(-9);
		}

		// First error calculation in the last layer based on the given expected array

		// Normalizing expected array
		double* normalizedExpected = new double[num];
		normalizeData(expectedArray, num, normalizedExpected);

		Neuron* lastLayer = layers[layers.size() - 1].getThisLayer();

		// Parallelize error calculation for each neuron in the last layer
#pragma omp parallel for
		for (int i = 0; i < num; i++)
		{
			// Cost function is: E = (a^L - y)^2, d/dE = 2 * (a^L - y) , where y is the expected value at a given index, and a^L is the L-th layer's activation at the given index
			// So the error should be calculated like this: 
			//           Error =  2 *       activation             -  ||expected value||
			lastLayer[i].setError(2 * lastLayer[i].getActivation() - normalizedExpected[i]);
		}

		delete[] normalizedExpected;
	}

	void calculateDeltaActivation();
	void calculateDeltaBias();
	void calculateDeltaWeight();
	void setNewParameters();

	// Normalizing functions
	template<typename T>
	double* normalizeInput(T* arr, unsigned num) { return normalizeData(arr, num); }

	template<typename T>
	double* normalizeExpected(T* arr, unsigned num) { return normalizeData(arr, num); }

	/***************************************/
	/******* Optimization functions ********/
	/***************************************/

	void rmspropOptimization(double learningRate = 0.001f, double decayRate = 0.9f, double epsilon = 1e-8f);
	void adagradOptimization(double learningRate = 0.01f, double epsilon = 1e-8f);
	void adadeltaOptimization(double decayRate = 0.9f, double epsilon = 1e-8f);
	void nagOptimization(double learningRate = 0.001f, double momentum = 0.9f);
	void adamaxOptimization(double learningRate = 0.002f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 1e-8f);
	void adamOptimization(double learningRate = 0.001f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 1e-8f);

	// Implementation of 1 backpropagation process
	template <typename T1, typename T2>
	void backPropagation(T1* inputArr, unsigned inNum, T2* expectedArr, unsigned expNum)
	{
		// Feedforward process
		feedForwardNetwork(inputArr, inNum);

		// First error calculation in last layer
		calculateError(expectedArr, expNum);

		// Calculating delta activations
		calculateDeltaActivation();

		// Parallelize delta bias calculation
#pragma omp parallel
		{
			calculateDeltaBias();
		}

		// Parallelize delta weight calculation
#pragma omp parallel
		{
			calculateDeltaWeight();
		}

		// Setting new parameters to the network
		setNewParameters();

		// Using the given optimization techniwue
		if (useRmspropOptimization)
			rmspropOptimization();
		else if (useAdagradOptimization)
			adagradOptimization();
		else if (useAdadeltaOptimization)
			adadeltaOptimization();
		else if (useNagOptimization)
			nagOptimization();
		else if (useAdamaxOptimization)
			adamaxOptimization();
		else if (useAdamOptimization)
			adamOptimization();
	}

	// Stochastic gradient descent function
	template <typename T1, typename T2>
	void stochasticGradientDescent(vector<T1*>& inArr, vector<unsigned>& inNum, vector<T2*>& expArr, vector<unsigned>& expNum, unsigned epochNum)
	{
		// Perform stochastic gradient descent for each epoch
		for (unsigned epoch = 0; epoch < epochNum; ++epoch)
		{
			// Iterate over each training example
//#pragma omp parallel for
			for (unsigned i = 0; i < inArr.size(); ++i)
			{
				// Perform backpropagation and update parameters for each training example
				backPropagation(inArr[i], inNum[i], expArr[i], expNum[i]);
			}
		}
	}

	// Gradient descent function
	template <typename T1, typename T2>
	void gradientDescent(vector<T1*>& inArr, vector<unsigned>& inNum, vector<T2*>& expArr, vector<unsigned>& expNum, unsigned epochNum)
	{
		// Perform gradient descent for each epoch
		for (unsigned epoch = 0; epoch < epochNum; ++epoch)
		{
			for (unsigned i = 0; i < inArr.size(); i++)
			{
				// Perform backpropagation and update parameters for the entire dataset
				backPropagation(inArr[i], inNum[i], expArr[i], expNum[i]);
			}
		}
	}

	// Minibatch gradient descent function
	template <typename T1, typename T2>
	void minibatchGradientDescent(vector<T1*>& inArr, vector<unsigned>& inNum, vector<T2*>& expArr, vector<unsigned>& expNum, unsigned epochNum)
	{
		// Define the minibatch size (e.g., 32)
		const unsigned minibatchSize = 32;
		const unsigned numExamples = inArr.size();

		// Perform minibatch gradient descent for each epoch
		for (int epoch = 0; epoch < epochNum; ++epoch)
		{
			// Iterate over the training examples in minibatches
#pragma omp parallel for
			for (int startIdx = 0; startIdx < numExamples; startIdx += minibatchSize)
			{
				int endIdx = std::min(startIdx + minibatchSize, numExamples);

				// Extract the minibatch
				vector<T1*> minibatchInArr(inArr.begin() + startIdx, inArr.begin() + endIdx);
				vector<unsigned> minibatchInNum(inNum.begin() + startIdx, inNum.begin() + endIdx);
				vector<T2*> minibatchExpArr(expArr.begin() + startIdx, expArr.begin() + endIdx);
				vector<unsigned> minibatchExpNum(expNum.begin() + startIdx, expNum.begin() + endIdx);

				// Perform backpropagation and update parameters for the minibatch
				backPropagation(minibatchInArr[startIdx], minibatchInNum[startIdx], minibatchExpArr[startIdx], minibatchExpNum[startIdx]);
			}
		}
	}

	// Converting input arrays
	template <typename T1, typename T2>
	void convertInput(std::vector<std::vector<T1>>& inArr, std::vector<std::vector<T2>>& expArr, std::vector<T1*>& inArrConverted, std::vector<T2*>& expArrConverted)
	{
		// Converting input array
		for (unsigned i = 0; i < inArr.size(); i++)
		{
			// Allocating new memory
			inArrConverted.push_back(new T1[inArr[i].size()]);

			// Copying data
			for (unsigned j = 0; j < inArr[i].size(); j++)
				inArrConverted[i][j] = inArr[i][j];
		}

		// Converting expected array
		for (unsigned i = 0; i < expArr.size(); i++)
		{
			expArrConverted.push_back(new T2[expArr[i].size()]);

			// Copying data
			for (unsigned j = 0; j < expArr[i].size(); j++)
				expArrConverted[i][j] = expArr[i][j];
		}
	}

public:
	
	// Save network to file
	static void saveNetworkToXML(const string& fileName, Network& network);

	// Load network to previously created network
	static void loadNetworkFromXML(const string& route);

	/**
	 * Trains the neural network using the specified training method and parameters.
	 *
	 * @param s Method selector string indicating the training algorithm to be used.
	 * @param inArr Vector of input arrays. Each array must be of numerical type.
	 * @param expArr Vector of expected value arrays. Each array must be of numerical type.
	 * @param epochNum Number of epochs for training.
	 * @tparam T1 Type of the input array elements.
	 * @tparam T2 Type of the expected value array elements.
	 */
	template <typename T1, typename T2>
	void trainNetwork(const std::string& s, std::vector<vector<T1>> inArr, std::vector<vector<T2>> expArr, unsigned epochNum)
	{
		// Creatin in sizes array
		vector<unsigned> inNum;

		// Filling with sizes
		for (unsigned i = 0; i < inArr.size(); i++)
			inNum.push_back(inArr[i].size());

		// Creating expected sizes array
		vector<unsigned> expNum;

		// Filling expected array
		for (unsigned i = 0; i < expArr.size(); i++)
			expNum.push_back(expArr[i].size());

		// Containers
		vector<T1*> convertedInArr;
		vector<T2*> convertedExpArr;

		// Converting vectors
		convertInput(inArr, expArr, convertedInArr, convertedExpArr);

		if (s == "Minibatch" || s == "minibatch" || s == "mb")
		{
			// Training using the minibatch gradient descent method
			minibatchGradientDescent(convertedInArr, inNum, convertedExpArr, expNum, epochNum);
		}
		else if (s == "StochasticGradientDescent" || s == "SGD" || s == "sgd")
		{
			// Traingin using the stochastic gradient descent method
			stochasticGradientDescent(convertedInArr, inNum, convertedExpArr, expNum, epochNum);
		}
		else if (s == "GradientDescent" || s == "GD" || s == "gd")
		{
			// Trainging using the gradient descent methdo
			gradientDescent(convertedInArr, inNum, convertedExpArr, expNum, epochNum);
		}
		else
		{
			throw out_of_range("Invalid traingin technique, check code!");
			exit(-10);
		}

		// Free allocated memory
		for (unsigned i = 0; i < convertedInArr.size(); i++)
			delete[] convertedInArr[i];
		for (unsigned i = 0; i < convertedExpArr.size(); i++)
			delete[] convertedExpArr[i];
	}
};


#endif /*_NETWORK_H_*/