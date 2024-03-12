#pragma once

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>	// Include for vector container
#include <algorithm>// Include for random shuffle and etc.
#include <random>
#include <iostream>
#include <string>
#include <iomanip> // Include for std::setprecision and std::fixed

#include "Layer.h"	// Include for Layer class
#include "Edge.h"	// Include for Edge class

#include <cstdlib>	// Include <cstdlib> for rand() and srand()
#include <ctime>	// Include <ctime> for time()

#include <omp.h>	// Include for OMP paralellization

using std::vector;
using std::string;

const double beta1PowerStarter = 0.9f;		// Starter values to beta1power and beta2power's for adam optimization
const double beta2PowerStarter = 0.999f;

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

	// Regularization technikques
	bool useL1;
	bool useL2;
	bool useDropout;

	// Initializer technique
	bool useWe;
	bool useXavier;

	double dropoutRate; // Probability of dropping out a neuron in applicable layers ranging: [0.2 0.5]

	double beta1Power;	// Decay rate for adam optimization
	double beta2Power;

	double desiredPrecision;	// Precision that user wants to achieve
	double currentPrecision;	// Currently obtained precision by the network

	// Random number generation using the modern <random> library
	std::mt19937 gen; // Standard mersenne_twister_engine seeded with time()
	std::uniform_real_distribution<> dis; // Uniform distribution between -1.0 and 1.0

public:
	Network() : layers(), edges(), gen(std::random_device{}()), dis(-1.0, 1.0),
		useRmspropOptimization(false), useAdagradOptimization(false), useAdadeltaOptimization(false),
		useNagOptimization(false), useAdamaxOptimization(false), useAdamOptimization(false),
		useL1(false), useL2(false), useDropout(false), dropoutRate(0.5f), useWe(false), useXavier(false), 
		desiredPrecision(0.002f), currentPrecision(1.0f)
	{
		beta1Power = beta1PowerStarter;
		beta2Power = beta2PowerStarter;
	}

	/**
	 * Constructs a neural network with specified configurations for optimization, regularization, initialization techniques, and dropout rate.
	 *
	 * @param numNeurons A vector containing the number of neurons in each layer, ordered from the input to the output layer.
	 * @param opt A string specifying the optimization technique to be used. Options include "RMS", "Adagrad", "Adadelta", "NAG", "Adamax", "Adam", or "none" for no optimization.
	 * @param reg A string indicating the regularization technique to be applied. Options are "L1", "L2", "Dropout", or "None" for no regularization.
	 * @param initer A string denoting the weight initialization method. Options include "We" for He initialization, "Xavier", or default initialization if unspecified.
	 * @param dpr The dropout rate to be applied if dropout regularization is chosen. Represents the fraction of neurons to drop in each forward pass during training.
	 *
	 * This constructor initializes the neural network's structure, setting up layers and connections based on the provided specifications. It validates the input parameters
	 * for layer configurations, ensures at least two layers are defined, and initializes weights and biases according to the chosen methods. Throws exceptions for invalid configurations.
	 */
	Network(vector<unsigned> numNeurons, const string& opt, const string& reg, const string& initer, double desiredPrec = 0.002f, double dpr = 0.5f) : gen(std::random_device{}()), dis(-1.0, 1.0),
		useRmspropOptimization(false), useAdagradOptimization(false), useAdadeltaOptimization(false),
		useNagOptimization(false), useAdamaxOptimization(false), useAdamOptimization(false),
		useL1(false), useL2(false), useDropout(false), dropoutRate(dpr), useWe(false), useXavier(false), 
		desiredPrecision(desiredPrec), currentPrecision(1.0f)
	{
		beta1Power = beta1PowerStarter;
		beta2Power = beta2PowerStarter;

		// Choosing optimization technique
		if (opt == "RMS" || opt == "rms")
			useRmspropOptimization = true;
		else if (opt == "Adagrad" || opt == "adargad")
			useAdagradOptimization = true;
		else if (opt == "Adadelta" || opt == "adadelta")
			useAdadeltaOptimization = true;
		else if (opt == "NAG" || opt == "nag")
			useNagOptimization = true;
		else if (opt == "Adamax" || opt == "adamax")
			useAdamaxOptimization = true;
		else if (opt == "Adam" || opt == "adam")
			useAdamOptimization = true;
		else if (opt == "none" || opt == "NONE" || opt == "None" || opt == "n")
		{
		}
		else
			throw out_of_range("Invalid argument at optimization technique, none will be used, check code!");

		// Choosing regularization technique
		if (reg == "L1" || reg == "l1")
			useL1 = true;
		else if (reg == "L2" || reg == "l2")
			useL2 = true;
		else if (reg == "Dropout" || reg == "dropout" || reg == "d")
			useDropout = true;
		else if (reg == "None" || reg == "none" || reg == "N" || reg == "n")
		{
		}
		else
			throw out_of_range("Invalid argument at regularization technique, none will be used, check code!");

		// Choosing initialization technique
		if (initer == "We" || initer == "we")
			useWe = true;
		else if (initer == "Xavier" || initer == "xavier" || initer == "x")
			useXavier = true;

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

		// By default initializing with random weights and biases
		// Initializing network
		randInitNetwork();

		// If initialization technique is selected, then applying it
		if (useWe)
			initializeWeightsHe();
		else if (useXavier)
			initializeWeightsXavier();
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

	// Setter for dropout rate
	void setDropoutRate(double parameter) { dropoutRate = parameter; }

	// Set beta 1 power
	void setBeta1Power(double parameter) { beta1Power = parameter; }

	// Set beta 2 power
	void setBeta2Power(double parameter) { beta2Power = parameter; }

	// Setting current precision variable
	void setCurrentPrecision(double parameter) { currentPrecision = parameter; }

	// Setting desired precision
	void setDesiredPrecision(double parameter) { desiredPrecision = parameter; }

	// Getter functions
	vector<Edge**> getEdges() const { return edges; }
	vector<Layer> getLayers() const { return layers; }
	string getOptimization() const 
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
	string getRegularization() const 
	{
		string reg = "None";
		if (useL1)
			reg = "L1";
		else if (useL2)
			reg = "L2";
		else if (useDropout)
			reg = "Dropout";
		return reg;
	}
	double getDropoutRate() const { return dropoutRate; }

	// Getter for beta 1 and beta 2
	double getBeta1Power() const { return beta1Power; }
	double getBeta2Power() const { return beta2Power; }

	// Getter function for precision variables
	double getCurrentPrecision() const { return currentPrecision; }
	double getDesiredPrecision() const { return desiredPrecision; }

private:

	/***************************************/
	/******* Initializing functions ********/
	/***************************************/

	// Function to generate a random number (you can customize this as needed)
	double getRandomNumber();
	void randInitEdges();
	void randInitNeurons();

	// He weight initializing algorithm
	void initializeWeightsHe();

	// Xavier weight initializing algorithm
	void initializeWeightsXavier();
public:
	// Randomly init each neuron and weight in the network
	void randInitNetwork() { randInitEdges(); randInitNeurons(); }

	/***************************************/
	/****** File handling functions ********/
	/***************************************/
	void printNetwork();

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

		// Handle if all elements are the same
		if (min == max)
		{
			for (unsigned i = 0; i < num; i++)
				retArr[i] = (double)(arr[i]);
			return;
		}

		double dMin = (double)min;
		double dMax = (double)max;

		// Normalize the data
		for (unsigned i = 0; i < num; i++)
			retArr[i] = (double)((arr[i] - dMin) / (dMax - dMin));
	}

	// Normalizing functions, TESTED: OK
	template<typename T>
	double* normalizeInput(T* arr, unsigned num) { return normalizeData(arr, num); }

	// Normalize expected array, TESTED: OK
	template<typename T>
	double* normalizeExpected(T* arr, unsigned num) { return normalizeData(arr, num); }

	// Feedforward process on network, TESTED: OK
	// Each activation is calculated in this form:
	// a^(L)j = ActFun((SUM(a^(L-1)k*Wjk) + B^(L)j)
	template<typename T>
	void feedForwardNetwork(T* inputArr, unsigned num)
	{
		// Check if input array size is correct
		if (num > layers[0].getNumberOfNeurons())
		{
			// Input data will be lost, if input data is hierarchly ordered, and last portion of data is not necessary this could be ignored
			throw out_of_range("Input array is larger then input layer, data will be lost, check code if this is a problem!");
		}

		// 1st step is normalizing input array, to 0-1 values, and setting first layers activations as these values

		// Normalizing data
		double* normalizedInput = new double[num];
		normalizeData(inputArr, num, normalizedInput);

		Neuron* firstLayer = layers[0].getThisLayer();
		unsigned numberOfNeurons = layers[0].getNumberOfNeurons();

		// Setting first layer's activations
		for (unsigned i = 0; i < num; i++)
			firstLayer[i].setActivation(normalizedInput[i]);

		// Starting feedforward process
		for (int i = 0; i < layers.size() - 1; i++)
		{
			// Neuron array on the right side
			Neuron* rightLayer = layers[i + 1].getThisLayer();
			unsigned sizeRight = layers[i + 1].getNumberOfNeurons();

			// Neuron array on the left side
			Neuron* leftLayer = layers[i].getThisLayer();
			unsigned sizeLeft = layers[i].getNumberOfNeurons();
	
			// Iterationg through each neuron in the right layer
			for (int j = 0; j < sizeRight; j++)
			{

				double z = 0.0f;
				// Iterating through each neuron in left layer
#pragma omp parallel for reduction(+:z)
				for (int k = 0; k < sizeLeft; k++)
				{
					// calc    =        left activation       *       edge between         
					z += leftLayer[k].getActivation() * edges[i][k][j].getWeight();
				}
				// Adding bias to the activation
				z += rightLayer[j].getBias();

				// Saving z value
				rightLayer[j].setZ(z);

				// Setting current neurons activation as: activation = activationFunction(z); [for example: a = sigmoid(z)]
				double activation = rightLayer[j].activateNeuron(rightLayer[j].getZ());
				rightLayer[j].setActivation(activation);
			}
		}

		// Free allocated memory
		delete[] normalizedInput;
	}

	// TODO: Add more cost functions besides quadratic const function

	// Calculating network's error based on the provided expected array, TESTED: OK
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
			lastLayer[i].setError(2.0f * lastLayer[i].getActivation() - normalizedExpected[i]);
		}

		// Free allocated memory
		delete[] normalizedExpected;
	}

	// Calculated delta activations in the network, TESTED: OK
	void calculateDeltaActivation();

	// Calculate delta biases in the network, TESTED: OK
	void calculateDeltaBias();

	// Calcualte delta weights in the network, TESTED: OK
	void calculateDeltaWeight();

	// Setting newly calulated parameters to the network, TESTED: OK
	void setNewParameters();

	/***************************************/
	/******* Optimization functions ********/
	/***************************************/

	void rmspropOptimization(double learningRate = 0.001f, double decayRate = 0.9f, double epsilon = 1e-8f);
	void adagradOptimization(double learningRate = 0.01f, double epsilon = 1e-8f);
	void adadeltaOptimization(double decayRate = 0.9f, double epsilon = 1e-8f);
	void nagOptimization(double learningRate = 0.001f, double momentum = 0.9f);
	void adamaxOptimization(double learningRate = 0.0005f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 1e-8f);
	void adamOptimization(double learningRate = 0.0005f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 1e-8f);

	// If there is a selected optimization method, then using it
	// @return true if optimization is applied false if not
	bool useOptimization() 
	{
		// Using the given optimization techniwue
		if (useRmspropOptimization)
		{
			rmspropOptimization();
			return true;
		}
		else if (useAdagradOptimization)
		{
			adagradOptimization();
			return true;
		}
		else if (useAdadeltaOptimization)
		{
			adadeltaOptimization();
			return true;
		}
		else if (useNagOptimization)
		{
			nagOptimization();
			return true;
		}
		else if (useAdamaxOptimization)
		{
			adamaxOptimization();
			return true;
		}
		else if (useAdamOptimization)
		{
			adamOptimization();
			return true;
		}

		return false;
	}

	/***************************************/
	/****** Regularization functions *******/
	/***************************************/
	void updateWeightsL1(double learningRate = 0.001f, double lambda = 0.001f);
	void updateWeightsL2(double learningRate = 0.001f, double lambda = 0.001f);

	void updateNeuronsL1(double learningRate = 0.001f, double lambda = 0.001f);
	void updateNeuronsL2(double learningRate = 0.001f, double lambda = 0.001f);

	void applyL1Regularization(double learningRate = 0.001f, double lambda = 0.001f) { updateWeightsL1(learningRate, lambda); updateNeuronsL1(learningRate, lambda); }
	void applyL2Regularization(double learningRate = 0.001f, double lambda = 0.001f) { updateWeightsL2(learningRate, lambda); updateNeuronsL2(learningRate, lambda); }

	void applyDropoutRegularization();

	// Applying regularization technique, if selected
	// @return true if regularization is used false if not
	bool useRegularization()
	{
		if (useL1)
		{
			applyL1Regularization();
			return true;
		}
		else if (useL2)
		{
			applyL2Regularization();
			return true;
		}
		return false;
	}
private:

	// Implementation of 1 backpropagation process
	// The implementation is highly based on 3Blue1Brown: Backpropagation video and article
	// And also on this arcticle: http://neuralnetworksanddeeplearning.com/chap2.html
	template <typename T1, typename T2>
	void backPropagation(T1* inputArr, unsigned inNum, T2* expectedArr, unsigned expNum)
	{
		// Feedforward process
		feedForwardNetwork(inputArr, inNum);

		// Applying dropout regularization if selected
		if (useDropout)
			applyDropoutRegularization();

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
		// Applying optimization and regularization
		// technique is inside of this function
		setNewParameters();
	}

	// Stochastic gradient descent function
	template <typename T1, typename T2>
	void stochasticGradientDescent(vector<T1*>& inArr, vector<unsigned>& inNum, vector<T2*>& expArr, vector<unsigned>& expNum, unsigned epochNum)
	{
		// Perform stochastic gradient descent for each epoch
		for (unsigned epoch = 0; epoch < epochNum; ++epoch)
		{
			// Resetting beta1 and beta2 power variables for each epoch
			beta1Power = beta1PowerStarter;
			beta2Power = beta2PowerStarter;

			// Iterate over each training example
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
			// Resetting beta1 and beta2 power variables for each epoch
			beta1Power = beta1PowerStarter;
			beta2Power = beta2PowerStarter;

			for (unsigned i = 0; i < inArr.size(); i++)
			{
				// Perform backpropagation and update parameters for the entire dataset
				backPropagation(inArr[i], inNum[i], expArr[i], expNum[i]);
			}
		}
	}

	// Batch gradient descent function
	template <typename T1, typename T2>
	void batchGradientDescent(vector<T1*>& inArr, vector<unsigned>& inNum, vector<T2*>& expArr, vector<unsigned>& expNum, unsigned epochNum)
	{
		const unsigned batchSize = inArr.size(); // In batch gradient descent, the batch size is the whole dataset

		// Perform batch gradient descent for each epoch
		for (unsigned epoch = 0; epoch < epochNum; epoch++)
		{
			// Resetting beta1 and beta2 power variables for each epoch
			beta1Power = beta1PowerStarter;
			beta2Power = beta2PowerStarter;

			// Since we're using the entire dataset as a batch, there's no need to loop through subsets of the data.
			// However, computation within each backpropagation step can be parallelized.

			// Reset gradients before each epoch
			// (Assuming there's a method to reset gradients, if your implementation accumulates gradients across batches)
			// resetGradients();

			// Parallelize the backpropagation for each input in the batch
			// Note: Depending on how your backpropagation is implemented, you might need to accumulate gradients
			// from each data point before applying them, rather than updating the weights directly.
			for (int i = 0; i < inArr.size(); ++i)
			{
				backPropagation(inArr[i], inNum[i], expArr[i], expNum[i]);
			}

			// Apply accumulated gradients to update network parameters
			// This should be done after computing the gradients for the entire batch
			// updateParameters();
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
			// Resetting beta1 and beta2 power variables for each epoch
			beta1Power = beta1PowerStarter;
			beta2Power = beta2PowerStarter;

			// Iterate over the training examples in minibatches
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

		// Training until the precision gets to the desired one
		while (currentPrecision > desiredPrecision)
		{
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
				// Trainging using the gradient descent method
				gradientDescent(convertedInArr, inNum, convertedExpArr, expNum, epochNum);
			}
			else if (s == "BatchGradientDescent" || s == "BGD" || s == "bgd")
			{
				// Training using the batch gradient descent method
				batchGradientDescent(convertedInArr, inNum, convertedExpArr, expNum, epochNum);
			}
			else
			{
				throw out_of_range("Invalid traingin technique, check code!");
				exit(-10);
			}

			// Calculating precision of network
			calculatePrecision(convertedInArr, inArr[0].size(), convertedExpArr, expArr[0].size());
			std::cout << "\n\nCurrent precision: " << this->getCurrentPrecision() << "\n\n";
		}

		// Free allocated memory
		for (unsigned i = 0; i < convertedInArr.size(); i++)
			delete[] convertedInArr[i];
		for (unsigned i = 0; i < convertedExpArr.size(); i++)
			delete[] convertedExpArr[i];
	}
	/***************************************/
	/********* Testing functions ***********/
	/***************************************/
private:
	// Calculates the precision
	template <typename T1, typename T2>
	double calculateFramePrecision(T1* inFrame, unsigned sizeIn, T2* expOut, unsigned sizeOut)
	{
		double error = 0.0f;

		// Feedforward the input
		feedForwardNetwork(inFrame, sizeIn);

		// Calculate error for the frame
		Neuron* lastLayer = layers[layers.size() - 1].getThisLayer();
		unsigned layerSize = layers[layers.size() - 1].getNumberOfNeurons();

//#pragma omp parallel for reduction(+:error)
		for (int i = 0; i < layerSize; i++)
		{
			double squaredAct = pow(lastLayer[i].getActivation(), 2);
			double squaredExp = pow((double)expOut[i], 2);
			double pr = abs(squaredAct - squaredExp);
			error += sqrt(pr);
		}

		// Divide error for averaging
		error /= layerSize;

		return error;
	}

public:
	template <typename T1, typename T2>
	void calculatePrecision(vector<T1*> inFrame, unsigned sizeIn, vector<T2*> expOut, unsigned sizeOut)
	{
		double prec = 0.0f;

		unsigned numFrames = inFrame.size();

		// Calculating precision for all frames
#pragma omp parallel for reduction(+:prec)
		for (int i = 0; i < numFrames; i++)
			prec += calculateFramePrecision(inFrame[i], sizeIn, expOut[i], sizeOut);

		// Divide precision for averaging
		prec /= inFrame.size();

		// Printing last iteration
		std::cout << "\n";
		for (unsigned i = 0; i < sizeOut; i++)
			std::cout << " |E" << i << ": " << std::fixed << std::setprecision(5) << (double)expOut[expOut.size() - 1][i];

		std::cout << "\n";
		for (unsigned i = 0; i < sizeOut; i++)
			std::cout << " |A" << i << ": " << std::fixed << std::setprecision(5) << layers[layers.size() - 1].getThisLayer()[i].getActivation();

		this->setCurrentPrecision(prec);
	}

	/* Tester functions for the network, performs feedforward
	 * operaition(s) on the network with the given input,
	 * prints the finel layer's activations on the standard
	 * output, and returns the squared mean error.
	 * 
	 * @param inFrame Frame that the network used or not through training
	 * @param sizeIn Size of the frame
	 * @param expOut Expected output array for error calculation
	 * @param sizeOut Size of on expected array
	 * @tparam T Type of the input array elements
	 * @return squared mean error of the network
	 */
	template <typename T1, typename T2>
	double testNetwork(T1* inFrame, unsigned sizeIn, T2* expOut, unsigned sizeOut)
	{
		double error = 0.0f;

		// Feedforward the input
		feedForwardNetwork(inFrame, sizeIn);

		std::cout << "Activations : ";
		for (unsigned i = 0; i < sizeOut; i++) {
			std::cout << "A" << i << ":"
				<< std::fixed << std::setprecision(4)
				<< layers[layers.size() - 1].getThisLayer()[i].getActivation() << " ";
		}
		std::cout << "End of activations\n";

		// Calculating error
#pragma omp parallel for reduction(+:error)
		for (int i = 0; i < layers[layers.size() - 1].getNumberOfNeurons(); i++)
			error += sqrt(pow(layers[layers.size() - 1].getThisLayer()[i].getActivation(), 2) - pow((double)expOut[i], 2));

		std::cout << "\nFrame Error: " << error << "\n";

		// Printing maxes
		std::cout << "\n";
		
		double minDiff = 99999999.0f;
		unsigned minIdx = 0;
		double diff = 0.0f;

		for (int i = 0; i < layers[layers.size() - 1].getNumberOfNeurons(); i++)
		{
			diff = layers[layers.size() - 1].getThisLayer()[i].getActivation() - (double)expOut[i];
			if (diff < minDiff)
			{
				minDiff = diff;
				minIdx = i;
			}
		}

		double minAct = layers[layers.size() - 1].getThisLayer()[minIdx].getActivation();
		double minExp = (double)expOut[minIdx];

		std::cout << "\nMin idx: " << minIdx << ", Difference: D = act - exp = " << minAct <<  " - " << minExp << " = " << minDiff << "\n";
		return error;
	}

	// Test network for a given batch of frames
	template <typename T1, typename T2>
	double testNetwork(vector<T1*> inFrame, unsigned sizeIn, vector<T2*> expOut, unsigned sizeOut)
	{
		double error = 0.0f;

		// Calculating error for each frame
		for (int i = 0; i < inFrame.size(); i++)
			error += testNetwork(inFrame[i], sizeIn, expOut[i], sizeOut);

		std::cout << "\nBatch error: " << error << "\n";

		return error;
	}
};

#endif /*_NETWORK_H_*/