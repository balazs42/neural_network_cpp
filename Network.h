#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>	// Include for vector container
#include <algorithm>// Include for random shuffle and etc.

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

public:
	Network() : layers(), edges(), 
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

		// Creating the layers of neurons
		for (unsigned i = 0; i < numNeurons.size(); i++)
		{
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
	// Normalizing functions to 0 - 1 intervall
	template<typename T>
	double* normalizeData(T* arr, unsigned num);		

	// Training functions
	template<typename T>
	void feedForwardNetwork(T* inputArr, unsigned num);

	template<typename T>
	void calculateError(T* expectedArray, unsigned num);

	void calculateDeltaActivation();
	void calculateDeltaBias();
	void calculateDeltaWeight();
	void setNewParameters();

	template <typename T1, typename T2>
	void backPropagation(T1* inputArr, unsigned inNum, T2* expectedArr, unsigned expNum);

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

	/***************************************/
	/********* Training functions **********/
	/***************************************/

	// Implementation of stochstic gradient descent method
	template <typename T1, typename T2>
	void stochasticGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

	// Implementation of gradient descent method
	template <typename T1, typename T2>
	void gradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

	// Implementation of the minibatch gradient descent method
	template <typename T1, typename T2>
	void minibatchGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

	// Convert input arrays
	template <typename T1, typename T2>
	void convertInput(vector<vector<T1>> inArr, vector<vector<T2>> expArr, vector<T1*>& inArrConverted, vector<T2*>& expArrConverted);

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
			gradientDescent(convertedInArr, inNum, expArr, convertedExpArr, epochNum);
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