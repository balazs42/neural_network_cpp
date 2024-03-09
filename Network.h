#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>
#include "Layer.h"
#include "Edge.h"

#include <cstdlib> // Include <cstdlib> for rand() and srand()
#include <ctime>   // Include <ctime> for time()

using std::vector;

class Network
{
private:
	vector<Layer> layers;	// Layers in the network
	vector<Edge**> edges;	// Edges in the network
	bool useAdam;

public:
	Network() : layers(), edges(), useAdam(false) {}

	// Vector of numbers of neurons in each layer, from left to right should be passed, and if adam optimization should be used
	Network(vector<unsigned> numNeurons, bool adam)
	{
		useAdam = adam;

		// Checking to see if at least we have at least 2 layers
		if (numNeurons.size() < 2)
		{
			throw out_of_range("Invalid number of layers, check code!");
			exit(-5);
		}

		// Checking if there is a 0 size layer
		for (int i = 0; i < numNeurons.size(); i++)
		{
			if (numNeurons[i] <= 0)
			{
				throw out_of_range("Cannout init a layer with 0 neurons, check code!");
				exit(-6);
			}
		}

		// Creating the layers of neurons
		for (int i = 0; i < numNeurons.size(); i++)
		{
			layers.push_back(Layer(numNeurons[i]));
		}

		// Creating the edges between layers
		for (int i = 0; i < numNeurons.size() - 1; i++)
		{
			// Getting the number of neurons between the 2 layers where we want to create the edges
			unsigned leftLayerNumNeurons = numNeurons[i];
			unsigned rightLayerNumNeurons = numNeurons[i + 1];

			// Creating first dimension of the egdes
			Edge** tmp = new Edge*[leftLayerNumNeurons];

			// Adding 2nd dimension to edges
			for (int j = 0; j < leftLayerNumNeurons; j++)
				tmp[j] = new Edge[rightLayerNumNeurons];

			// Adding allocated 2D edge array to the network
			edges.push_back(tmp);
		}

		// Randomly initializing each weight and bias
		randInitNetwork();
	}

	// Setting the activation function to be the same for the whole network
	void setAllActivationFunctions(const string& s) { for (int i = 0; i < layers.size(); i++) layers[i].setLayerActivationFunction(s); }

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
		for (int i = 0; i < v.size(); i++)
			layers[i].setLayerActivationFunction(v[i]);
	}

	// Getter functions
	vector<Edge**> getEdges() { return edges; }

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

public:
	// Normalizing functions
	template<typename T>
	double* normalizeInput(T* arr, unsigned num) { return normalizeData(arr, num); }

	template<typename T>
	double* normalizeExpected(T* arr, unsigned num) { return normalizeData(arr, num); }

private:
	// Training functions
	void adamOptimization();

	template <typename T1, typename T2>
	void stochasticGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

	template <typename T1, typename T2>
	void gradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

	template <typename T1, typename T2>
	void minibatchGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);

public:
	template <typename T1, typename T2>
	void trainNetwork(const string& s, vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum);
};

#endif /*_NETWORK_H_*/