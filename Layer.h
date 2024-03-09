#ifndef _LAYER_H_
#define _LAYER_H_

#include "Neuron.h"

class Layer
{
private:
	Neuron* neurons;		// Neuron array in this layer	
	unsigned numNeurons;	// Number of neurons in this layer

public:
	// Default constructor
	Layer() : neurons(nullptr), numNeurons(0) {}

	// Previously allocated neurons can be passed by creating a neuron array by default, or loading it from file
	Layer(Neuron* pNeurons, unsigned pNum) : neurons(pNeurons), numNeurons(pNum) {}

	// You can pass a number as an argument for this constructor, and it will allocate that number of neurons as a layer
	Layer(unsigned pNum)
	{
		// Allocating memory for neurons
		neurons = new Neuron[pNum];

		// Setting number of neurons variable
		numNeurons = pNum;
	}

	// Getter and setter functions
	Neuron* getThisLayer() { return neurons;}				// Returns the Neuron array in this layer
	unsigned getNumberOfNeurons() { return numNeurons; }	// Returns the number of neurons in this layer

	void setNeurons(Neuron* p) { neurons = p; }
	void setNumberOfNeurons(unsigned p) { numNeurons = p; }
	void setLayerActivationFunction(const string& s) { for (int i = 0; i < numNeurons; i++) neurons[i].setActivationFunction(s); }

};

#endif /*_LAYER_H_*/