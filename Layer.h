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
	Layer() : neurons(new Neuron), numNeurons(0) {}
	~Layer() { delete[] neurons; }

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
	Neuron* getThisLayer() { if (neurons != nullptr)return neurons; else return nullptr; }				// Returns the Neuron array in this layer
	unsigned getNumberOfNeurons() { return numNeurons; }	// Returns the number of neurons in this layer

	void setNeurons(Neuron* parameter) { neurons = parameter; }

	// Deallocates old neuron array, then reallocating memory,
	// Copiing old data, and setting new parameters, user should only
	// pass the number of neurons in new layer
	void setNumberOfNeurons(unsigned parameter) 
	{ 
		Neuron* newLayer = new Neuron[parameter];

		// Setting all neurons to be default
		for (unsigned i = 0; i < parameter; i++)
			newLayer[i] = Neuron();

		// Copiing old data
		for (unsigned i = 0; i < numNeurons || i < parameter; i++)
			newLayer[i] = neurons[i];

		// Free previously allocated memory
		delete[] neurons;

		// Assigning newly created layer
		neurons = newLayer;

		// Setting new parameter
		numNeurons = parameter;
	}
	void setLayerActivationFunction(const string& s) { for (unsigned i = 0; i < numNeurons; i++) neurons[i].setActivationFunction(s); }

};

#endif /*_LAYER_H_*/