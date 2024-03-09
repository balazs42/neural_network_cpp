#ifndef _NEURON_H_
#define _NEURON_H_

#include <math.h>	// For activations functions
#include <iostream>	// String handling
#include <exception>// Exception handling

using std::string;
using std::out_of_range;

// Define finctionPtr to be a function pointer
typedef double(*functionPtr)(double);

// Neuron class
class Neuron
{
private:
	double activation;			// Activation of the neuron	
	double bias;				// Bias of the neuron
	double error;				// Error in the neuron
	double z;					// Parameter that will be passed to the activation function
	
	double deltaActivation;		// Delta activation calculated from error
	double deltaBias;			// Delta bias calculated from error

	typedef double (Neuron::* FunctionPointer)(double);// Function pointer type

	// Activation function pointers
	FunctionPointer activationFunction;
	FunctionPointer derivativeActivationFunction;

public:
	Neuron() : activation(0.0f), bias(0.0f), error(0.0f), z(0.0f), activationFunction(Sigmoid), derivativeActivationFunction(DSigmoid), deltaActivation(0.0f), deltaBias(0.0f) {}

	// Getter functions
	double getActivation() { return activation; }
	double getBias() { return bias; }
	double getError() { return error; }
	double getDeltaActivation() { return deltaActivation; }
	double getDeltaBias() { return deltaBias; }
	double getZ() { return z; }

	// Setter functions
	void setActivation(double p) { activation = p; }
	void setBias(double p) { bias = p; }
	void setError(double p) { error = p; }
	void setDeltaActivation(double p) { deltaActivation = p; }
	void setDeltaBias(double p) { deltaBias = p; }
	void setZ(double p) { z = p; }

private:
	// Activation functions
	double Sigmoid(double z);
	double DSigmoid(double z);
	double Tanh(double z);
	double DTanh(double z);
	double Relu(double z);
	double DRelu(double z);
	double Swish(double z);
	double DSwish(double z);
public:
	void setActivationFunction(const string& s);
	double activateNeuron() { return (this->*activationFunction)(z); }
	double activateNeuron(double p) { return (this->*activationFunction)(p); }
	double activateDerivative(double p) { return (this->*activationFunction)(p); }
};

#endif /*_NEURON_H_*/