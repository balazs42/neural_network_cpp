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

	double firstMoment;			// First moment estimate (Adam optimization)
	double secondMoment;		// Second moment estimate (Adam optimization)

	typedef double (*FunctionPointer)(double);// Function pointer type

	// Activation function pointers
	FunctionPointer activationFunction;
	FunctionPointer derivativeActivationFunction;

public:
	Neuron() : activation(0.0f), bias(0.0f), error(0.0f), z(0.0f), activationFunction(&Sigmoid),
		derivativeActivationFunction(&DSigmoid), deltaActivation(0.0f), deltaBias(0.0f), 
		firstMoment(0.0f), secondMoment(0.0f) {}

	// Overriding operator=, so copiing all parameters
	Neuron operator=(const Neuron& rhs)
	{
		activation = rhs.activation;
		bias = rhs.bias;
		error = rhs.error;
		z = rhs.z;

		deltaActivation = rhs.deltaActivation;
		deltaBias = rhs.deltaBias;

		firstMoment = rhs.firstMoment;
		secondMoment = rhs.secondMoment;

		activationFunction = rhs.activationFunction;
		derivativeActivationFunction = rhs.derivativeActivationFunction;
	}

	// Getter functions
	double getActivation() { return activation; }
	double getBias() { return bias; }
	double getError() { return error; }
	double getDeltaActivation() { return deltaActivation; }
	double getDeltaBias() { return deltaBias; }
	double getZ() { return z; }
	double getFirstMoment() { return firstMoment; }
	double getSecondMoment() { return secondMoment; }

	// Setter functions
	void setActivation(double parameter) { activation = parameter; }
	void setBias(double parameter) { bias = parameter; }
	void setError(double parameter) { error = parameter; }
	void setDeltaActivation(double parameter) { deltaActivation = parameter; }
	void setDeltaBias(double parameter) { deltaBias = parameter; }
	void setZ(double parameter) { z = parameter; }
	void setFirstMoment(double parameter) { firstMoment = parameter; }
	void setSecondMoment(double parameter) { secondMoment = parameter; }

private:
	// Activation functions
	static double Sigmoid(double z);
	static double DSigmoid(double z);
	static double Tanh(double z);
	static double DTanh(double z);
	static double Relu(double z);
	static double DRelu(double z);
	static double Swish(double z);
	static double DSwish(double z);
	static double LeakyRelu(double z);
	static double DLeakyRelu(double z);
	static double ELU(double z);
	static double DELU(double z);
public:
	void setActivationFunction(const string& s);
	double activateNeuron() { return (this->activationFunction)(z); }
	double activateNeuron(double parameter) { return (this->activationFunction)(parameter); }
	double activateDerivative(double parameter) { return (this->activationFunction)(parameter); }
};

#endif /*_NEURON_H_*/