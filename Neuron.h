#ifndef _NEURON_H_
#define _NEURON_H_

#include <iostream>
#include <string>
#include <fstream>

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

	double firstMoment;			// First moment estimate 
	double secondMoment;		// Second moment estimate

	double squaredGradientSum;    // Accumulated squared gradients for Adagrad
	double squaredGradientAvg;    // Moving average of squared gradients for AdaDelta

	typedef double (*FunctionPointer)(double);// Function pointer type

	// Activation function pointers
	FunctionPointer activationFunction;
	FunctionPointer derivativeActivationFunction;

public:
	Neuron() : activation(0.0f), bias(0.0f), error(0.0f), z(0.0f), activationFunction(&Sigmoid),
		derivativeActivationFunction(&DSigmoid), deltaActivation(0.0f), deltaBias(0.0f), 
		firstMoment(0.0f), secondMoment(0.0f),
		squaredGradientAvg(0.0f), squaredGradientSum(0.0f){}

	// Overriding operator=, so copiing all parameters
	Neuron operator=(const Neuron& rhs)
	{
		// Check for self-assignment
		if (this == &rhs)
			return *this;

		// Copy data
		activation = rhs.activation;
		bias = rhs.bias;
		error = rhs.error;
		z = rhs.z;
		deltaActivation = rhs.deltaActivation;
		deltaBias = rhs.deltaBias;
		firstMoment = rhs.firstMoment;
		secondMoment = rhs.secondMoment;
		squaredGradientSum = rhs.squaredGradientSum;
		squaredGradientAvg = rhs.squaredGradientAvg;
		activationFunction = rhs.activationFunction;
		derivativeActivationFunction = rhs.derivativeActivationFunction;

		// Return reference to the left-hand side object
		return *this;
	}

	// Getter functions
	double getActivation() const { return activation; }
	double getBias() const { return bias; }
	double getError() const { return error; }
	double getDeltaActivation() const { return deltaActivation; }
	double getDeltaBias() const { return deltaBias; }
	double getZ() const{ return z; }
	double getFirstMoment() const { return firstMoment; }
	double getSecondMoment() const { return secondMoment; }
	double getSquaredGradientSum() const { return squaredGradientSum; }
	double getSquaredGradientAvg() const { return squaredGradientAvg; }


	// Setter functions
	void setActivation(double parameter) { activation = parameter; }
	void setBias(double parameter) { bias = parameter; }
	void setError(double parameter) { error = parameter; }
	void setDeltaActivation(double parameter) { deltaActivation = parameter; }
	void setDeltaBias(double parameter) { deltaBias = parameter; }
	void setZ(double parameter) { z = parameter; }
	void setFirstMoment(double parameter) { firstMoment = parameter; }
	void setSecondMoment(double parameter) { secondMoment = parameter; }
	void setSquaredGradientSum(double parameter) { squaredGradientSum = parameter; }
	void setSquaredGradientAvg(double parameter) { squaredGradientAvg = parameter; }

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

	void saveNeuronToXML(std::ofstream& outFile) 
	{
		// Write XML tags to represent neuron properties
		outFile << "<Neuron>" << "\n";
		outFile << "    <Bias>" << bias << "</Bias>" << "\n";

		// Add more properties as needed
		outFile << "</Neuron>" << "\n";
	}
};

#endif /*_NEURON_H_*/