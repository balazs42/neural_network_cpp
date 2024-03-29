#pragma once

#ifndef _NEURON_H_
#define _NEURON_H_


#include <iostream>
#include <string>
#include <fstream>

#include <math.h>	// For activations functions
#include <iostream>	// String handling
#include <exception>// Exception handling
#include <cmath>
#include <algorithm>

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
	double z;					// Weighted input of neurons to this neurons
	
	double deltaActivation;		// Delta activation calculated from error
	double deltaBias;			// Delta bias calculated from error

	double firstMoment;			// First moment estimate 
	double secondMoment;		// Second moment estimate

	double squaredGradientSum;    // Accumulated squared gradients for Adagrad
	double squaredGradientAvg;    // Moving average of squared gradients for AdaDelta
	double deltaBiasAvg;		  // For Adadelta - running average of squared updates for bias

	typedef double (*FunctionPointer)(double);// Function pointer type

	// Activation function pointers
	FunctionPointer actFun;
	FunctionPointer derActFun;

public:
	Neuron() : activation(0.0f), bias(0.0f), error(0.0f), z(0.0f), actFun(&Sigmoid),
		derActFun(&DSigmoid), deltaActivation(0.0f), deltaBias(0.0f), 
		firstMoment(1.0f), secondMoment(1.0f), 
		squaredGradientAvg(1.0f), squaredGradientSum(1.0f), deltaBiasAvg(1.0f)
	{}

	// Override for operator=
	bool operator==(const Neuron& other)
	{
		return (bias == other.getBias() && activation == other.getActivation());
	}

	// Overriding operator=, so copiing all parameters
	Neuron operator=(const Neuron& rhs)
	{
		// Check for self-assignment
		if (this == &rhs)
			return *this;

		// Copy data
		activation = rhs.getActivation();
		bias = rhs.getBias();
		error = rhs.getError();
		z = rhs.getZ();
		deltaActivation = rhs.getDeltaActivation();
		deltaBias = rhs.getDeltaBias();
		firstMoment = rhs.getFirstMoment();
		secondMoment = rhs.getSecondMoment();
		squaredGradientSum = rhs.getSquaredGradientSum();
		squaredGradientAvg = rhs.getSquaredGradientAvg();
		actFun = rhs.getActivationFunction();
		derActFun = rhs.getDerivativeActivationFunction();

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
	double getDeltaBiasAvg() const { return deltaBiasAvg; }
	functionPtr getActivationFunction() const { return actFun; }
	functionPtr getDerivativeActivationFunction() const { return derActFun; }

	string getActivationFunctionString() const {
		if (actFun == &Sigmoid)
			return "Sigmoid";
		else if (actFun == &Relu)
			return "Relu";
		else if (actFun == &Tanh)
			return "Tanh";
		else if (actFun == &LeakyRelu)
			return "LeakyRelu";
		else if (actFun == &ELU)
			return "Elu";
		else if (actFun == &Swish)
			return "Swish";
	}

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
	void setDeltaBiasAvg(double parameter) { deltaBiasAvg = parameter; }

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

	// Min and max functions for clamping
	static double max(double l, double r) { return (l > r) ? l : r; }
	static double min(double l, double r) { return (l < r) ? l : r; }
public:
	void setActivationFunction(const string& s);

	// @return ActFun(this->z)
	double activationFunction() { return (this->actFun)(z); }

	// Returns the activation function applyed to the parameter
	// @param parameter The variable that you want to apply ActFunc() to
	// @return returns the ActFunc(parameter)
	double activationFunction(double parameter) { return (this->actFun)(parameter); }

	// Returns the derivative activation function applyed to the parameter
	// @param parameter The variable that you want to apply d/d ActFunc() to
	// @return returns the d/d ActFunc(parameter)
	double derivativeActivationFunction(double parameter) { return (this->derActFun)(parameter); }

	// Saver function
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