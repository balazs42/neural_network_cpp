#include "Neuron.h"

/********************************************/
/********** Activation functions ************/
/********************************************/

// Sigmoid activation function
double Neuron::Sigmoid(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

// Derivative of sigmoid activation function
double Neuron::DSigmoid(double z)
{
	return (Sigmoid(z) - (1.0 - Sigmoid(z)));
}

// Hyperbolic tangent activation function
double Neuron::Tanh(double z)
{
	return tanh(z);
}

// Derivative of the hyperbolic tangent function
double Neuron::DTanh(double z)
{
	return pow((1.0 / cosh(z)), 2);
}

// ReLu activation function
double Neuron::Relu(double z)
{
	if (z > 0) return z;
	else return 0;
}

// Derivative of ReLu function
double Neuron::DRelu(double z)
{
	return (z > 0) ? 1 : 0;
}

// Swish activation function
double Neuron::Swish(double z)
{
	return ((z / (1.0 + exp(-z))));
}

// Derivative of Swish function
double Neuron::DSwish(double z)
{
	return ((exp(-z) * (z + 1.0) + 1.0) / (pow((1.0 + exp(-z)), 2)));
}

// Function for setting activation function
void Neuron::setActivationFunction(const string& s)
{
	if (s == "Sigmoid")
	{
		this->activationFunction = &Sigmoid;
		this->derivativeActivationFunction = &DSigmoid;
	}
	else if (s == "Tanh")
	{
		this->activationFunction = &Tanh;
		this->derivativeActivationFunction = &DTanh;
	}
	else if (s == "Relu")
	{
		this->activationFunction = &Relu;
		this->derivativeActivationFunction = &DRelu;
	}
	else if (s == "Swish")
	{
		this->activationFunction = &Swish;
		this->derivativeActivationFunction = &DSwish;
	}
	else throw out_of_range("Invalid activation function, check code!");
}