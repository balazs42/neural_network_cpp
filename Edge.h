#pragma once

#ifndef _EDGE_H_
#define _EDGE_H_

#include <iostream>
#include <fstream>
#include <string>

using std::string;

// Class representing a connection (edge) between two neurons in the neural network.
// Contains information about the weight of the connection, error, delta weight,
// and other attributes relevant for optimization algorithms.
class Edge 
{
private:
	double weight;				// Weight of the edge
	double error;				// Calculated error on the edge, used in backpropagation
	double deltaWeight;			// Change in weight to be applied during optimization
	double deltaWeightAvg;		// Average change in weight, used by some optimizers
	double squaredGradientAvg;	// Average of squared gradients, used in optimizers like RMSProp and AdaDelta

public:
	Edge() : weight(0.0f), error(0.0f), deltaWeight(0.0f), deltaWeightAvg(0.0f), squaredGradientAvg(0.0f) {}

	// Getter functions
	double getWeight() const { return weight; }
	double getError() const { return error; }
	double getDeltaWeight() const { return deltaWeight; }
	double getDeltaWeightAvg() const { return deltaWeightAvg; }
	double getSquaredGradientAvg() const { return squaredGradientAvg; }

	// Setter functions
	void setWeight(double parameter) { weight = parameter; }
	void setError(double parameter) { error = parameter; }
	void setDeltaWeight(double parameter) { deltaWeight = parameter; }
	void setDeltaWeightAvg(double parameter) { deltaWeightAvg = parameter; }
	void setSquaredGradientAvg(double parameter) { squaredGradientAvg = parameter; }

	void saveEdgeToXML(std::ofstream& outFile) 
	{
		// Write XML tags to represent edge properties
		outFile << "<Edge>" << std::endl;
		outFile << "    <Weight>" << weight << "</Weight>" << std::endl;
		// Add more properties as needed
		outFile << "</Edge>" << std::endl;
	}
};

#endif /*_EDGE_H_*/