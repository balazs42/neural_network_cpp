#pragma once

#ifndef _EDGE_H_
#define _EDGE_H_

#include <iostream>
#include <fstream>
#include <string>

using std::string;

class Edge 
{
private:
	double weight;			// Weight of the edge
	double error;			// Calcuated error on the edge
	double deltaWeight;		// Calculated delta weight on the edge
	double deltaWeightAvg;	// Average delta weight for optimization algorithms
	double squaredGradientAvg;

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