#ifndef _EDGE_H_
#define _EDGE_H_

class Edge 
{
private:
	double weight;
	double error;
	double deltaWeight;

public:
	Edge() : weight(0.0f), error(0.0f), deltaWeight(0.0f) {}

	// Getter functions
	double getWeight() { return weight; }
	double getError() { return error; }
	double getDeltaWeight() { return deltaWeight; }

	// Setter functions
	void setWeight(double parameter) { weight = parameter; }
	void setError(double parameter) { error = parameter; }
	void setDeltaWeight(double parameter) { deltaWeight = parameter; }
};

#endif /*_EDGE_H_*/