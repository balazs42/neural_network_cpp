#include "Neuron.h"

/********************************************/
/********** Activation functions ************/
/********************************************/

double Neuron::Sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double Neuron::DSigmoid(double z) {
    return Sigmoid(z) * (1.0 - Sigmoid(z));
}

double Neuron::Tanh(double z) {
    return tanh(z);
}

double Neuron::DTanh(double z) {
    return 1.0 - tanh(z) * tanh(z);
}

double Neuron::Relu(double z) {
    return (z > 0) ? z : 0;
}

double Neuron::DRelu(double z) {
    return (z > 0) ? 1 : 0;
}

double Neuron::Swish(double z) {
    return z * Sigmoid(z);
}

double Neuron::DSwish(double z) {
    double sigmoidZ = Sigmoid(z);
    return sigmoidZ + z * sigmoidZ * (1.0 - sigmoidZ);
}

double Neuron::LeakyRelu(double z) {
    const double alpha = 0.01; // Leaky coefficient
    return (z > 0) ? z : alpha * z;
}

double Neuron::DLeakyRelu(double z) {
    const double alpha = 0.01; // Leaky coefficient
    return (z > 0) ? 1 : alpha;
}

double Neuron::ELU(double z) {
    const double alpha = 1.0; // ELU coefficient
    return (z > 0) ? z : alpha * (exp(z) - 1);
}

double Neuron::DELU(double z) {
    const double alpha = 1.0; // ELU coefficient
    return (z > 0) ? 1 : alpha * exp(z);
}

void Neuron::setActivationFunction(const string& s) {
    if (s == "Sigmoid") {
        this->activationFunction = &Sigmoid;
        this->derivativeActivationFunction = &DSigmoid;
    }
    else if (s == "Tanh") {
        this->activationFunction = &Tanh;
        this->derivativeActivationFunction = &DTanh;
    }
    else if (s == "Relu") {
        this->activationFunction = &Relu;
        this->derivativeActivationFunction = &DRelu;
    }
    else if (s == "Swish") {
        this->activationFunction = &Swish;
        this->derivativeActivationFunction = &DSwish;
    }
    else if (s == "LeakyRelu") {
        this->activationFunction = &LeakyRelu;
        this->derivativeActivationFunction = &DLeakyRelu;
    }
    else if (s == "ELU") {
        this->activationFunction = &ELU;
        this->derivativeActivationFunction = &DELU;
    }
    else {
        throw out_of_range("Invalid activation function, check code!");
    }
}