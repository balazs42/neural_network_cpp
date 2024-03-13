#include "Neuron.h"

/********************************************/
/********** Activation functions ************/
/********************************************/

// Sigmoid function with a check to avoid extremely large input values
double Neuron::Sigmoid(double z) 
{
    // Clamp z to avoid overflow in exp(-z)
    double clampedZ = min(z, 20.0f);
    clampedZ = max(clampedZ, -20.0f);
    return 1.0 / (1.0 + exp(-clampedZ));
}

// Derivative of the Sigmoid function
double Neuron::DSigmoid(double z) 
{
    double sig = Sigmoid(z);
    return sig * (1.0f - sig);
}

// Hyperbolic tangent function as is, since std::tanh is already quite stable
double Neuron::Tanh(double z) 
{
    return tanh(z);
}

// Derivative of the hyperbolic tangent function
double Neuron::DTanh(double z) 
{
    double t = Tanh(z);
    return 1.0f - t * t;
}

// Rectified Linear Unit (ReLU)
double Neuron::Relu(double z) 
{
    return (z > 0.0f) ? z : 0.0f;
}

// Derivative of ReLU
double Neuron::DRelu(double z) 
{
    return (z > 0.0f) ? 1.0f : 0.0f;
}

// Swish function, which is sigmoid-weighted input, stable as per Sigmoid adjustments
double Neuron::Swish(double z) 
{
    return z * Sigmoid(z);
}

// Derivative of the Swish function
double Neuron::DSwish(double z) 
{
    double sigmoidZ = Sigmoid(z);
    return sigmoidZ + z * sigmoidZ * (1.0 - sigmoidZ);
}

// Leaky Rectified Linear Unit
double Neuron::LeakyRelu(double z) 
{
    const double alpha = 0.01f; // Leaky coefficient
    return (z > 0.0f) ? z : alpha * z;
}

// Derivative of Leaky ReLU
double Neuron::DLeakyRelu(double z) 
{
    const double alpha = 0.01f; // Leaky coefficient
    return (z > 0.0f) ? 1.0f : alpha;
}

// Exponential Linear Unit (ELU) with checks to avoid overflow
double Neuron::ELU(double z) 
{
    const double alpha = 1.0f; // ELU coefficient
    return (z > 0.0f) ? z : alpha * (exp(max(z, (double)(-20.0f))) - 1.0f);
}

// Derivative of ELU
double Neuron::DELU(double z) 
{
    const double alpha = 1.0f; // ELU coefficient
    return (z > 0) ? 1 : alpha * exp(z);
}

void Neuron::setActivationFunction(const string& s) 
{
    if (s == "Sigmoid") 
    {
        this->actFun = &Sigmoid;
        this->derActFun = &DSigmoid;
    }
    else if (s == "Tanh") 
    {
        this->actFun = &Tanh;
        this->derActFun = &DTanh;
    }
    else if (s == "Relu") 
    {
        this->actFun = &Relu;
        this->derActFun = &DRelu;
    }
    else if (s == "Swish") 
    {
        this->actFun = &Swish;
        this->derActFun = &DSwish;
    }
    else if (s == "LeakyRelu") 
    {
        this->actFun = &LeakyRelu;
        this->derActFun = &DLeakyRelu;
    }
    else if (s == "ELU") 
    {
        this->actFun = &ELU;
        this->derActFun = &DELU;
    }
    else {
        throw out_of_range("Invalid activation function, check code!");
    }
}