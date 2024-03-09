#include "Network.h"

/***************************************/
/******* Initializing functions ********/
/***************************************/

// Function to generate a random number
double Network::getRandomNumber() 
{
    // Generate a random number between -1.0 and 1.0
    return (rand() / (RAND_MAX / 2.0)) - 1.0;
}

void Network::randInitEdges()
{
    // Seed the random number generator
    srand(time(nullptr));

    // Iterating through each edge and setting it to a random values
    for (unsigned i = 0; i < layers.size() - 1; i++)
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
            for (unsigned k = 0; k < layers[i + 1].getNumberOfNeurons(); k++)
                edges[i][j][k].setWeight(getRandomNumber());                    // Setting each weight with a random number
}

void Network::randInitNeurons()
{
    // Seed the random number generator
    srand(time(nullptr));

    for (unsigned i = 0; i < layers.size(); i++)
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
            layers[i].getThisLayer()->setBias(getRandomNumber());               // Setting each bias with a random number
}

/***************************************/
/****** File handling functions ********/
/***************************************/

// Loading network from a file
void Network::loadNetworkFromFile(const string& route)
{
    
}

// Saving network to a file
void Network::saveNetworkToFile(const string& route)
{

}

/***************************************/
/******* Normalizing functions *********/
/***************************************/

template<typename T>
double* Network::normalizeData(T* arr, unsigned num)
{
    // Initialize min and max with the extreme values of the data type
    T max = std::numeric_limits<T>::lowest();
    T min = std::numeric_limits<T>::max();

    // Find the min and max values in the array
    for (unsigned i = 0; i < num; i++) 
    {
        if (arr[i] > max) 
            max = arr[i];
        
        if (arr[i] < min) 
            min = arr[i];
    }

    if (min == max)
    {
        throw out_of_range("Division by zero, check arrays, at noramlizing input or output!");
        exit(-8);
    }

    // Normalize the data
    for (unsigned i = 0; i < num; i++) 
        arr[i] = (arr[i] - min) / (max - min);

    // Allocating double return array
    double* retArr = new double[num];

    // Converting everything to a double array
    for (unsigned i = 0; i < num; i++)
        retArr[i] = (double)arr[i];

    return retArr;
}

/***************************************/
/********* Training functions **********/
/***************************************/

template<typename T>
void Network::feedForwardNetwork(T* inputArr, unsigned num)
{
    // Check if input array size is correct
    if (num > layers[0].getNumberOfNeurons())
    {
        // Input data will be lost, if input data is hierarchly ordered, and last portion of data is not necessary this could be ignored
        throw out_of_range("Input array is larger then input layer, data will be lost, check code if this is a problem!");
    }

    // 1st step is normalizing input array, to 0-1 values, and setting first layers activations as these values

    // Normalizing data
    double* normalizedInput = normalizeData(inputArr, num);

    // Setting first layer's activations
    Neuron* firstLayer = layers[0].getThisLayer();
    unsigned numberOfNeurons = layers[0].getNumberOfNeurons();

    for (unsigned i = 0; i < num; i++)
        firstLayer[i].setActivation(normalizedInput[i]);

    // Starting feedforward process
    for (unsigned i = 0; i < layers.size() - 1; i++)
    {
        // Iterationg through each neuron in the right layer
        for (unsigned j = 0; j < layers[i + 1].getNumberOfNeurons(); j++)
        {
            // Neuron array on the right side
            Neuron* rightLayer = layers[i + 1].getThisLayer();

            // Neuron array on the left side
            Neuron* leftLayer = layers[i].getThisLayer();

            double z = 0.0f;
            // Iterating through each neuron in left layer
            for (unsigned k = 0; k < layers[i].getNumberOfNeurons(); k++)
            {
                // calc    =        left activation       *       edge between         
                z += leftLayer[k].getActivation() * edges[i][k][j].getWeight();
            }
            // Adding bias to the activation
            z += rightLayer[j].getBias();

            // Saving z value
            rightLayer[j].setZ(z);

            // Setting current neurons activation as: activation = activationFunction(z); [for example: a = sigmoid(z)]
            z = rightLayer[j].activateNeuron(rightLayer[j].getZ());
            rightLayer[j].setActivation(z);
        }
    }
}

// Calculating network's error based on the provided expected array
template<typename T>
void Network::calculateError(T* expectedArray, unsigned num)
{
    // Checking if expected array is same size as output layer
    if (num != layers[layers.size() - 1].getNumberOfNeurons())
    {
        throw out_of_range("Expected array not same size as output layer, check code!");
        exit(-9);
    }

    // First error calculation in the last layer based on the given expected array

    // Normalizing expected array
    double* normalizedExpected = normalizeData(expectedArray, num);

    Neuron* lastLayer = layers[layers.size() - 1].getNumberOfNeurons();

    // Calculating error in final layer
    for (unsigned i = 0; i < num; i++)
    {
        // Cost function is: E = (a^L - y)^2, d/dE = 2 * (a^L - y) , where y is the expected value at a given index, and a^L is the L-th layer's activation at the given index
        // So the error should be calculated like this: 
        //           Error =  2 *       activation             -  ||expected value||
        lastLayer[i].setError(2 * lastLayer[i].getActivation() - normalizedExpected[i]);
    }
}

// Calculate delta activations in the network for all layers
// Should be called, after the first error calculating in the rightmost layer is done
void Network::calculateDeltaActivation()
{
    double error = 0.0f;

    // Going through each layer, starting from right to left
    for (unsigned i = layers.size() - 1; i > 0; i--)
    {
        Neuron* rightLayer = layers[i].getThisLayer();
        Neuron* leftLayer = layers[i - 1].getThisLayer();

        // Iteratin through left layers neurons
        for (unsigned j = 0; j < layers[i - 1].getNumberOfNeurons(); j++)
        {
            error = 0.0f;
            // Iterating through right layer's neurons
            for (unsigned k = 0; k < layers[i].getNumberOfNeurons(); k++)
            {
                // Error = edge between right and left layer's neurons * d/dActFun(z)               * righ layer's neuron's error
                error += edges[i - 1][j][k].getWeight() * rightLayer[k].activateDerivative(rightLayer[k].getZ()) * rightLayer[k].getError();
            }
            leftLayer[j].setError(error);
        }
    }
}

// Calculate delat bias values in all of the network
void Network::calculateDeltaBias()
{
    double dBias = 0.0f;
    for (unsigned i = 0; i < layers.size(); i++)
    {
        dBias = 0.0f;

        // Current layer's neuron array
        Neuron* thisLayer = layers[i].getThisLayer();

        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // dBias =                      d/dActFun(z)                 *  current neuron error     
            dBias = thisLayer[j].activateDerivative(thisLayer[j].getZ()) * thisLayer[j].getError();
            
            // Setting calculated delta bias
            thisLayer[j].setDeltaBias(dBias);
        }
    }
}

void Network::calculateDeltaWeight()
{
    double dWeight = 0.0f;

    for (unsigned i = 0; i < edges.size(); i++)
    {
        // Number of neurons in the left layer, and left layer's neuron array
        unsigned leftSize = layers[i].getNumberOfNeurons();
        Neuron* leftLayer = layers[i].getThisLayer();

        // Number of neurons in the right layer, and right layer's neuron array
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();
        Neuron* rightLayer = layers[i + 1].getThisLayer();

        for (unsigned j = 0; j < leftSize; i++)
        {
            for (unsigned k = 0; k < rightSize; k++)
            {
                // DeltaWeight =  left activation      *                      d/dActFun(z)                      *   right neuron error      
                dWeight = leftLayer[j].getActivation() * rightLayer[k].activateDerivative(rightLayer[j].getZ()) * rightLayer[k].getError();

                // Setting delta weight
                edges[i][j][k].setDeltaWeight(dWeight);
            }
        }
    }
}

// First setting the new weight to the edges based on the calculated delta values
// Then setting the new biases to the neurons based on the calculated delta values
void Network::setNewParameters()
{
    double newWeight = 0.0f;

    // Setting new weights for edges
#pragma omp parallel for
    for (unsigned i = 0; i < edges.size(); i++)
    {
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            for (unsigned k = 0; k < layers[i + 1].getNumberOfNeurons(); k++)
            {
                // New weight =         old weigth     -      calculated delta weight
                newWeight = edges[i][j][k].getWeight() - edges[i][j][k].getDeltaWeight();
                
                // Setting new weight
                edges[i][j][k].setWeight(newWeight);
            }
        }
    }

    double newBias = 0.0f;

    // Setting new biases for each neuron in each layer
#pragma omp parallel for
    for (unsigned i = 0; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // New bias =       old bias     -  calculated delta bias
            newBias = thisLayer[j].getBias() - thisLayer[j].getDeltaBias();

            // Setting new bias
            thisLayer[j].setBias(newBias);
        }
    }
}

// Implementation of 1 backpropagation process
template <typename T1, typename T2>
void Network::backPropagation(T1* inputArr, unsigned inNum, T2* expectedArr, unsigned expNum)
{
    // Feedforward process
    feedForwardNetwork(inputArr, inNum);

    // First error calculation in last layer
    calculateError(expectedArr, expNum);

    // Calculating delta activations
    calculateDeltaActivation();

    // Parallelize delta bias calculation
#pragma omp parallel
    {
        calculateDeltaBias();
    }

    // Parallelize delta weight calculation
#pragma omp parallel
    {
        calculateDeltaWeight();
    }

    // Setting new parameters to the network
    setNewParameters();

    if (useAdam)
        adamOptimization();
}

// Update weights and biases using Adam optimization
void Network::adamOptimization(double learningRate, double beta1, double beta2, double epsilon)
{
    // Initialize Adam parameters
    double beta1Power = 1.0;
    double beta2Power = 1.0;

    // Perform Adam optimization for each neuron's bias
    // -------------------------------------------------------------------
    // #pragma omp parallel for schedule(dynamic) instructs the compiler 
    // to parallelize the following loop using OpenMP, with the iterations 
    // dynamically scheduled among the threads to achieve workload balance.
    // -------------------------------------------------------------------
#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Update first and second moments
            thisLayer[j].setFirstMoment(beta1 * thisLayer[j].getFirstMoment() + (1 - beta1) * thisLayer[j].getDeltaBias());
            thisLayer[j].setSecondMoment(beta2 * thisLayer[j].getSecondMoment() + (1 - beta2) * thisLayer[j].getDeltaBias() * thisLayer[j].getDeltaBias());

            // Bias correction
            double firstMomentCorrected = thisLayer[j].getFirstMoment() / (1 - beta1Power);
            double secondMomentCorrected = thisLayer[j].getSecondMoment() / (1 - beta2Power);

            // Update bias using the learning rate, firstMomentCorrected, secondMomentCorrected, and epsilon
            thisLayer[j].setBias(thisLayer[j].getBias() - learningRate * firstMomentCorrected / (std::sqrt(secondMomentCorrected) + epsilon));
        }
    }

    // Update beta1Power and beta2Power
    beta1Power *= beta1;
    beta2Power *= beta2;
}

template <typename T1, typename T2>
void Network::stochasticGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum)
{

}

// Gradient descent method for training the network, TODO: modifiable threshhold value, currently 0.1% is the static thrashhold
template <typename T1, typename T2>
void Network::gradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum)
{
    const double thrashHold = 0.001f;
    double error = 0.0f;

    // TODO: modifiable thrashhold

    while (error > thrashHold)
    {
        error = 0.0f;

        // Completing full training on given data arrays
        for (unsigned i = 0; i < inArr.size(); i++)
            backPropagation(inArr[i], inNum[i], expArr[i], expNum[i]);
        
        // Checking of the networks error rate is lower then the given thrashhold
        Neuron* lastLayer = layers[layers.size() - 1].getThisLayer();

        for (unsigned i = 0; i < layers[layers.size() - 1].getNumberOfNeurons(); i++)
            error += lastLayer[i].getError();
    }
}

template <typename T1, typename T2>
void Network::minibatchGradientDescent(vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum)
{
    for (unsigned epoch = 0; epoch < epochNum; ++epoch) 
    {
        // Shuffle the training data for each epoch
        std::random_shuffle(inArr.begin(), inArr.end());

        // Process mini-batches
        for (unsigned i = 0; i < trainingData.size(); i += miniBatchSize) 
        {
            // Get the mini-batch
            std::vector<Data> miniBatch(trainingData.begin() + i, trainingData.begin() + std::min(i + miniBatchSize, trainingData.size()));

            // Update network parameters using the mini-batch
            updateMiniBatch(miniBatch, learningRate);
        }
    }
}

/**
 * Trains the neural network using the specified training method and parameters.
 *
 * @param s Method selector string indicating the training algorithm to be used.
 * @param inArr Vector of input arrays. Each array must be of numerical type.
 * @param inNum Vector containing the lengths of the input arrays.
 * @param expArr Vector of expected value arrays. Each array must be of numerical type.
 * @param expNum Vector containing the lengths of the expected value arrays.
 * @param epochNum Number of epochs for training.
 * @tparam T1 Type of the input array elements.
 * @tparam T2 Type of the expected value array elements.
 */
template <typename T1, typename T2>
void Network::trainNetwork(const string& s, vector<T1*> inArr, vector<unsigned> inNum, vector<T2*> expArr, vector<unsigned> expNum, unsigned epochNum)
{
    if (s == "Minibatch" || s == "minibatch" || s == "mb")
    {
        // Training using the minibatch gradient descent method
        minibatchGradientDescent(inArr, inNum, expArr, expNum, epochNum);
    }
    else if (s == "StochasticGradientDescent" || s == "SGD" || s == "sgd")
    {
        // Traingin using the stochastic gradient descent method
        stochasticGradientDescent(inArr, inNum, expArr, expNum, epochNum);
    }
    else if (s == "GradientDescent" || s == "GD" || s == "gd")
    {
        // Trainging using the gradient descent methdo
        gradientDescent(inArr, inNum, expArr, expNum, epochNum);
    }
    else
    {
        throw out_of_range("Invalid traingin technique, check code!");
        exit(-10);
    }
}