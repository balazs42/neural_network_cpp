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

// Initializing edges with random numbers
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

// Initializing neuron biases with random numbers
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

// Save network to file
void Network::saveNetworkToXML(const string& filename, Network& network)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error: Unable to open file for writing: " << filename << "\n";
        return;
    }

    // Write XML header
    outFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << "\n";
    outFile << "<Network>" << "\n";

    //<parameter> layer1NeuronNum layer2NeuronNum ... LayerNNeuronNum OptimizationTechnique</parameter> 
    outFile << "<parameter> ";
    for (unsigned i = 0; i < network.getLayers().size(); i++)
        outFile << network.getLayers()[i].getNumberOfNeurons() << " ";
    
    // Printing optimization technique
    string opt = network.getOptimization();

    outFile << opt << "\n";

    outFile << "</parameter>" << "\n";

    // Save neurons
    for (unsigned i = 0; i < network.getLayers().size(); i++)
    {
        Neuron* thisLayer = network.getLayers()[i].getThisLayer();

        // Saving each neuron
        for (unsigned j = 0; j < network.getLayers()[i].getNumberOfNeurons(); j++)
            thisLayer[j].saveNeuronToXML(outFile);
    }

    // Save edges
    for (unsigned i = 0; i < network.getEdges().size(); i++)
    {
        for (unsigned j = 0; j < network.getLayers()[i].getNumberOfNeurons(); j++)
        {
            for (unsigned k = 0; k < network.getLayers()[i + 1].getNumberOfNeurons(); k++)
            {
                network.getEdges()[i][j][k].saveEdgeToXML(outFile);
            }
        }
    }

    // Close XML tag
    outFile << "</Network>" << "\n";

    outFile.close(); // Close the file
}

// Load network to previously created network
void Network::loadNetworkFromXML(const string& route)
{

}

/***************************************/
/********* Training functions **********/
/***************************************/

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

        // Parallelize delta bias calculation for each neuron in the current layer
#pragma omp parallel for
        for (unsigned i = 0; i < layers[i].getNumberOfNeurons(); ++i)
        {
            double deltaBias = thisLayer[i].getError() * thisLayer[i].activateDerivative(thisLayer[i].getZ());
            thisLayer[i].setDeltaBias(deltaBias);
        }
    }
}

// Calculating delta weights
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

/***************************************/
/******* Optimization functions ********/
/***************************************/

// Update weights and biases using Adam optimization (parallelized)
void Network::adamOptimization(double learningRate, double beta1, double beta2, double epsilon)
{
    // Initialize Adam parameters
    double beta1Power = 1.0;
    double beta2Power = 1.0;

    // Perform Adam optimization for each neuron's bias
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (unsigned i = 1; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // Update first and second moments
            double deltaBias = thisLayer[j].getDeltaBias();
            double firstMoment = thisLayer[j].getFirstMoment();
            double secondMoment = thisLayer[j].getSecondMoment();
            thisLayer[j].setFirstMoment(beta1 * firstMoment + (1 - beta1) * deltaBias);
            thisLayer[j].setSecondMoment(beta2 * secondMoment + (1 - beta2) * deltaBias * deltaBias);

            // Bias correction
            double firstMomentCorrected = thisLayer[j].getFirstMoment() / (1 - beta1Power);
            double secondMomentCorrected = thisLayer[j].getSecondMoment() / (1 - beta2Power);

            // Update bias using the learning rate, firstMomentCorrected, secondMomentCorrected, and epsilon
            double updatedBias = thisLayer[j].getBias() - learningRate * firstMomentCorrected / (sqrt(secondMomentCorrected) + epsilon);
            thisLayer[j].setBias(updatedBias);
        }
    }

    // Update beta1Power and beta2Power
    beta1Power *= beta1;
    beta2Power *= beta2;
}

// Update weights and biases using RMSProp optimization
void Network::rmspropOptimization(double learningRate, double decayRate, double epsilon) 
{
    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

#pragma omp paralell for
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute gradients
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update moving average of squared gradients
            double squaredGradient = deltaBias * deltaBias;
            double squaredGradientAvg = decayRate * thisLayer[j].getSquaredGradientAvg() + (1 - decayRate) * squaredGradient;
            thisLayer[j].setSquaredGradientAvg(squaredGradientAvg);

            // Update bias using the learning rate and scaled gradient
            double scaleFactor = learningRate / (sqrt(squaredGradientAvg) + epsilon);
            double updatedBias = thisLayer[j].getBias() - scaleFactor * deltaBias;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

// Update weights and biases using Adagrad optimization
void Network::adagradOptimization(double learningRate, double epsilon) 
{
    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

#pragma omp paralell for
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute gradients
            double deltaBias = thisLayer[j].getDeltaBias();

            // Accumulate squared gradients
            double squaredGradientSum = thisLayer[j].getSquaredGradientSum() + deltaBias * deltaBias;
            thisLayer[j].setSquaredGradientSum(squaredGradientSum);

            // Update bias using the learning rate and scaled gradient
            double scaleFactor = learningRate / (sqrt(squaredGradientSum) + epsilon);
            double updatedBias = thisLayer[j].getBias() - scaleFactor * deltaBias;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

// Update weights and biases using AdaDelta optimization
void Network::adadeltaOptimization(double decayRate, double epsilon)
{
    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        Edge** edgeArray = edges[i - 1];

#pragma omp paralell for
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // Compute gradients
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update moving average of squared gradients
            double squaredGradientAvg = decayRate * thisLayer[j].getSquaredGradientAvg() + (1 - decayRate) * deltaBias * deltaBias;
            thisLayer[j].setSquaredGradientAvg(squaredGradientAvg);

            // Get delta weight average from the corresponding edge
            double deltaWeightAvg = edgeArray[j]->getDeltaWeightAvg();

            // Compute parameter update
            double update = -sqrt(deltaWeightAvg + epsilon) / sqrt(squaredGradientAvg + epsilon) * deltaBias;

            // Update weights using AdaDelta update rule
            double updatedBias = thisLayer[j].getBias() + update;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

// Update weights and biases using Nesterov Accelerated Gradient (NAG) optimization
void Network::nagOptimization(double learningRate, double momentum) 
{
    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
#pragma omp paralell for
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute gradients
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update momentum
            double prevMomentum = thisLayer[j].getFirstMoment();
            double newMomentum = momentum * prevMomentum + learningRate * deltaBias;
            thisLayer[j].setFirstMoment(newMomentum);

            // Update bias using NAG update rule
            double updatedBias = thisLayer[j].getBias() - momentum * prevMomentum + (1 + momentum) * newMomentum;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

// Update weights and biases using Adamax optimization
void Network::adamaxOptimization(double learningRate, double beta1, double beta2, double epsilon) 
{
    // Initialize beta1Power and beta2Power
    double beta1Power = 1.0;
    double beta2Power = 1.0;

    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

#pragma omp paralell for
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute gradients
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update first and second moments
            double firstMoment = beta1 * thisLayer[j].getFirstMoment() + (1 - beta1) * deltaBias;
            double secondMoment = fmax(beta2 * thisLayer[j].getSecondMoment(), fabs(deltaBias));

            // Update bias using the learning rate, firstMoment, secondMoment, and epsilon
            double updatedBias = thisLayer[j].getBias() - (learningRate / (1 - beta1Power)) * (firstMoment / (secondMoment + epsilon));
            thisLayer[j].setBias(updatedBias);

            // Update beta1Power and beta2Power
            beta1Power *= beta1;
            beta2Power *= beta2;

            // Update first and second moments for the neuron
            thisLayer[j].setFirstMoment(firstMoment);
            thisLayer[j].setSecondMoment(secondMoment);
        }
    }
}