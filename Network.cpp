#include "Network.h"

/***************************************/
/******* Initializing functions ********/
/***************************************/

/**
 * Generates a random number using a uniform distribution.
 *
 * @return A random double between 0.0 and 1.0.
 */
double Network::getRandomNumber() 
{
    return dis(gen);
}

/* Function that randomly initializes the weights of the edges
 *
 * Initializes the weights of all edges in the network randomly.
 * Utilizes getRandomNumber() for generating weights, aiming to break symmetry in the learning process.
 */
void Network::randInitEdges() 
{
#pragma omp parallel for collapse(3)
    for (int i = 0; i < layers.size() - 1; i++) 
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
                edges[i][j][k].setWeight(getRandomNumber());
}

/* Function that initializes the activations with random numbers
 *
 * Initializes the biases of all neurons in the network randomly.
 * Utilizes getRandomNumber() to assign a random bias to each neuron, promoting diverse initial conditions.
 */
void Network::randInitNeurons() 
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < layers.size(); i++) 
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
            layers[i].getThisLayer()[j].setBias(getRandomNumber());
}

/* He initialization technique, to set initial weights
 *
 * Initializes the network weights using the Xavier (Glorot) initialization method.
 * This method is tailored for maintaining the variance of activations across layers,
 * making it suitable for networks with sigmoid or tanh activations.
 */
void Network::initializeWeightsHe() 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double std = sqrt(2.0f / layers[0].getNumberOfNeurons()); // for He initialization
    std::normal_distribution<> dis(0, std);

    for (int i = 0; i < layers.size() - 1; i++)
    {
        Neuron* leftLayer = layers[i].getThisLayer();
        Neuron* rightLayer = layers[i + 1].getThisLayer();

        unsigned leftSize = layers[i].getNumberOfNeurons();
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();

        // Iterating through each edge, to set weights
        for (int j = 0; j < leftSize; j++)
            for (int k = 0; k < rightSize; k++)
                edges[i][j][k].setWeight(dis(gen));
        
    }
}

/**
 * Initializes the network weights using the Xavier (Glorot) initialization method.
 * This method is tailored for maintaining the variance of activations across layers,
 * making it suitable for networks with sigmoid or tanh activations.
 */
void Network::initializeWeightsXavier() 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double std = sqrt(2.0f / (layers[0].getNumberOfNeurons() + layers[1].getNumberOfNeurons())); // for Xavier
    std::normal_distribution<> dis(0, std);

    for (int i = 0; i < layers.size() - 1; i++)
    {
        Neuron* leftLayer = layers[i].getThisLayer();
        Neuron* rightLayer = layers[i + 1].getThisLayer();

        unsigned leftSize = layers[i].getNumberOfNeurons();
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();

        // Iterating through each edge, to set weights
        for (int j = 0; j < leftSize; j++)
            for (int k = 0; k < rightSize; k++)
                edges[i][j][k].setWeight(dis(gen));
    }
}

/***************************************/
/****** File handling functions ********/
/***************************************/

/**
 * Saves the current state of the network to an XML file.
 *
 * @param filename Path and name of the file to save the network's configuration.
 * @param network Reference to the network instance to be saved.
 */
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

/**
 * Loads a network configuration from an XML file into the current network instance.
 *
 * @param route Path and name of the XML file containing the network's configuration.
 */
void Network::loadNetworkFromXML(const string& route)
{

}

/***************************************/
/********* Training functions **********/
/***************************************/

/**
 * Calculates the delta activations for the network during backpropagation.
 * This function must be called after computing the initial error in the output layer,
 * propagating this error through the network to update neuron activations accordingly.
 */
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

/**
 * Calculates the adjustments (deltas) needed for the biases of all neurons.
 * Utilizes the computed errors and the derivatives of the activation functions
 * to adjust biases to minimize the network's overall error.
 */
void Network::calculateDeltaBias()
{
    for (unsigned i = 0; i < layers.size(); i++)
    {
        // Current layer's neuron array
        Neuron* thisLayer = layers[i].getThisLayer();

        // Parallelize delta bias calculation for each neuron in the current layer
//#pragma omp parallel for
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            double deltaBias = thisLayer[j].getError() * thisLayer[j].activateDerivative(thisLayer[j].getZ());
            thisLayer[j].setDeltaBias(deltaBias);
        }
    }
}

/**
 * Computes the weight adjustments (deltas) across all edges in the network.
 * Based on the activation of neurons and the computed errors, it updates weights
 * to optimize the network's performance.
 */
void Network::calculateDeltaWeight()
{
    for (unsigned i = 0; i < edges.size(); i++)
    {
        // Number of neurons in the left layer, and left layer's neuron array
        unsigned leftSize = layers[i].getNumberOfNeurons();
        Neuron* leftLayer = layers[i].getThisLayer();

        // Number of neurons in the right layer, and right layer's neuron array
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();
        Neuron* rightLayer = layers[i + 1].getThisLayer();

#pragma parallel for collapse(2)
        for (int j = 0; j < leftSize; j++)
        {
            for (int k = 0; k < rightSize; k++)
            {
                // DeltaWeight =  left activation      *                      d/dActFun(z)                      *   right neuron error      
                double dWeight = leftLayer[j].getActivation() * rightLayer[k].activateDerivative(rightLayer[j].getZ()) * rightLayer[k].getError();

                // Setting delta weight
                edges[i][j][k].setDeltaWeight(dWeight);
            }
        }
    }
}

/**
 * Applies the calculated adjustments to weights and biases.
 * This function updates the network's parameters with the new values computed
 * during backpropagation to gradually reduce error.
 */
void Network::setNewParameters()
{
    // Setting new weights for edges
#pragma omp parallel for collapse(3)
    for (int i = 0; i < edges.size(); i++)
    {
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++)
            {
                // New weight =         old weigth     -      calculated delta weight
                double newWeight = edges[i][j][k].getWeight() - edges[i][j][k].getDeltaWeight();
                
                // Setting new weight
                edges[i][j][k].setWeight(newWeight);
            }
        }
    }

    // Setting new biases for each neuron in each layer
    for (int i = 0; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
#pragma omp parallel for 
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // New bias =       old bias     -  calculated delta bias
            double newBias = thisLayer[j].getBias() - thisLayer[j].getDeltaBias();

            // Setting new bias
            thisLayer[j].setBias(newBias);
        }
    }
}

/***************************************/
/******* Optimization functions ********/
/***************************************/

/* RMSProp, Adagrad, Adadelta, NAG, and Adamax optimizations follow similar pattern :
 * They differ in how they compute the update values based on gradients and apply them to parameters.
 *
 * For RMSProp:
 * E[g^2]_t = 0.9 * E[g^2]_{t-1} + 0.1 * g_t^2
 * theta_t+1 = theta_t - (alpha / sqrt(E[g^2]_t + epsilon)) * g_t
 *
 * For Adagrad:
 * Accumulate squared gradients: G_t = G_{t-1} + g_t^2
 * Update rule: theta_t+1 = theta_t - (alpha / sqrt(G_t + epsilon)) * g_t
 *
 * For Adadelta:
 * Compute running averages of squared gradients and squared updates
 * Parameter update: theta_t+1 = theta_t + Delta_theta_t, where Delta_theta_t is derived from RMS values of gradients and updates
 *
 * For NAG (Nesterov Accelerated Gradient):
 * Momentum term is calculated considering future position of parameters
 * theta_t+1 = theta_t + momentum_term - alpha * grad
 *
 * For Adamax:
 * Variation of Adam that uses infinity norm for scaling gradients
 * v_t = max(beta2 * v_{t-1}, abs(g_t))
 * Update rule is adapted to use v_t in place of the second moment estimate
 */

/*
 * Applies Adam optimization algorithm to update the weights and biases of the network.
 *
 * @param learningRate The step size used for each iteration of the optimization.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param epsilon A small value to prevent division by zero in the implementation.
 *
 * Adam combines the advantages of two other extensions of stochastic gradient descent,
 * specifically Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
 */
void Network::adamOptimization(double learningRate, double beta1, double beta2, double epsilon)
{
    // Initialize Adam parameters
    double beta1Power = 1.0;
    double beta2Power = 1.0;

    // Perform Adam optimization for each neuron's bias
    // Adam Equation:
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    // m_hat_t = m_t / (1 - beta1^t)
    // v_hat_t = v_t / (1 - beta2^t)
    // theta_t+1 = theta_t - learningRate * m_hat_t / (sqrt(v_hat_t) + epsilon)

    // Perform Adam optimization for each neuron's bias
    // Adam combines advantages of AdaGrad and RMSProp, adjusting learning rates based on first and second moments of gradients
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 1; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // Update first and second moments
            // Equation 1: Update first moment (m_t)
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            double deltaBias = thisLayer[j].getDeltaBias();
            double firstMoment = thisLayer[j].getFirstMoment();

            // Equation 2: Update second moment (v_t)
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            double secondMoment = thisLayer[j].getSecondMoment();
            thisLayer[j].setFirstMoment(beta1 * firstMoment + (1 - beta1) * deltaBias);
            thisLayer[j].setSecondMoment(beta2 * secondMoment + (1 - beta2) * deltaBias * deltaBias);

            // Apply bias correction for both moments
            // Equation 3: Bias-corrected first moment estimate (m_hat_t)
            // m_hat_t = m_t / (1 - beta1^t)
            double firstMomentCorrected = thisLayer[j].getFirstMoment() / (1 - beta1Power);
            
            // Equation 4: Bias-corrected second moment estimate (v_hat_t)
            // v_hat_t = v_t / (1 - beta2^t)
            double secondMomentCorrected = thisLayer[j].getSecondMoment() / (1 - beta2Power);

            // Update parameters
            // Equation 5: Update parameters (theta_t+1)
            // theta_t+1 = theta_t - learningRate * m_hat_t / (sqrt(v_hat_t) + epsilon)
            double updatedBias = thisLayer[j].getBias() - learningRate * firstMomentCorrected / (sqrt(secondMomentCorrected) + epsilon);
            thisLayer[j].setBias(updatedBias);
        }
    }

    // Update power factors for beta1 and beta2 for next iteration
    beta1Power *= beta1;
    beta2Power *= beta2;
}

/**
 * Applies RMSProp optimization to update weights and biases of the network.
 * RMSProp adapts the learning rates by dividing by an exponentially decaying average
 * of squared gradients, helping to stabilize the convergence.
 *
 * @param learningRate The initial learning rate.
 * @param decayRate The decay rate for the moving average of the squared gradients.
 * @param epsilon A small constant added to the denominator to improve numerical stability.
 */
void Network::rmspropOptimization(double learningRate, double decayRate, double epsilon) 
{
    // Iterate over each layer and neuron
    // RMSProp Equation:
    // E[g^2]_t = decayRate * E[g^2]_{t-1} + (1 - decayRate) * g_t^2
    // theta_t+1 = theta_t - learningRate * g_t / (sqrt(E[g^2]_t) + epsilon)

    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

#pragma omp parallel for
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute gradients and update moving average of squared gradients
            // Equation 1: E[g^2]_t = decayRate * E[g^2]_{t-1} + (1 - decayRate) * g_t^2
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update moving average of squared gradients
            double squaredGradient = deltaBias * deltaBias;

            // E[g^2]_t = decayRate * E[g^2]_(t-1) + (1 - decayRate) * g_t^2
            double squaredGradientAvg = decayRate * thisLayer[j].getSquaredGradientAvg() + (1 - decayRate) * squaredGradient;
            thisLayer[j].setSquaredGradientAvg(squaredGradientAvg);

            // Equation 2: Update parameters (theta_t+1)
            // theta_t+1 = theta_t - learningRate * g_t / (sqrt(E[g^2]_t) + epsilon)
            double scaleFactor = learningRate / (sqrt(squaredGradientAvg) + epsilon);
            double updatedBias = thisLayer[j].getBias() - scaleFactor * deltaBias;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

/**
 * Utilizes the Adagrad optimization method for updating the network's parameters.
 * Adagrad adapts the learning rate for each parameter, favoring parameters that
 * are infrequently updated. It's particularly effective for dealing with sparse data.
 *
 * @param learningRate The learning rate.
 * @param epsilon A small constant to prevent division by zero.
 */
void Network::adagradOptimization(double learningRate, double epsilon) 
{
    // Iterate over each layer and neuron
    for (unsigned i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

#pragma omp parallel for
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute the gradient for the neuron's bias
            double deltaBias = thisLayer[j].getDeltaBias();

            // Accumulate the squared gradient
            // This accumulation is the key feature of Adagrad, which allows
            // it to adjust the learning rate based on the historical squared gradients.
            double squaredGradientSum = thisLayer[j].getSquaredGradientSum() + deltaBias * deltaBias;
            thisLayer[j].setSquaredGradientSum(squaredGradientSum);

            // Update the bias using the scaled gradient
            // The scaling factor is the learning rate divided by the square root of the
            // accumulated squared gradient plus a small epsilon value to avoid division by zero.
            double scaleFactor = learningRate / (sqrt(squaredGradientSum) + epsilon);
            double updatedBias = thisLayer[j].getBias() - scaleFactor * deltaBias;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

/**
 * Implements the AdaDelta optimization algorithm, an extension of Adagrad that reduces
 * its aggressively decreasing learning rate. Unlike Adagrad, AdaDelta does not require
 * a learning rate and uses the ratio of the moving averages of the gradients to adjust parameters.
 *
 * @param decayRate The decay rate for the moving averages.
 * @param epsilon A small constant added to improve numerical stability.
 */
void Network::adadeltaOptimization(double decayRate, double epsilon)
{
    // AdaDelta does not require a learning rate. It uses the ratio of the moving average of the gradients to the moving average of the parameter updates.
    // AdaDelta Equation:
    // E[g^2]_t = decayRate * E[g^2]_{t-1} + (1 - decayRate) * g_t^2
    // Delta x_t = - sqrt((E[Delta x^2]_{t-1} + epsilon) / (E[g^2]_t + epsilon)) * g_t
    // theta_t+1 = theta_t + Delta x_t
    // E[Delta x^2]_t = decayRate * E[Delta x^2]_{t-1} + (1 - decayRate) * (Delta x_t)^2

    // Iterate over each layer
    for (int i = 0; i < layers.size() - 1; i++) // Ensures we have pairs of layers to work with (current and next)
    {
        // Iterate over each neuron in the current layer and its connections to the next layer
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // Iterate over each connection from neuron j in layer i to all neurons in layer i+1
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++)
            {
                // Retrieve the current edge object, which contains weight, gradient information, etc.
                Edge& currentEdge = edges[i][j][k];

                // Equation 1: Update the exponentially decaying average of past squared gradients
                // This is the first moment estimate (running average of gradients).
                // It's calculated as a decayed average of past squared gradients.
                double squaredGradientAvg = decayRate * currentEdge.getSquaredGradientAvg() + (1 - decayRate) * currentEdge.getDeltaWeight() * currentEdge.getDeltaWeight();
                currentEdge.setSquaredGradientAvg(squaredGradientAvg);

                // Equation 2: Calculate the RMS of the update amount using the RMS of parameter updates and gradients
                // RMSProp scales the gradient by the inverse of the square root of the exponentially decaying average
                // of squared past gradients. AdaDelta adapts this idea by also considering the square root of the 
                // exponentially decaying average of squared parameter updates.
                double RMSUpdateAvg = sqrt(currentEdge.getDeltaWeightAvg() + epsilon); // RMS of parameter updates
                double RMSGradientAvg = sqrt(squaredGradientAvg + epsilon); // RMS of gradients
                double update = -(RMSUpdateAvg / RMSGradientAvg) * currentEdge.getDeltaWeight(); // The update rule

                // Apply the calculated update to the current weight
                double updatedWeight = currentEdge.getWeight() + update;
                currentEdge.setWeight(updatedWeight);

                // Equation 3: Update the exponentially decaying average of squared updates
                // This tracks the second moment estimate but for updates rather than gradients,
                // providing an adaptive learning rate effect based on the history of updates.
                double deltaWeightAvg = decayRate * currentEdge.getDeltaWeightAvg() + (1 - decayRate) * update * update;
                currentEdge.setDeltaWeightAvg(deltaWeightAvg);
            }
        }
    }
}

/**
 * Applies Nesterov Accelerated Gradient (NAG) optimization to update network parameters.
 * NAG is a momentum-based optimization technique that makes a lookahead correction
 * to the gradient, improving the convergence speed compared to classical momentum.
 *
 * @param learningRate The learning rate.
 * @param momentum The momentum coefficient.
 */
void Network::nagOptimization(double learningRate, double momentum) 
{
    // NAG Equation:
    // v_t = momentum * v_{t-1} - learningRate * g_t
    // theta_t+1 = theta_t + momentum * v_t - learningRate * g_{t+1}

    // Iterate over each layer and neuron
    for (int i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();

        // NAG update rule for each neuron's bias
#pragma omp parallel for
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            // Compute the look-ahead gradient
            // The look-ahead is achieved by first moving in the direction of the previous momentum
            // and then calculating the gradient and making a correction.
            double deltaBias = thisLayer[j].getDeltaBias();

            // Update the momentum
            // The momentum is updated similarly to classical momentum, but the key difference
            // is in how it is used to update the parameter.
            double prevMomentum = thisLayer[j].getFirstMoment();
            double newMomentum = momentum * prevMomentum + learningRate * deltaBias;
            thisLayer[j].setFirstMoment(newMomentum);

            // Update the bias using the NAG update rule
            // The update rule uses the new momentum directly for updating the parameter,
            // providing the acceleration effect.
            double updatedBias = thisLayer[j].getBias() - momentum * prevMomentum + (1 + momentum) * newMomentum;
            thisLayer[j].setBias(updatedBias);
        }
    }
}

/**
 * Implements Adamax optimization, a variant of the Adam algorithm that is based on
 * the infinity norm of the gradients rather than the L2 norm. This method is less sensitive
 * to large gradients, making it robust in various conditions, especially with sparse gradients.
 *
 * @param learningRate The learning rate.
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the scaled gradient values.
 * @param epsilon A small constant for numerical stability.
 */
void Network::adamaxOptimization(double learningRate, double beta1, double beta2, double epsilon)
{

    // Adamax is a variant of Adam based on the infinity norm.
    // Adamax Equations are similar to Adam but uses max operation instead of sum for second moment.
    // This makes it suitable for embeddings and sparse data.
    // theta_t+1 = theta_t - learningRate / (1 - beta1^t) * m_t / (max(v_t, epsilon))

    // Initialize beta1Power and beta2Power
    double beta1Power = 1.0;
    double beta2Power = 1.0;

    // Iterate over each layer and neuron
    for (int i = 1; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();

        // Adamax update rule for each neuron's bias
#pragma omp parallel for
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            // Update first moment (mean) similar to Adam
            // Update the second moment using the max operation, not the sum of squares
            // This change is what differentiates Adamax from Adam
            double deltaBias = thisLayer[j].getDeltaBias();

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

/***************************************/
/****** Regularization functions *******/
/***************************************/

/**
 * Applies L1 regularization to update the weights across the network.
 *
 * @param learningRate The learning rate used for the weight update.
 * @param lambda The regularization strength, influencing the penalty for large weights.
 */
void Network::updateWeightsL1(double learningRate, double lambda) 
{
    // Parallelize the outer loop over layers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < edges.size(); i++) 
    {
        // Iterate over each connection
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
            {
                double weight = edges[i][j][k].getWeight();
                double deltaWeight = edges[i][j][k].getDeltaWeight();

                // Apply the L1 penalty
                double penalty = lambda * (weight > 0 ? 1 : -1);

                // Update the weight with L1 regularization
                edges[i][j][k].setWeight(weight - learningRate * deltaWeight - penalty);
            }
        }
    }
}

/**
 * Applies L2 regularization to update the weights across the network.
 *
 * @param learningRate The learning rate used for the weight update.
 * @param lambda The regularization strength, contributing to the penalty on weight sizes.
 */
void Network::updateWeightsL2(double learningRate, double lambda)
{
    // Parallelize the outer loop over layers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < edges.size(); i++) 
    {
        // Iterate over each connection
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
            {
                double weight = edges[i][j][k].getWeight();
                double deltaWeight = edges[i][j][k].getDeltaWeight();

                // Apply the L2 penalty
                double penalty = lambda * weight;

                // Update the weight with L2 regularization
                edges[i][j][k].setWeight(weight - learningRate * (deltaWeight + penalty));
            }
        }
    }
}

/**
 * Applies L1 regularization to update the biases of all neurons across the network.
 * This method encourages sparsity in the model by applying a penalty proportional
 * to the absolute value of the neuron biases, promoting feature selection.
 *
 * @param learningRate The learning rate used for the bias update.
 * @param lambda The regularization strength, controlling the penalty for large biases.
 */
void Network::updateNeuronsL1(double learningRate, double lambda)
{
    // Iterate over each layer and each neuron
    for (int i = 0; i < layers.size(); i++)
    {
        Neuron* neurons = layers[i].getThisLayer();
        unsigned numNeurons = layers[i].getNumberOfNeurons();
#pragma omp parallel for
        for (int j = 0; j < numNeurons; j++)
        {
            // Update bias with L1
            double sign = neurons[j].getBias() > 0 ? 1 : -1;
            neurons[j].setBias(neurons[j].getBias() - learningRate * lambda * sign);
        }
    }
}

/**
 * Applies L2 regularization to update the biases of all neurons across the network.
 * This method discourages large biases by applying a penalty proportional
 * to the square of the bias values, which helps in preventing overfitting.
 *
 * @param learningRate The learning rate used for the bias update.
 * @param lambda The regularization strength, influencing the penalty on bias sizes.
 */
void Network::updateNeuronsL2(double learningRate, double lambda)
{
    // Iterate over each layer and each neuron
    for (int i = 0; i < layers.size(); i++)
    {
        Neuron* neurons = layers[i].getThisLayer();
        unsigned numNeurons = layers[i].getNumberOfNeurons();
#pragma omp parallel for
        for (int j = 0; j < numNeurons; j++)
        {
            // Update bias with L2
            neurons[j].setBias(neurons[j].getBias() - learningRate * lambda * neurons[j].getBias());
        }
    }
}

/**
 * Applies dropout regularization during the network's training process.
 * Randomly sets a subset of neuron activations to zero in each layer, except for the input and output layers,
 * to prevent overfitting by reducing the network's sensitivity to specific weights.
 */
void Network::applyDropoutRegularization()
{
    // Assuming dropoutRate is defined in the Network class
    std::uniform_real_distribution<> dropoutDistribution(0.0, 1.0);

    // Apply dropout to each hidden layer (not the input or output layers)
    for (size_t layerIndex = 1; layerIndex < layers.size() - 1; ++layerIndex) 
    {
        // Retrieve the current layer
        Neuron* neurons = layers[layerIndex].getThisLayer();
        unsigned numNeurons = layers[layerIndex].getNumberOfNeurons();

        #pragma omp parallel for
        for (int neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex) 
        {
            double dropoutDecision = dropoutDistribution(gen);
            if (dropoutDecision < dropoutRate) 
            {
                // Dropout this neuron by setting its activation to zero
                neurons[neuronIndex].setActivation(0.0);
            }
        }
    }
}