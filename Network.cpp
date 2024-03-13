#include "Network.h"

// Operator overload for network addition
// This will create a new instance of neural network, where
// if output-input sizes are the same they will be linked directly
// if not then a new layer of egdes will be inserted
// @rhs Right hand side network
// @return The new concated network
Network Network::operator+(const Network& rhs) const 
{
    // Check if the last layer of LHS and the first layer of RHS have the same number of neurons
    const size_t lhsNeuronCount = this->getLayers().back().getNumberOfNeurons();
    const size_t rhsNeuronCount = rhs.getLayers().front().getNumberOfNeurons();

    vector<unsigned> layerSizes;
    vector<string> actFunctions;

    // Iterating through left layer's neuron array sizes
    for (unsigned i = 0; i < this->getLayers().size(); i++)
    {
        layerSizes.push_back(this->getLayers()[i].getNumberOfNeurons());
        actFunctions.push_back(this->getLayers()[i].getThisLayer()[0].getActivationFunctionString());
    }

    // Iterating through right layer's neuron array sizes
    for (unsigned i = 0; i < rhs.getLayers().size(); i++)
    {
        layerSizes.push_back(rhs.getLayers()[i].getNumberOfNeurons());
        actFunctions.push_back(rhs.getLayers()[i].getThisLayer()[0].getActivationFunctionString());
    }

     vector<Edge**> newEdges;
     // Copiing left hand's edges
     for (unsigned i = 0; i < this->getEdges().size(); i++)
          newEdges.push_back(this->getEdges()[i]);
    
     // Checking if new edges should be inserted
     if (lhsNeuronCount == rhsNeuronCount)
     {
         Edge** wiringEdges = new Edge*[lhsNeuronCount];
         for (unsigned j = 0; j < lhsNeuronCount; j++)
             wiringEdges[j] = new Edge[rhsNeuronCount];
     }

     // Copiing right hand side's edges
     for (unsigned i = 0; i < rhs.getEdges().size(); i++)
         newEdges.push_back(rhs.getEdges()[i]);

     vector<Layer> newLayers;

     // Copiing layers of left hand side
     for (unsigned i = 0; i < this->getLayers().size(); i++)
         newLayers.push_back(this->getLayers()[i]);

     // Copiing layers of right hand side
     for (unsigned i = 0; i < rhs.getLayers().size(); i++)
         newLayers.push_back(rhs.getLayers()[i]);

     // Creating return instance
     Network newNetwork(newEdges, newLayers);

     // Setting activation functions
     newNetwork.setLayerActivationFunctions(actFunctions);

    return newNetwork;
}

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

#pragma omp parallel for collapse(3) schedule(dynamic)
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

#pragma omp parallel for collapse(3) schedule(dynamic)
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
 * Prints the whole network structure to the standard output
 */
void Network::printNetwork()
{
    std::cout << "\n Neurons\n";
    // Printing neurons
    for (unsigned i = 0; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (unsigned j = 0; j < layers[i].getNumberOfNeurons(); j++)
            std::cout << thisLayer[j].getBias() << " ";
        std::cout << "\n";
    }

    std::cout << "\nEdges\n";
    // Printing edges
    for (unsigned i = 0; i < layers.size() - 1; i++)
    {
        unsigned leftSize = layers[i].getNumberOfNeurons();
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();
        for (unsigned j = 0; j < leftSize; j++)
        {
            for (unsigned k = 0; k < rightSize; k++)
                std::cout << edges[i][j][k].getWeight() << " ";
        }
        std::cout << "\n";
    }
}

/**
 * Saves the current state of the network to an XML file.
 *
 * @param filename Path and name of the file to save the network's configuration.
 * @param network Reference to the network instance to be saved.
 */
void Network::saveNetworkToXML(const string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error: Unable to open file for writing: " << filename << "\n";
        return;
    }

    // Write XML header
    outFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    outFile << "<Network>\n";

    // Print basic network information
    outFile << "\t<parameters>\n";
    outFile << "\t\t<layers>\n";

    // Printing layer information
    for (unsigned i = 0; i < layers.size(); i++)
    {
        outFile << "\t\t\t<layer" << i << ">" << layers[i].getNumberOfNeurons() << "</layer" << i << ">\n";                                                         // Printing number of neurons
        outFile << "\t\t\t<activationFunction" << i << ">" << layers[i].getThisLayer()[0].getActivationFunctionString() << "</activationFunction" << i << ">\n";    // Printing layer activation functions
    }

    outFile << "\t\t</layers>\n";
    outFile << "\t</parameters>\n";

    // Printing baises
    outFile << "\t<biases>\n";
    for (unsigned i = 0; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned layerSize = layers[i].getNumberOfNeurons();

        for (unsigned j = 0; j < layerSize; j++)
            outFile << "\t\t<bias" << j << ">" << thisLayer[j].getBias() << "</bias" << j << ">\n";
    }
    outFile << "\t</biases>\n";

    outFile << "\t<edges>\n";
    // Printing edge weights
    for (unsigned i = 0; i < layers.size() - 1; i++)
    {
        unsigned leftSize = layers[i].getNumberOfNeurons();
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();

        // Printing edges
        for (unsigned j = 0; j < leftSize; j++)
            for (unsigned k = 0; k < rightSize; k++)
                outFile << "\t\t\<edge" << j << "," << k << ">" << edges[i][j][k].getWeight() << "</edge" << j << "," << k << ">\n";
    }
    outFile << "\t</edges>\n";

    // Close XML tag
    outFile << "</Network>" << "\n";

    outFile.close(); // Close the file
}

// Fetching numerical value in string from an XML like structure
// in a style of: ....>NUMBERsubstring....
// @param line Line where number is being searched
// @param subString The substring which is exactly behind the number
// @return String of the number or "null" string if nout found any numbers
string fetchString(const string& line, const string& subString)
{
    size_t pos = line.find(subString);
    if (pos != std::string::npos) 
    {
        size_t start = line.rfind('>', pos) + 1; // Find the last '>' before our substring
        return line.substr(start, pos - start);
    }
    return "null";
}

// Skips the lines in a file until the specified substring is found
// @param stream Stream to skip lines on
// @param until The substring the is searched in the line
void skipLineUntil(std::ifstream& stream, const string& until)
{
    string line;
    while (std::getline(stream, line))
    {
        auto iter = line.find(until);
        if (iter != std::string::npos)return;
    }
}

/**
 * Loads a network configuration from an XML file into the current network instance.
 *
 * @param route Path and name of the XML file containing the network's configuration.
 */
Network Network::loadNetworkFromXML(const string& route)
{
    std::ifstream networkFile(route);

    if (!networkFile.is_open())
    {
        std::cerr << "Error: Unable to open file for reading: " << route << "\n";
        std::cerr << "Returning empty network object\n";
        return Network();
    }

    string line;

    // Find where data starts
    skipLineUntil(networkFile, "<Network>");

    vector<string> activationFunctions;
    vector<unsigned> layerSizes;

    // Getting layer information
    skipLineUntil(networkFile, "<layers>");

    while (std::getline(networkFile, line))
    {
        auto iter = line.find("</layers>");
        if (iter != std::string::npos)
            break;
       
        // If the read line is number of neurons information
        string fetched = fetchString(line, "</layer");

        if (fetched != "null")
        {
            layerSizes.push_back(stol(fetched));
            continue;
        }

        // If the read line is an activation funciton
        fetched = fetchString(line, "</activationFunction");
        
        if (fetched != "null")
        {
            activationFunctions.push_back(fetched);
            continue;
        }
    }

    // Creating network constructor objects variables
    // Creating layers, only allocating memory, no initialization or loading
    vector<Layer> loadedLayers;
    for (unsigned i = 0; i < layerSizes.size(); i++)
        loadedLayers.push_back(Layer(layerSizes[i]));

    // Creating edges, only allocating memory no initialization or laoding 
    vector<Edge**> loadedEdges;
    for (unsigned i = 0; i < layerSizes.size() - 1; i++)
    {
        unsigned leftSize = loadedLayers[i].getNumberOfNeurons();
        unsigned rightSize = loadedLayers[i + 1].getNumberOfNeurons();
        
        Edge** edges = new Edge * [leftSize];
        for (unsigned j = 0; j < leftSize; j++)
            edges[j] = new Edge[rightSize];

        loadedEdges.push_back(edges);
    }

    vector<double> biases;
    vector<double> weights;

    // Going to bias data
    skipLineUntil(networkFile, "<biases>");

    // Reading each line for biases
    while (std::getline(networkFile, line))
    {
        auto iter = line.find("</biases>");     // Reading until the end of biases
        if (iter != std::string::npos)break;

        string fetched = fetchString(line, "</bias");
        if (fetched != "null")
        {
            biases.push_back(std::stod(fetched));
            continue;
        }
    }

    // Going to edge data
    skipLineUntil(networkFile, "<edges>");

    // Reading each edges data
    while (std::getline(networkFile, line))
    {
        // Escape condition
        auto iter = line.find("</edges>");
        if (iter != std::string::npos)break;

        string fetched = fetchString(line, "</edge");
        if (fetched != "null")
        {
            weights.push_back(std::stod(fetched));
            continue;
        }
    }

    // Closing file to start loading values
    networkFile.close();
    unsigned biasCounter = 0;

    // Loading biases into the created network object
    for (unsigned i = 0; i < loadedLayers.size(); i++)
    {
        Neuron* thisLayer = loadedLayers[i].getThisLayer();
        unsigned layerSize = loadedLayers[i].getNumberOfNeurons();

        for (unsigned j = 0; j < layerSize; j++)
        {
            // Setting bias of each neuron to be the loaded one
            thisLayer[j].setBias(biases[biasCounter]);
            biasCounter++;
        }
    }

    unsigned weightCounter = 0;

    // Loading weights into the created network object
    for (unsigned i = 0; i < loadedLayers.size() - 1; i++)
    {
        unsigned leftSize = loadedLayers[i].getNumberOfNeurons();
        unsigned rightSize = loadedLayers[i + 1].getNumberOfNeurons();

        for (unsigned j = 0; j < leftSize; j++)
        {
            for (unsigned k = 0; k < rightSize; k++)
            {
                loadedEdges[i][j][k].setWeight(weights[weightCounter]);
                weightCounter++;
            }
        }
    }

    // Creating return instance
    Network newNetwork(edges, layers);

    // Setting activation functions
    newNetwork.setLayerActivationFunctions(activationFunctions);

    return newNetwork;
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
    for (int i = edges.size() - 1; i >= 0; i--)
    {
        Neuron* rightLayer = layers[i + 1].getThisLayer();
        Neuron* leftLayer = layers[i].getThisLayer();

        unsigned rightSize = layers[i + 1].getNumberOfNeurons();
        unsigned leftSize = layers[i].getNumberOfNeurons();

        // Iteratin through left layers neurons
#pragma omp parallel for collapse(2) reduction(+:error)
        for (int j = 0; j < leftSize; j++)
        {
            error = 0.0f;
            // Iterating through right layer's neurons
            for (int k = 0; k < rightSize; k++)
            {
                // Error = edge between right and left layer's neurons * righ layer's neuron's error
                error += edges[i][j][k].getWeight() * rightLayer[k].getError();
            }
            // * d/dActFun(z) 
            error *= leftLayer[j].derivativeActivationFunction(leftLayer[j].getZ());
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
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < layers.size(); i++)
    {
        // Current layer's neuron array
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned thisSize = layers[i].getNumberOfNeurons();

        // Parallelize delta bias calculation for each neuron in the current layer
        for (int j = 0; j < thisSize; j++)
        {
            // deltaBias     = error in current neuron * d/dActFun(z)
            //double deltaBias = thisLayer[j].getError() * thisLayer[j].activateDerivative(thisLayer[j].getZ());
            thisLayer[j].setDeltaBias(thisLayer[j].getError());
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
    double learningRate = 0.01f;

#pragma parallel for collapse(3) schdeule(dynamic)
    for (int i = 0; i < edges.size(); i++)
    {
        // Number of neurons in the left layer, and left layer's neuron array
        unsigned leftSize = layers[i].getNumberOfNeurons();
        Neuron* leftLayer = layers[i].getThisLayer();

        // Number of neurons in the right layer, and right layer's neuron array
        unsigned rightSize = layers[i + 1].getNumberOfNeurons();
        Neuron* rightLayer = layers[i + 1].getThisLayer();

        for (int j = 0; j < leftSize; j++)
        {
            for (int k = 0; k < rightSize; k++)
            {
                // DeltaWeight =  left activation      *                      d/dActFun(z)                      *   right neuron error      
                //double dWeight = leftLayer[j].getActivation() * rightLayer[k].activateDerivative(rightLayer[j].getZ()) * rightLayer[k].getError();
                double dWeight = learningRate * leftLayer[j].getActivation() * rightLayer[k].getError();
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
void Network::setNewParameters(const double learningRate)
{
    // If certain regularization technique is selected, then 
    // updating weights with respect to the regularization
    // technique
    if (useRegularization()) {}
         
    // If a certain optimization method is selected, then updating 
    // biases with the given optimization technique, if none is selected
    // then applying basic gradient descent calculated baises
    if (useOptimization())
        return;
    else
    {
        // Setting new biases for each neuron in each layer
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < layers.size(); i++)
        {
            Neuron* thisLayer = layers[i].getThisLayer();
            unsigned layerSize = layers[i].getNumberOfNeurons();
            for (int j = 0; j < layerSize; j++)
            {
                // New bias =       old bias     -  calculated delta bias
                double newBias = thisLayer[j].getBias() - thisLayer[j].getDeltaBias();

                // Setting new bias
                thisLayer[j].setBias(newBias);
            }
        }

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
    double beta1pow = this->getBeta1Power();
    double beta2pow = this->getBeta2Power();

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
        unsigned layerSize = layers[i].getNumberOfNeurons();

        for (int j = 0; j < layerSize; j++)
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
            double firstMomentCorrected = thisLayer[j].getFirstMoment() / (1.0f - beta1pow);
            
            // Equation 4: Bias-corrected second moment estimate (v_hat_t)
            // v_hat_t = v_t / (1 - beta2^t)
            double secondMomentCorrected = thisLayer[j].getSecondMoment() / (1.0f - beta2pow);

            // Update parameters
            // Equation 5: Update parameters (theta_t+1)
            // theta_t+1 = theta_t - learningRate * m_hat_t / (sqrt(v_hat_t) + epsilon)
            double updatedBias = thisLayer[j].getBias() - learningRate * firstMomentCorrected / (sqrt(secondMomentCorrected) + epsilon);
            thisLayer[j].setBias(updatedBias);
        }
    }

    // Updating weights
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < edges.size(); i++) 
    {
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
            {
                Edge& edge = edges[i][j][k];
                double grad = edge.getDeltaWeight();

                // Update moments
                double m = beta1 * edge.getFirstMoment() + (1 - beta1) * grad;
                double v = beta2 * edge.getSecondMoment() + (1 - beta2) * grad * grad;
                edge.setFirstMoment(m);
                edge.setSecondMoment(v);

                // Bias correction
                double mCorrected = m / (1 - beta1Power);
                double vCorrected = v / (1 - beta2Power);

                // Apply updates
                double weightUpdate = -learningRate * mCorrected / (sqrt(vCorrected) + epsilon);
                edge.setWeight(edge.getWeight() + weightUpdate);
            }
        }
    }

    // Update power factors for beta1 and beta2 for next iteration
    this->setBeta1Power(this->getBeta1Power() * beta1);
    this->setBeta1Power(this->getBeta1Power() * beta1);
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

    // Updating biases

    // Iterate over each layer and neuron
#pragma omp parallel for collapse(2)
    for (int i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned layerSize = layers[i].getNumberOfNeurons();

        for (int j = 0; j < layerSize; j++) 
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

    // Updating edges
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < edges.size(); i++) 
    {
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
            {
                Edge& edge = edges[i][j][k];
                double grad = edge.getDeltaWeight(); 
                double squaredGradientAvg = decayRate * edge.getSquaredGradientAvg() + (1 - decayRate) * grad * grad;
                edge.setSquaredGradientAvg(squaredGradientAvg);
                double weightUpdate = -learningRate * grad / (sqrt(squaredGradientAvg) + epsilon);
                edge.setWeight(edge.getWeight() + weightUpdate);
            }
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
    // Updating biases
   
    // Iterate over each layer and neuron
#pragma omp parallel for collapse(2) schedule (dynamic)
    for (int i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned layerSize = layers[i].getNumberOfNeurons();

        for (int j = 0; j < layerSize; j++) 
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

    // Updating weights
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < edges.size(); i++)
    {
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++)
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++)
            {
                Edge& edge = edges[i][j][k];
                double grad = edge.getDeltaWeight(); // Assuming this stores the gradient for the weight.

                // Update the accumulated squared gradient
                double accumulatedSquaredGradient = edge.getSquaredGradientSum() + grad * grad;
                edge.setSquaredGradientSum(accumulatedSquaredGradient);

                // Calculate the adjusted learning rate
                double adjustedLearningRate = learningRate / (sqrt(accumulatedSquaredGradient) + epsilon);

                // Update the weight
                double newWeight = edge.getWeight() - adjustedLearningRate * grad;
                edge.setWeight(newWeight);
            }
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

    // Update biases
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            Neuron& neuron = thisLayer[j];

            double gradBias = neuron.getError() * neuron.derivativeActivationFunction(neuron.getZ()); 
            double squaredGrad = gradBias * gradBias;

            // Update running averages
            neuron.setSquaredGradientAvg(decayRate * neuron.getSquaredGradientAvg() + (1 - decayRate) * squaredGrad);

            double rmsGradBias = sqrt(neuron.getSquaredGradientAvg() + epsilon);
            double rmsUpdateBias = sqrt(neuron.getDeltaBiasAvg() + epsilon);

            // Compute and apply bias update
            double updateBias = (rmsUpdateBias / rmsGradBias) * gradBias;
            neuron.setBias(neuron.getBias() - updateBias);

            // Update deltaBiasAvg
            neuron.setDeltaBiasAvg(decayRate * neuron.getDeltaBiasAvg() + (1 - decayRate) * updateBias * updateBias);
        }
    }

    // Update weights
    
    // Iterate over each layer
#pragma omp parallel for collapse(3) schedule(dynamic)
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
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 1; i < layers.size(); i++) 
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned layerSize = layers[i].getNumberOfNeurons();

        // NAG update rule for each neuron's bias
        for (int j = 0; j < layerSize; j++) 
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
    double beta1pow = this->getBeta1Power();

    // Adamax is a variant of Adam based on the infinity norm.
    // Adamax Equations are similar to Adam but uses max operation instead of sum for second moment.
    // This makes it suitable for embeddings and sparse data.
    // theta_t+1 = theta_t - learningRate / (1 - beta1^t) * m_t / (max(v_t, epsilon))
    
    // Setting biases

    // Iterate over each layer and neuron
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 1; i < layers.size(); i++)
    {
        Neuron* thisLayer = layers[i].getThisLayer();
        unsigned layerSize = layers[i].getNumberOfNeurons();

        // Adamax update rule for each neuron's bias
        for (int j = 0; j < layerSize; j++)
        {
            // Update first moment (mean) similar to Adam
            // Update the second moment using the max operation, not the sum of squares
            // This change is what differentiates Adamax from Adam
            double deltaBias = thisLayer[j].getDeltaBias();

            double firstMoment = beta1 * thisLayer[j].getFirstMoment() + (1.0f - beta1) * deltaBias;
            double secondMoment = fmax(beta2 * thisLayer[j].getSecondMoment(), fabs(deltaBias));

            // Update first and second moments for the neuron
            thisLayer[j].setFirstMoment(firstMoment);
            thisLayer[j].setSecondMoment(secondMoment);

            // Update bias using the learning rate, first moment, second moment, and epsilon
            // Bias-corrected first moment is used, but not the second moment, as per Adamax formula
            double mCorrected = firstMoment / (1.0 - beta1pow);
            double updatedBias = thisLayer[j].getBias() - (learningRate / (1.0f - beta1pow)) * (mCorrected / (std::max(secondMoment, epsilon)));
            thisLayer[j].setBias(updatedBias);
        }
    }

    // Setting weights

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < edges.size(); i++) 
    {
        for (int j = 0; j < layers[i].getNumberOfNeurons(); j++) 
        {
            for (int k = 0; k < layers[i + 1].getNumberOfNeurons(); k++) 
            {
                Edge& edge = edges[i][j][k];

                // Update first moment (m) and \(\infty\)-norm (u)
                double grad = edge.getDeltaWeight();
                edge.setFirstMoment(beta1 * edge.getFirstMoment() + (1.0 - beta1) * grad);
                edge.setUNorm(std::max(beta2 * edge.getUNorm(), std::fabs(grad)));

                // Compute bias-corrected first moment estimate
                double mHat = edge.getFirstMoment() / (1.0 - std::pow(beta1, epoch + 1));

                // Calculate and apply the update
                double denom = edge.getUNorm() + epsilon;
                double update = learningRate / denom * mHat;
                edge.setWeight(edge.getWeight() - update);
            }
        }
    }

    // Update beta1Power for the next epoch
    // This should be done once per epoch, so if this function is called multiple times within an epoch,
    // ensure that beta1Power is only updated at the end of the epoch.
    this->setBeta1Power(this->getBeta1Power() * beta1);
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
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < layers.size(); i++)
    {
        Neuron* neurons = layers[i].getThisLayer();
        unsigned numNeurons = layers[i].getNumberOfNeurons();
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
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < layers.size(); i++)
    {
        Neuron* neurons = layers[i].getThisLayer();
        unsigned numNeurons = layers[i].getNumberOfNeurons();
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
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 1; i < layers.size() - 1; i++) 
    {
        // Retrieve the current layer
        Neuron* neurons = layers[i].getThisLayer();
        unsigned numNeurons = layers[i].getNumberOfNeurons();

        for (int j = 0; j < numNeurons; j++) 
        {
            double dropoutDecision = dropoutDistribution(gen);
            if (dropoutDecision < dropoutRate) 
            {
                // Dropout this neuron by setting its activation to zero
                neurons[j].setActivation(0.0);
            }
        }
    }
}