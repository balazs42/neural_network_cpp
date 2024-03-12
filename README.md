Neural Network Project
This project is developed by an MSC EE student at the Budapest University of Technology and Economics (BME). It is inspired by 3Blue1Brown's insightful article on backpropagation calculus, which can be found here.

Project Overview
The implementation offers a fresh take on the backpropagation algorithm. The classes used throughout the project draw significant inspiration from the aforementioned article.

Features
The project includes versatile input and output vector generator functions to accommodate a wide range of data types, enhancing the neural network's applicability. Currently supported data types include:

Folders of images (supported formats: .JPG, .PNG, .BMP, .TGA, .GIF)
Single images
Boolean vectors
Vectors of strings
Text data (.TXT) with customizable separators (e.g., "dog,dog,cat,dog,cat...")
Chopped frames from an .MP4 video file (Note: Requires proper inclusion of OpenCV, defining _VIDEO_, and setup of compiler and linker)
Numerical vectors of any type, convertible to other numerical vectors
Time series data with selected window size determining the number of input neurons
Audio files (.MP3)
Basic Usage
cpp
Copy code
vector inputs = convertInput(your_path); // Converts the specified data to neural network compatible inputs
vector expecteds = convertExpecteds(your_path); // Converts specified data to expected outputs
vector layerSizes = {L1, L2, ..., Ln}; // Define the number of neurons in each layer
vector activationFunctions = {"Relu", "Sigmoid", ..., "LeakyRelu"}; // Define activation functions for each layer

// Initialize the network with various parameters
Network network(layerSizes, "OptimizationMethod", "RegularizationMethod", "InitializationMethod", optionalDesiredPrecision, optionalDropoutRate);

// Train the network
network.trainNetwork("TrainingMethod", inputs, outputs, optionalEchoNumber);

// Test the network
unseenInputs = convertInput(your_path_to_other_inputs);
OPTIONAL: unseenExpecteds = convertExpecteds(your_path_to_other_expecteds);
network.testNetwork(unseenInputs, unseenExpecteds);
Example
cpp
Copy code
// Example showing basic network setup and training with artificially created data
int main() {
    try {
        // Define input and expected output sizes
        const unsigned inSize = 4; 
        const unsigned expectedSize = 10;

        vector<unsigned> layerSizes = { inSize, 50, 50, expectedSize };
        Network network(layerSizes, "adam", "none", "none");

        network.randInitNetwork(); // Optionally reinitialize network weights and biases
        vector<string> activationFunctions = {"Sigmoid", "Relu", "Sigmoid", "Tanh"};
        network.setAllActivationFunctions("Relu");
        network.setLayerActivationFunctions(activationFunctions);

        // Example data preparation and network training omitted for brevity
    }
    catch (std::exception& e) {
        // Handle exceptions appropriately
    }
    return 0;
}
Project Status
This project is currently a WORK IN PROGRESS. The core components such as Neuron, Edge, Layer, and Network classes are implemented but require further testing and completion. As this project is a solo endeavor, feedback, bug reports, and suggestions for improvement are highly encouraged. Please feel free to contact me through the GitHub repository.

Note
It is not necessary for the input and expected arrays to be of the same numerical type. The training process is designed to be straightforward, requiring minimal data preprocessing.
