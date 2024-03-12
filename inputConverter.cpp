#include <vector>
#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

#define STB_IMAGE_IMPLEMENTATION    // Image handling functions
#include "stb_image.h"

#define MINIMP3_IMPLEMENTATION      // Music handling functions
#include "minimp3.h"
#include "minimp3_ex.h"

//#define _VIDEO_
#ifdef _VIDEO_
#include <opencv2/opencv.hpp>       // Video handling functions
#endif

using std::string;  
using std::vector;
using std::fstream;
using std::ifstream;
using std::ofstream;

namespace fs = std::filesystem;

// Works for: JPEG, PNG, BMP, TGA, and GIF.
// @param route Valid route for a .jpg/.png/.bmp/.tga/.gif file
// @return vector<unsigned> type that is usable for the neural network
template<typename T>
vector<T> convertImage(const string& route)
{
    // Initialize variables for image width, height, and number of channels per pixel
    int width, height, channels;

    // Load the image
    // Use stbi_load() for 8-bit images or stbi_loadf() for HDR images
    unsigned char* img = stbi_load(route.c_str(), &width, &height, &channels, 0);

    // Check if the image was loaded
    if (img == nullptr) 
    {
        throw std::runtime_error("Failed to load image: " + route);
    }

    // Assume the image is grayscale; calculate the size
    size_t imgSize = width * height;
    std::vector<T> imageData;
    imageData.reserve(imgSize);

    // Convert each pixel to grayscale and store it in the vector
    // Note: This simple approach averages the RGB channels to get a grayscale value
    for (int y = 0; y < height; ++y) 
    {
        for (int x = 0; x < width; ++x) 
        {
            int offset = (y * width + x) * channels;
            unsigned grayValue = 0;
            if (channels >= 3) 
            {
                // Simple average for grayscale
                grayValue = static_cast<T>((img[offset] + img[offset + 1] + img[offset + 2]) / 3);
            }
            else 
            {
                // If it's already a single channel, just copy it
                grayValue = static_cast<T>(img[offset]);
            }
            imageData.push_back(grayValue);
        }
    }

    // Free the image memory
    stbi_image_free(img);

    return imageData;
}

// Converting a whole folder of images
// @param route Route to the folder
// @return Returns the array of generated input arrays
template<typename T>
vector<vector<T>> convertImageFolder(const string& route)
{
    // Creating return vector
    std::vector<std::vector<unsigned>> imagesData;

    try 
    {
        // Iterate over all items in the directory
        for (const auto& entry : fs::directory_iterator(route)) 
        {
            // Check if the entry is a file
            if (entry.is_regular_file()) 
            {
                // Load and process the image, then add to imagesData
                std::string imagePath = entry.path().string();
                std::vector<T> imageData = convertImage<T>(imagePath);
                imagesData.push_back(imageData);
            }
        }
    }
    catch (const fs::filesystem_error& e) 
    {
        std::cerr << "Filesystem error: " << e.what() << '\n';
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << '\n';
    }

    return imagesData;
}

// Converts a string array to input and output type
template<typename T>
vector<T> convertStrings(const vector<string>& inputStrings)
{
    // Return vector
    std::vector<unsigned> output;

    // Iterating through every string 
    for (const std::string& str : inputStrings) 
    {
        try 
        {
            // Convert string to unsigned int and add to the output vector
            unsigned long val = std::stoul(str); // stoul converts to unsigned long
            output.push_back(static_cast<T>(val)); // Cast to unsigned if necessary
        }
        catch (const std::exception& e) 
        {
            // Handle the case where the conversion fails
            std::cerr << "Conversion failed for string \"" << str << "\": " << e.what() << '\n';
            // Decide how to handle the error, e.g., skip the item, use a default value, or terminate
        }
    }

    return output;
}

// Converts a .txt file to the substrings separated by the separator
// @param1 route Route to the txt
// @param2 separator Character that separates strings
// @return NN inputable array
template<typename T>
vector<T> convertTxtStrings(const string& route, const char separator)
{
    // Vector for the strings in the file denoted by separator
    std::vector<string> outputStrings;

    // File handling
    std::ifstream file(route);

    // Check if the file is open
    if (file.is_open()) 
    {
        throw std::runtime_error("Could not open file, check path!");
    }

    std::string line;
    std::string word;

    // Reading line by line
    while (std::getline(file, line))
    {
        // Chopping up line by separator character
        for (unsigned i = 0; i < line.size(); i++)
        {
            if (line[i] == separator)
            {
                outputStrings.push_back(word);
                word.clear();
            }
            else
                word.push_back(line[i]);
        }
    }

    // Returning the converted strings
    return convertStrings<T>(outputStrings);
}

// Convert numerical input to unsigneds
// @param input Numerical type input converted to unsigned
// @return Returns unsigned type array
template<typename T1, typename T2>
std::vector<T2> convertNumericVector(const std::vector<T1>& input) 
{
    std::vector<T2> output;

    for (T1 val : input)
        output.push_back(static_cast<T2>(val));

    return output;
}

// Convert boolean inputs
// @param input Input of booelan vectors
// @return Converted boolean array to unsigned type
template<typename T>
std::vector<T> convertBooleanVector(const std::vector<bool>& input) 
{
    const T one = 1;
    const T zero = 0;
    std::vector<unsigned> output;
    for (bool val : input) 
        output.push_back(val ? one : zero);

    return output;
}

// On hot encoding
std::vector<unsigned> oneHotEncode(const std::string& category, const std::vector<std::string>& categories) 
{
    std::vector<unsigned> encoding(categories.size(), 0);

    auto it = std::find(categories.begin(), categories.end(), category);

    if (it != categories.end()) 
        encoding[std::distance(categories.begin(), it)] = 1;

    return encoding;
}

// Time-series option with defined window size for time series 
// @param data Time series data
// @param windowSize Size of window for analysis, this will determine frame lengths
// @return Full batch of frames for the time series data, that is NN comaptible
std::vector<std::vector<float>> createSequences(const std::vector<float>& data, size_t windowSize) 
{
    std::vector<std::vector<float>> sequences;

    for (size_t i = 0; i <= data.size() - windowSize; ++i)
        sequences.push_back(std::vector<float>(data.begin() + i, data.begin() + i + windowSize));

    return sequences;
}

// Converting MP3 to NN readable input
// @param route Filesystem route to the .MP3
// @return NN readable input frame
vector<float> convertMP3(const std::string& route)
{
    // Initialize an mp3 decoder instance
    static mp3dec_t mp3d;
    mp3dec_init(&mp3d);

    // Structure to hold the decoded MP3 file information
    mp3dec_file_info_t info;

    // Load and decode the MP3 file. This function reads the entire file, decodes it,
    // and fills the info structure with decoded samples and metadata (like sample rate).
    // The return value indicates success (0) or failure (non-zero).
    int load_result = mp3dec_load(&mp3d, route.c_str(), &info, NULL, NULL);
    if (load_result != 0) 
    {
        // If loading or decoding failed, print an error message to standard error.
        std::cerr << "Failed to load " << route << std::endl;
        // Return an empty vector if there was an error.
        return {};
    }

    // Create a vector of floats to hold the normalized audio samples.
    // info.samples contains the total number of samples decoded.
    std::vector<float> samples(info.samples);
    for (size_t i = 0; i < info.samples; i++) 
    {
        // Normalize samples from the MP3's native int16_t range (-32768 to 32767)
        // to a floating-point range (-1.0f to 1.0f).
        samples[i] = info.buffer[i] / 32768.0f;
    }

    // Free the buffer allocated by mp3dec_load. This is important to avoid memory leaks.
    free(info.buffer);

    // Returning array
    return samples;
}

#ifdef _VIDEO_
// Converts a video file to frames that NN readable
// @param videoFilePath Path to the video type file
// @return Each frame converted to a float vector
std::vector<std::vector<float>> convertVideoToFrames(const std::string& videoFilePath) 
{
    // Vector to hold the frames as float vectors
    std::vector<std::vector<float>> frameVectors;

    // Open the video file
    cv::VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) 
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return frameVectors;
    }

    cv::Mat frame;
    while (true) 
    {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Convert frame to grayscale
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Convert the grayscale frame to float vector
        std::vector<float> frameVector;
        frameVector.assign((float*)grayFrame.datastart, (float*)grayFrame.dataend);

        // Add the float vector to the list of frame vectors
        frameVectors.push_back(frameVector);
    }

    // When everything done, release the video capture object
    cap.release();

    return frameVectors;
}

#endif