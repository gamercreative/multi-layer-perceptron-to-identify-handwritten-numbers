#include <iostream> 
#include <time.h>         // For seeding the random number generator
#include <cmath>
#include <vector>         // To use vectors for dynamic arrays
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>    // For loading images 3rd party library
#include <filesystem>     // To iterate through directories
#include <string>         
#include <chrono>         // For measuring the duration of operations
using namespace std;

// Generate a random weight between -0.05 and 0.05
double rand_weight() {
    return ((double)rand() / RAND_MAX) * 0.1 - 0.05;
}

// Sigmoid activation function for the output layer
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function, used during backpropagation
double sigmoid_derv(double x) {
    return x * (1.0 - x);
}

// ReLU activation function for the hidden layer
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Derivative of the ReLU function, used during backpropagation
double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

// Neural Network class definition
class neural_network {
public:
    // Constructor to initialize the network with given sizes and biases
    neural_network(int input_size, int hidden_size, int output_size, 
                   double hidden_bais_value, double output_bais_value) {
        // Resize weight matrices and bias vectors
        weights_input_hidden.resize(input_size, vector<double>(hidden_size));
        weights_hidden_output.resize(hidden_size, vector<double>(output_size));
        hidden_bais.resize(hidden_size);
        output_bais.resize(output_size);

        // Initialize weights randomly
        for (vector<double>& vec : weights_input_hidden) {
            for (double& weight : vec) {
                weight = rand_weight();
            }
        }
        for (vector<double>& vec : weights_hidden_output) {
            for (double& weight : vec) {
                weight = rand_weight();
            }
        }
        // Set biases to the provided initial values
        fill(hidden_bais.begin(), hidden_bais.end(), hidden_bais_value);
        fill(output_bais.begin(), output_bais.end(), output_bais_value);
    }

    // Forward pass: calculates the output for given input data
    vector<double> ff(const vector<double> data) {
        hidden_layer.clear();
        output_layer.clear();
        hidden_layer.resize(hidden_bais.size());
        output_layer.resize(output_bais.size());

        // Calculate hidden layer activations
        for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
            for (int input_neuron = 0; input_neuron < data.size(); input_neuron++) {
                hidden_layer[hidden_neuron] += data[input_neuron] * 
                                               weights_input_hidden[input_neuron][hidden_neuron];
            }
            hidden_layer[hidden_neuron] = relu(hidden_layer[hidden_neuron] + hidden_bais[hidden_neuron]);
        }

        // Calculate output layer activations
        for (int output_neuron = 0; output_neuron < output_bais.size(); output_neuron++) {
            for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
                output_layer[output_neuron] += hidden_layer[hidden_neuron] * 
                                                weights_hidden_output[hidden_neuron][output_neuron];
            }
            output_layer[output_neuron] = sigmoid(output_layer[output_neuron] + output_bais[output_neuron]);
        }

        return output_layer;
    }

    // Backpropagation: trains the network using one set of input and target data
    void bb(const vector<double> train_data, const vector<double> ans_data) {
        // Initialize layer activations and error vectors
        hidden_layer.clear();
        output_layer.clear();
        hidden_layer.resize(hidden_bais.size());
        output_layer.resize(output_bais.size());

        hidden_layer_error.clear();
        hidden_layer_error_delta.clear();
        hidden_layer_error.resize(hidden_bais.size());
        hidden_layer_error_delta.resize(hidden_bais.size());
        output_error.clear();
        delta_output_error.clear();
        output_error.resize(output_bais.size());
        delta_output_error.resize(output_bais.size());

        // Forward pass: calculate hidden and output layer activations
        for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
            for (int input_neuron = 0; input_neuron < train_data.size(); input_neuron++) {
                hidden_layer[hidden_neuron] += train_data[input_neuron] * 
                                               weights_input_hidden[input_neuron][hidden_neuron];
            }
            hidden_layer[hidden_neuron] = relu(hidden_layer[hidden_neuron] + hidden_bais[hidden_neuron]);
        }

        for (int output_neuron = 0; output_neuron < output_bais.size(); output_neuron++) {
            for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
                output_layer[output_neuron] += hidden_layer[hidden_neuron] * 
                                                weights_hidden_output[hidden_neuron][output_neuron];
            }
            output_layer[output_neuron] = sigmoid(output_layer[output_neuron] + output_bais[output_neuron]);
        }

        // Calculate output layer error
        for (int output_neuron = 0; output_neuron < output_bais.size(); output_neuron++) {
            output_error[output_neuron] = ans_data[output_neuron] - output_layer[output_neuron];
            delta_output_error[output_neuron] = output_error[output_neuron] * 
                                                sigmoid_derv(output_layer[output_neuron]);
        }

        // Calculate hidden layer error
        for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
            hidden_layer_error[hidden_neuron] = 0.0;
            for (int output_neuron = 0; output_neuron < weights_hidden_output[hidden_neuron].size(); output_neuron++) {
                hidden_layer_error[hidden_neuron] += weights_hidden_output[hidden_neuron][output_neuron] * 
                                                     delta_output_error[output_neuron];
            }
            hidden_layer_error_delta[hidden_neuron] = hidden_layer_error[hidden_neuron] * 
                                                      relu_derivative(hidden_layer[hidden_neuron]);
        }

        // Update weights and biases
        for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
            for (int output_neuron = 0; output_neuron < output_bais.size(); output_neuron++) {
                weights_hidden_output[hidden_neuron][output_neuron] += delta_output_error[output_neuron] * 
                                                                       hidden_layer[hidden_neuron] * learning_rate;
            }
        }

        for (int input_neuron = 0; input_neuron < train_data.size(); input_neuron++) {
            for (int hidden_neuron = 0; hidden_neuron < hidden_bais.size(); hidden_neuron++) {
                weights_input_hidden[input_neuron][hidden_neuron] += train_data[input_neuron] * 
                                                                     hidden_layer_error_delta[hidden_neuron] * learning_rate;
            }
        }

        for (int i = 0; i < hidden_bais.size(); i++) {
            hidden_bais[i] += hidden_layer_error_delta[i] * learning_rate;
        }
        for (int i = 0; i < output_bais.size(); i++) {
            output_bais[i] += delta_output_error[i] * learning_rate;
        }
    }

private:
    vector<double> output_layer, hidden_layer;
    vector<double> hidden_layer_error, hidden_layer_error_delta;
    vector<double> output_error, delta_output_error;
    vector<vector<double>> weights_input_hidden, weights_hidden_output;
    vector<double> hidden_bais, output_bais;
    double learning_rate = 0.05;
};

// Train the neural network with one batch of data
void train_once(neural_network& nn, const vector<double>& train_data, const vector<double>& ans_data) {
    nn.bb(train_data, ans_data);
}

int main() {
    srand(time(0));
    neural_network byter3nner(28*28 /* input size*/, 300 /*hidden size*/,10 /*output size*/,0.1 /*hidden bais*/, /*output bais */ 0.1); 
    vector<double> target;
    vector<double> pixels(28*28);
    int width,height,channel;
    vector<pair<vector<double>,vector<double>>> buffer;

    int epoch = 1000;
    for(int i = 0;i < epoch;i++) {
    for(int j = 0;j < 3;j++) { //limited to learning only 0 and 1 for faster operation
        vector<double> tar;
        if(j==0) {
            tar = {1,0,0,0,0,0,0,0,0,0};
        }else if(j == 1) {
            tar = {0,1,0,0,0,0,0,0,0,0};
        }else if(j == 2) {
            tar = {0,0,1,0,0,0,0,0,0,0};
        }
        auto start = std::chrono::high_resolution_clock::now();
            string path = "...\\hand_numbers\\dataset\\" + to_string(j) + "\\" + to_string(j);
            for(const auto fil : filesystem::directory_iterator(path) ) {
                pixels.clear();
                auto ako = fil;
                string path = ako.path().string();
                unsigned char* img = stbi_load(path.c_str(), &width, &height, &channel,0);
                for(int i = 0;i < width*height;i++) {
                    pixels.push_back(img[i*channel+3]/255.0);
                }
                //train_once(ref(byter3nner), move(pixels), move(tar));
                byter3nner.bb(pixels,tar);
                stbi_image_free(img);
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            cout << "\rBackPropagation is " << ((double)i/epoch)*100 << "% done        ";
        }
    }

    cout << "\nstaring test\n";
    vector<double> answer;
    string path = "...\\hand_let\\hand_numbers\\test";
    for(const auto fil : filesystem::directory_iterator(path) ) {
            answer.clear();
            pixels.clear();
            auto ako = fil;
            string path = ako.path().string();
            unsigned char* img = stbi_load(path.c_str(), &width, &height, &channel,0);
            for(int i = 0;i < width*height;i++) {
                pixels.push_back(img[i*channel+3]/255.0);
            }
            answer = byter3nner.ff(pixels);
            for(int i = 0;i < answer.size();i++) {
                cout << i << ":" << answer[i] << " ";
            }
            cout << "\n";
            stbi_image_free(img);
        }
    }