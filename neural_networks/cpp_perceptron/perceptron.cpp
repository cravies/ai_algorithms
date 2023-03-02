#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include "dataset.h"

using namespace std;

// single layer perceptron class
class perceptron {
    dataset df;
    dataset test_df;
    vector<double> weights;
    double bias;
    double eta;

    public: 
        // constructor
        perceptron(vector<vector<double>> dataframe, double eta) {
            this->eta = eta;
            this->df = dataset(dataframe);
            // initialise weights to 1.0.. 
            // should have same number of weights and features
            for (int i=0; i<df.cols; i++) {
                this->weights.push_back(1.0);
            }
            // initialise bias
            this->bias = 0;
        }

        // take the dot product of two equally long vectors
        double dot_product(vector<double> a, vector<double> b) {
            //cout << "a size: " << size(a) << " b size: " << size(b) << "\n";
            double res=0;
            assert(size(a)==size(b));
            for (int i=0; i<size(a); i++) {
                res += a[i]*b[i];
            }
            return res;
        }

        // train for #epoch epochs
        void train(int epochs) {
            int class_guess;
            double sum;
            float correct;
            for (int i=0; i<epochs; i++) {
                correct = 0;
                // get output for each example
                for (int j=0; j<df.rows; j++) {
                    sum = dot_product(this->weights,df.data[j]) + this->bias;
                    class_guess = threshold(sum);
                    if (update_weights(j, class_guess)) {
                        correct += 1;
                    }
                }
                // get correct percentage 
                cout << "train acc for epoch " << i << " is " << (correct / df.rows) << "\n";
            }
        }

        void test (dataset test) {
            int class_guess; 
            double sum;
            float correct=0;
            // make test dataset object
            this->test_df = dataset(test);
            for (int i=0; i<test.rows; i++) {
                sum = dot_product(this->weights,this->test_df.data[i]);
                class_guess = threshold(sum);
                if (class_guess==this->test_df.labels[i]) {
                    correct += 1;
                }
            }
            // get correct percentage 
            cout << "test acc is " << (correct / test.rows) << "\n";
        }

        // update the perceptron weights
        // given example index and weights
        bool update_weights(int example_ind, int class_guess) {
            int label;
            label = this->df.labels[example_ind];
            //cout << "guess " << class_guess << " was really " << label << "\n";
            if (label==0 && label!=class_guess) {
                // guessed 1, should be 0, decrement
                // weights -= eta * example
                for (int i=0; i<df.cols; i++) {
                    this->weights[i] -= this->eta * this->df.data[example_ind][i];
                }
                return false; // guessed wrong
            } else if (label==1 && label!=class_guess) {
                // guessed 0, should be 1, increment
                // weights += eta * example
                for (int i=0; i<df.cols; i++) {
                    this->weights[i] += this->eta * this->df.data[example_ind][i];
                }
                return false; // guessed wrong
            }
            return true; // guessed right
        }

        // threshold activation function
        int threshold(double activation) {
            if (activation > 0) {
                return 1;
            } else {
                return 0;
            }
        }
};

int main()
{
    // read train csv into dataframe
    vector<vector<double>> dataframe_train;
    vector<vector<double>> dataframe_test;
    ifstream  data("./Datasets/ionosphere.data");
    string line;
    int line_count;
    while(getline(data,line))
    {
        stringstream lineStream(line);
        string cell;
        vector<double> row;
        // special class encoding for ionosphere g=good (1) b=bad (0)
        while(getline(lineStream,cell,',')) {
            if (cell=="g") {
                cell="1";
            } else if (cell=="b") {
                cell="0";
            }
            row.push_back(stod(cell));
        }
        // chuck the first 30 lines into a test dataframe..
        // hacky solution but will be ok for now
        // assumes the file has been shuffled
        if (line_count < 30) {
            dataframe_test.push_back(row);
        } else {
            dataframe_train.push_back(row);
        }
        line_count+=1;
    }

    // make a new perceptron with out dataframe and run it
    // learning rate eta=0.01
    perceptron p(dataframe_train, 0.01);
    // train 50 epoch
    p.train(100);
    // test
    cout << "made it this far\n";
    p.test(dataframe_test);
}
