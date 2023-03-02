#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <vector>
#include <iostream>

using namespace std;

// dataframe struct
struct dataset {
    // our labels for each data sample
    vector<int> labels;
    // our data in rows
    vector<vector<double>> data;
    // dataset dimensions
    int rows;
    int cols;

    // default dataset to make c++ happy
    // 1,2,3
    // 4,5,6
    dataset(vector<vector<double>> data = {{1.0,2.0,3.0},{4.0,5.0,6.0}}) {
        this->data = data;
        this->rows = size(data);
        this->cols = size(data[0]);
        int label;
        for (int row=0; row<size(data); row++) {
            // two step pop 
            // grab it
            label = data[row].back();
            // pop it
            data[row].pop_back();
            this->labels.push_back((label));
        }
    };

    void print_labels() {
        for (int i=0; i<size(this->labels); i++) {
            cout << this->labels[i] << "\n";
        }
    };
};

void print_df(vector<vector<double>> dataframe) {
    // let's read out the dataframe
    for (int i=0; i<size(dataframe); i++) {
        for (int j=0; j<size(dataframe[i]);j++) {
            cout << " " << dataframe[i][j] << " ";
        }
        cout << "\n";
    }
}

#endif