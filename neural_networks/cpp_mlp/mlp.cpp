#include "read_csv.h"
#include "utils.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// MLP run data
const double ETA = 0.1;
const int EPOCHS = 300;
const int SAMPLES = 1000;

void print_stats(MatrixXd A, string name) {
    cout << name << " shape: " << A.rows() << "x" << A.cols() << "\n"; 
    cout << "mean: " << A.mean() << "\n";
}

double reLU(double x) {
    if (x>0) {
        return x;
    } else {
        return 0;
    }
}

double reLU_diff(double x) {
    if (x>0) {
        return 1;
    } else {
        return 0;
    }
}

// forward propagate
void forward_prop(MatrixXd A_0, MatrixXd &A_1, MatrixXd &A_2, 
                  MatrixXd &z_1, MatrixXd &z_2, MatrixXd W_1,
                  VectorXd b_1, MatrixXd W_2, VectorXd b_2) {
    z_1 = W_1*A_0;
    // add bias to each column
    z_1.colwise() += b_1;
    // relu
    A_1 = z_1.unaryExpr(&reLU);
    z_2 = W_2*A_1;
    z_2.colwise() += b_2;
    // TODO: apply softmax
    A_2 = z_2 / z_2.maxCoeff();
}

void back_prop(MatrixXd Y, MatrixXd A_0, MatrixXd A_1, 
               MatrixXd A_2, MatrixXd z_1, MatrixXd &W_1, 
               VectorXd &b_1, MatrixXd &W_2, VectorXd &b_2) {
    // initialise intermediate variables
    MatrixXd dz_2 = MatrixXd::Zero(10,SAMPLES);
    MatrixXd dz_1 = MatrixXd::Zero(10,SAMPLES);
    dz_2 = 2*(A_2 - Y);
    // adjust W2
    W_2 -= ETA * 1/SAMPLES * dz_2 * A_1.transpose();
    // adjust b2
    b_2 -= ETA * 1/SAMPLES * dz_2.rowwise().sum();
    dz_1 = (W_2.transpose() * dz_2).cwiseProduct(z_1.unaryExpr(&reLU_diff));
    // adjust W1
    W_1 -= ETA * 1/SAMPLES * dz_1 * A_0.transpose();
    // adjust b_1
    b_1 -= ETA * 1/SAMPLES * dz_1.rowwise().sum(); 
}

void check_acc(MatrixXd preds, MatrixXd Y) {
    double correct = 0;
    int max_ind = 0;
    double max;
    for (int j=0; j<preds.cols(); j++) {
        auto col_pred = preds.col(j);
        auto col_actual = Y.col(j);
        max = col_pred(0);
        max_ind = 0;
        for (int i=0; i<col_pred.size(); i++) {
            if (col_pred(i) > max) {
                max = col_pred(i);
                max_ind = i;
            }
        }
        if (col_actual(max_ind)==1) {
            correct += 1;
        }
    }
    cout << "Accuracy: " << correct/preds.cols() << "\n";
}

// use minibatch stochastic gradient descent
// faster, gives better results.
void stochastic_gradient_descent(MatrixXd Y, MatrixXd A_0, MatrixXd A_1, 
                      MatrixXd A_2, MatrixXd z_1, MatrixXd z_2, 
                      MatrixXd &W_1, VectorXd &b_1, MatrixXd &W_2, VectorXd &b_2) {
    MatrixXd Y_sample;
    MatrixXd A_0_sample;
    for (int i=0; i<EPOCHS; i++) {
        // randomly sample subset of A_0, Y
        // due to stochastic gradient descent
        // (upper_limit,size)
        auto cols = rand_vector(60000,SAMPLES);
        Y_sample = Y(all, cols);
        A_0_sample = A_0(all, cols);
        forward_prop(A_0_sample, A_1, A_2, z_1, z_2, W_1, b_1, W_2, b_2);
        check_acc(A_2, Y_sample);
        back_prop(Y_sample, A_0_sample, A_1, A_2, z_1, W_1, b_1, W_2, b_2);
    }
}

void one_hot_encode(MatrixXd Y_vec, MatrixXd &Y_1hot) {
    //cout << "one hot encoding.\n";
    int classnum;
    for(int i=0; i<size(Y_vec); i++) {
        classnum = (int)Y_vec(i);
        Y_1hot(classnum,i)=1;
    }
}

int main() {
    // load all data
    MatrixXd Y_train = load_csv<MatrixXd>("./y_train.csv");
    MatrixXd Y_test = load_csv<MatrixXd>("./y_test.csv");
    MatrixXd X_train = load_csv<MatrixXd>("./x_train.csv");
    MatrixXd X_test = load_csv<MatrixXd>("./x_test.csv");
    // let's make one hot encodings for Y_train, Y_test
    MatrixXd Y_train_1hot = MatrixXd::Zero(10,60000);
    MatrixXd Y_test_1hot = MatrixXd::Zero(10,60000);
    one_hot_encode(Y_train, Y_train_1hot);
    one_hot_encode(Y_test, Y_test_1hot);

    // initialise weight matrices and bias vectors
    MatrixXd W_1 = MatrixXd::Random(10,784); 
    MatrixXd W_2 = MatrixXd::Random(10,10);
    VectorXd b_1 = VectorXd::Random(10,1);
    VectorXd b_2 = VectorXd::Random(10,1);
    // initialise intermediate matrix/vectors
    // for forward pass / backward pass
    MatrixXd z_1 = MatrixXd::Random(10,SAMPLES);
    MatrixXd A_1 = MatrixXd::Random(10,SAMPLES);
    MatrixXd z_2 = MatrixXd::Random(10,SAMPLES);
    MatrixXd A_2 = MatrixXd::Random(10,SAMPLES);

    cout << "~~~~~~~ Training ~~~~~~~\n";
    // run gradient descent
    gradient_descent(Y_train_1hot, X_train, A_1, A_2, z_1, z_2, W_1, b_1, W_2, b_2);
    // test accuracy on testing set
    forward_prop(X_test, A_1, A_2, z_1, z_2, W_1, b_1, W_2, b_2);
    cout << "~~~~~~~ Testing ~~~~~~~~~\n";
    check_acc(A_2, Y_test_1hot);
}
