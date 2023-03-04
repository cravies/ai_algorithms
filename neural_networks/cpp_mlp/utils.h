#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

using namespace std;


vector<int> rand_vector(int limit, int size) {
    // First create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_int_distribution<int> dist {0,limit};
    
    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    vector<int> vec(size);
    generate(begin(vec), end(vec), gen);
    
    return vec;
}
