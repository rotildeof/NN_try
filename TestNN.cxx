#include <iostream>
#include "NeuralNetwork.h"
#include "NeuralNetwork.cxx"
#include <cmath>
#include <random>

int main(){
  //std::vector<std::vector<double> > input{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  //std::vector<std::vector<double> > answer{{1, 0}, {0, 1}, {0, 1}, {1, 0}};
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> uni(-1, 1);
  int nData = 500;
  std::vector<std::vector<double> > input(nData);
  std::vector<std::vector<double> > answer(nData);
  for(int i = 0 ; i < nData ; i++){
    double x = uni(mt);
    double y = x * x;
    input[i].push_back(x);
    answer[i].push_back(y);
  }
    
  NeuralNetwork NN("1:8:8:8:8:1");
  NN.SetPopulation(100);
  NN.SetRangeOfWeight(-15.0, 15.0);
  NN.SetMutationProbability(0.1);
  NN.SetNumOfDominantGene(5);
  NN.Learning(input, answer, 5000);
  NN.PrintWeightMatrix();
  //NN.PrintLastLayer(input);
  NN.PrintLastLayer(input[0]);
  NN.PrintLastLayer(input[1]);
  NN.PrintLastLayer(input[2]);

  return 0;
}
