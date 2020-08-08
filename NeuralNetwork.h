#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "GeneticAlgorithm.h"
#include "GeneticAlgorithm.cxx"

#ifndef _NeuralNetwork_
#define _NeuralNetwork_

using vec3D = std::vector<std::vector<std::vector<double> > >;
using vec2D = std::vector<std::vector<double> >;
using vec1D = std::vector<double>;
using vec1I = std::vector<int>;


class NeuralNetwork{
public:
  NeuralNetwork(std::string structure_);
  ~NeuralNetwork(){};


  void SetPopulation(int population){population_ = population;};
  void SetMutationProbability(double prob){mutation_prob = prob;};
  void SetNumOfDominantGene(double nDominantGene_){nDominantGene = nDominantGene_;};
  //[i][j] : [i] --> Data entry, [j] --> ith Neuron in first (last) layer.
  void Learning(vec2D const &inputDataSet,
		vec2D const &answerDataSet,
		int nRepetitions);
  void Learning(vec2D const &inputDataSet,
		vec2D const &answerDataSet,
		double threshold);

  
  void SetRangeOfWeight(double min, double max){lower = min; upper = max;};
  void PrintWeightMatrix();
  void PrintLastLayer(vec2D const &inputDataSet);
  void PrintLastLayer(vec1D const &inputData);
  vec1D::iterator GetOutputIterator(vec1D const &inputData);
  
private:
  std::string structure;
  // 3:3:2
  vec3D w;
  vec2D b;
  vec2D nLayerNeurons;
  
  void CalcuToNextLayerNeurons(int ith_connection);
  void CalculationAllStageOfLayer();
  void ParameterInitialization();
  double lower = -5.; // default
  double upper =  5.; // default
  template <class T> void InputData(std::vector<T> const &indata);
  double Sigmoid(double x);
  double population_ = 100; //default
  double mutation_prob = 0.05; // default
  double nDominantGene = 5; //default
  std::mt19937 mt;

  GeneticAlgorithm<double> GA;
  double ErrorEvaluation(vec1D const &lastNeurons, vec1D const &answerData);
  void SetWeightFromGene(GeneticAlgorithm<double> &GA, int ith_creature);
  void ShowInputAndOutput(GeneticAlgorithm<double> &GA, int ith_creature,
			  vec2D const &input);
};

#endif
