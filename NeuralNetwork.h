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
  
  void SetLossFunction(std::string nameLossFunction_);
  void SetActivationFunction_Hidden(std::string actFuncHidden);
  void SetActivationFunction_Output(std::string actFuncOutput);
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

  int numLastNeurons;
  void CalcuHiddenLayer();
  void CalculationAllStageOfLayer();
  void ParameterInitialization();
  double lower = -5.; // default
  double upper =  5.; // default
  template <class T> void InputData(std::vector<T> const &indata);
  static double Sigmoid(double x);
  static double ReLU(double x);
  static double Identity(double x);
  static void Sigmoid(vec1D &lastLayer, vec1D const &beforeActF);
  static void Softmax(vec1D &lastLayer, vec1D const &beforeActF);
  static void Identity(vec1D &lastLayer, vec1D const &beforeActF);
  double (*hidf_ptr)(double x) = &NeuralNetwork::ReLU;  // "Sigmoid", "ReLU"
  void (*outf_ptr)(vec1D &lL, vec1D const &bAF) = &NeuralNetwork::Sigmoid;
  double population_ = 100; //default
  double mutation_prob = 0.05; // default
  double nDominantGene = 5; //default

  std::mt19937 mt;

  GeneticAlgorithm<double> GA;
  // ---- Loss Function ---- //
  double (*loss_ptr)(vec1D const &lastNeurons, vec1D const &answerData)
  = &NeuralNetwork::MeanSquaredError;
  static double MeanSquaredError(vec1D const &lastNeurons, vec1D const &answerData); // MSE
  static double BinaryCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData); // BCE
  static double CategoricalCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData); //CCE
  // ---------------------- //
  void SetWeightFromGene(GeneticAlgorithm<double> &GA, int ith_creature);
  void ShowInputAndOutput(GeneticAlgorithm<double> &GA, int ith_creature,
			  vec2D const &input);

  double LIMITLESS_1 = 0.9999999999999999;
};

#endif
