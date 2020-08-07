#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <random>

#ifndef _GeneticAlgorithm_
#define _GeneticAlgorithm_

// You should use <int> or <double>.

template <typename T>
class GeneticAlgorithm{
public:
  GeneticAlgorithm(int gene_length, int population);
  ~GeneticAlgorithm(){};

  
  void GiveScore(int ith_creature, double score);
  void GeneInitialization(T min, T max);

  typename std::vector<T>::iterator GetGeneIterator(int ith_creature);

  void CrossOver(int numDominantGene, double mutation_prob, std::string optimization_option);

  
private:
  T min_;
  T max_;
  
  std::mt19937 mt;
  T type_keeper;
  struct Creature{
    std::vector<T> gene;
    double Score;
  };

  std::vector<Creature> creatures;  

  void Mutation(std::vector<T> &v);

};

#endif
