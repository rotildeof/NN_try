#include "NeuralNetwork.h"
#include <iomanip>

#define DBPR

NeuralNetwork::NeuralNetwork(std::string structure_){
  structure = structure_;
  std::string str = structure + ":";
  vec1I neurons;
  std::string store;
  for(int i = 0 ; i < (int)str.size() ; i++){
    if(str[i] != ':'){
      store += str[i];
    }else{
      int num = atoi(store.data());
      neurons.push_back(num);
      store.clear();
    }
  }
  w.assign(neurons.size() - 1, vec2D(0, vec1D() ) );
  b.assign(neurons.size() - 1, vec1D() );
  std::cout << "NN Structure --> " << structure << "  Number of Layers --> " << neurons.size() << std::endl;
  for(int i = 0 ; i < (int)w.size() ; i++){
    w[i].assign(neurons[i+1], vec1D(neurons[i]) );
    b[i].assign(neurons[i+1],0);
    std::cout << i << " : " << "Weight Matrix --> [" << neurons[i+1] << "][" << neurons[i] << "]  Offset Vector --> [" << neurons[i+1] << "]" << std::endl;
  }
  
  nLayerNeurons.assign(neurons.size(), vec1D());
  for(int i = 0 ; i < (int)nLayerNeurons.size() ; i++){
    nLayerNeurons[i].assign(neurons[i], 0);
  }

  std::random_device rd;
  mt = std::mt19937( rd() );

  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  std::cout << "gene length = " << gene_length << std::endl;
  
}


double NeuralNetwork::Sigmoid(double x){
  return 1. / ( 1 + std::exp( -x ) );
}

void NeuralNetwork::CalcuToNextLayerNeurons(int ith_connection){
  for(int i = 0 ; i < (int)w[ith_connection].size() ; i++){
    double sum = 0;
    for(int j = 0 ; j < (int)w[ith_connection][i].size() ; j++){
      sum += nLayerNeurons[ith_connection][j] * w[ith_connection][i][j];
    }
    sum += b[ith_connection][i];
    nLayerNeurons[ith_connection + 1][i] = Sigmoid(sum);
  }
  return;
}

void NeuralNetwork::CalculationAllStageOfLayer(){
  int nConnection = w.size();
  //std::cout << nConnection << std::endl;
  
  for(int ith_connection = 0 ; ith_connection < nConnection ; ith_connection++){
    CalcuToNextLayerNeurons(ith_connection);
  }
}

template <class T>
void NeuralNetwork::InputData(std::vector<T> const &indata){
  if(nLayerNeurons[0].size() != indata.size()){
    std::cout << "-- Error ! The number of size is wrong !! --  " << std::endl;
    return;
  }
  for(int i = 0 ; i < (int)indata.size(); i++){
    nLayerNeurons[0][i] = indata[i];
  }
  return;
}

void NeuralNetwork::ParameterInitialization(){
  std::uniform_real_distribution<> rand_real(lower, upper);
  for(int i = 0 ; i < (int)w.size() ; i++){
    for(int j = 0 ; j < (int)w[i].size() ; j++){
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){
	w[i][j][k] = rand_real(mt);
      }
    }
  }

  for(int i = 0 ; i < (int)b.size() ; i++){
    for(int j = 0 ; j < (int)b[i].size() ; j++){
      b[i][j] = rand_real(mt);
    }
  }
  return;
}

void NeuralNetwork::Learning(vec2D const &inputDataSet, vec2D const &answerDataSet, int nRepetitions){
  long long nEntries = (long long)inputDataSet.size();
  if(nEntries != (int)answerDataSet.size()){
    std::cout << "-- Error !! Number of Entries are different between input data and answer data !! --" << std::endl;
    return;
  }
  
  int last = (int)nLayerNeurons.size() - 1;

  // -- From here, Learning by Genetic Algorithm -- //
  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  
  GA = GeneticAlgorithm<double>(gene_length, population_);
  GA.GeneInitialization(lower, upper);
  
  for(int iLearn = 0 ; iLearn < nRepetitions ; iLearn++){
    if(iLearn % 100 == 0){
      std::cout << "------ " << iLearn << " Times Learning ----" << std::endl;
    }
    for(int iCreature = 0 ; iCreature < GA.GetPopulation() ;  iCreature++){
      SetWeightFromGene(GA, iCreature);
      double error = 0;
      for(int iEntry = 0 ; iEntry < nEntries ; iEntry++){
  	InputData(inputDataSet[iEntry]);	
  	CalculationAllStageOfLayer();
  	error += ErrorEvaluation( nLayerNeurons[last], answerDataSet[iEntry]);
      }// end of Data Entry
#ifdef DBPR      
      if(iLearn % 100 == 0 && iCreature == 0) std::cout << "Error : " << error << std::endl;
#endif
      GA.GiveScore(iCreature, error);
    }// End of looking into every creature
    //ShowInputAndOutput(GA, 0, inputDataSet);
    GA.CrossOver(nDominantGene, mutation_prob, "Minimize");
  }// End of Learning Repetition

  SetWeightFromGene(GA, 0);
}

void NeuralNetwork::Learning(vec2D const &inputDataSet, vec2D const &answerDataSet, double threshold){
  long long nEntries = (long long)inputDataSet.size();
  if(nEntries != (int)answerDataSet.size()){
    std::cout << "-- Error !! Number of Entries are different between input data and answer data !! --" << std::endl;
    return;
  }
  
  int last = (int)nLayerNeurons.size() - 1;

  // -- From here, Learning by Genetic Algorithm -- //
  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  
  GA = GeneticAlgorithm<double>(gene_length, population_);
  GA.GeneInitialization(lower, upper);

  long long iLearn = 0;
  while(1){
    if(iLearn % 100 == 0){
      std::cout << "------ " << iLearn << " Times Learning ----" << std::endl;
    }
    for(int iCreature = 0 ; iCreature < GA.GetPopulation() ;  iCreature++){
      SetWeightFromGene(GA, iCreature);
      double error = 0;
      for(int iEntry = 0 ; iEntry < nEntries ; iEntry++){
  	InputData(inputDataSet[iEntry]);	
  	CalculationAllStageOfLayer();
  	error += ErrorEvaluation( nLayerNeurons[last], answerDataSet[iEntry]);
      }// end of Data Entry
#ifdef DBPR      
      if(iLearn % 100 == 0 && iCreature == 0) std::cout << "Error : " << error << std::endl;
#endif
      GA.GiveScore(iCreature, error);
    }// End of looking into every creature

    GA.CrossOver(nDominantGene, mutation_prob, "Minimize");
    if(GA.GetScore(0) < threshold) break;
    
    iLearn++;
  }// End of Learning Repetition

  SetWeightFromGene(GA, 0);
}

void NeuralNetwork::SetWeightFromGene(GeneticAlgorithm<double> &GA, int ith_creature){
  
  auto it = GA.GetGeneIterator(ith_creature);
  
  for(int i = 0 ; i < (int)w.size(); i++){ // i th connections
    for(int j = 0 ; j < (int)w[i].size() ; j++){ // j th neuron
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){ // k th node
	w[i][j][k] = *it;
	it++;
      }
      b[i][j] = *it;
      it++;
    }
  }
  return;
}

double NeuralNetwork::ErrorEvaluation(vec1D const &lastNeurons, vec1D const &answerData){
  if(lastNeurons.size() != answerData.size()){
    std::cout << "-- Error !! Discrepancy between number of neurons in last layer and answer data -- " << std::endl;
    return -1;
  }
  double acc = 0;
  auto Square = [](double x){return x * x;};
  for(int i = 0 ; i < (int)lastNeurons.size() ; i++){
    acc += Square(lastNeurons[i] - answerData[i]);
  }
  return acc / lastNeurons.size();
}


void NeuralNetwork::ShowInputAndOutput(GeneticAlgorithm<double> &GA, int ith_creature, vec2D const &input){
  SetWeightFromGene(GA, ith_creature);
  for(int i = 0 ; i < (int)input.size() ; i++){
    std::cout << "Data ( ";
    for(auto it = input[i].begin() ; it != input[i].end(); it++){
      if(it + 1 != input[i].end()){
	std::cout << *it  << ", ";
      }else if(it + 1 == input[i].end()){
	std::cout << *it << " ";
      }
    }
    std::cout << ") --> ";
    std::cout << " Output : (" ;
    int last = (int)nLayerNeurons.size() - 1;
    InputData(input[i]);
    CalculationAllStageOfLayer();
    for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
      if(it != nLayerNeurons[last].end() - 1){
	std::cout << *it << ", ";
      }else{
	std::cout << *it << " ";
      }
    }
    std::cout << ")" << std::endl;
  }

}


void NeuralNetwork::PrintWeightMatrix(){
  for(int i = 0 ; i < (int)w.size() ; i++){
    std::cout << "-- Layer " << i << " to " << i+1 << " --" << std::endl;
    for(int j = 0 ; j < (int)w[i].size() ; j++){
      std::cout << "(" ;
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){
	std::cout << std::fixed << std::setprecision(6);
	std::cout << std::right << std::setw(10);
	std::cout << w[i][j][k] << " ";
      }
      std::cout << ")";

      std::cout << " ( " << std::right << std::setw(10)  << b[i][j] << " ) " << std::endl;
      
    }
  }
  return;
}

void NeuralNetwork::PrintLastLayer(vec2D const &inputDataSet){
  for(int i = 0 ; i < (int)inputDataSet.size() ; i++){
    std::cout << "Data ( ";
    for(auto it = inputDataSet[i].begin() ; it != inputDataSet[i].end(); it++){
      if(it + 1 != inputDataSet[i].end()){
	std::cout << *it  << ", ";
      }else if(it + 1 == inputDataSet[i].end()){
	std::cout << *it << " ";
      }
    }
    std::cout << ") --> ";
    std::cout << " Output : ( " ;
    int last = (int)nLayerNeurons.size() - 1;
    InputData(inputDataSet[i]);
    CalculationAllStageOfLayer();
    for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
      if(it != nLayerNeurons[last].end() - 1){
	std::cout << *it << ", ";
      }else{
	std::cout << *it << " ";
      }
    }
    std::cout << ")" << std::endl;
  }  

}

void NeuralNetwork::PrintLastLayer(vec1D const &inputData){
  std::cout << "Data ( ";
  for(auto it = inputData.begin() ; it != inputData.end(); it++){
    if(it + 1 != inputData.end()){
      std::cout << *it  << ", ";
    }else if(it + 1 == inputData.end()){
      std::cout << *it << " ";
    }
  }
  std::cout << ") --> ";
  std::cout << " Output : ( " ;
  int last = (int)nLayerNeurons.size() - 1;
  InputData(inputData);
  CalculationAllStageOfLayer();
  for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
    if(it != nLayerNeurons[last].end() - 1){
      std::cout << *it << ", ";
    }else{
      std::cout << *it << " ";
    }
  }
  std::cout << ")" << std::endl;

}

vec1D::iterator NeuralNetwork::GetOutputIterator(vec1D const &inputData){
  InputData(inputData);
  CalculationAllStageOfLayer();
  std::size_t size = nLayerNeurons.size();
  return nLayerNeurons[size - 1].begin();
}
