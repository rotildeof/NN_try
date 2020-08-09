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

  numLastNeurons = *(neurons.end() -1);
  
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
  double y = 1. / ( 1 + std::exp( - x ) );
  if(y == 1){
    y = 0.9999999999999999;
  }

  return y;
}

double NeuralNetwork::ReLU(double x){
  if(x >= 0){
    return x;
  }else{
    return 0;
  }
}

double NeuralNetwork::Identity(double x){
  return x;
}

void NeuralNetwork::Sigmoid(vec1D &lastLayer, vec1D const &beforeActF){
  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    lastLayer[i] = 1. / ( 1 + std::exp( - beforeActF[i] ) ) ;
    if(lastLayer[i] == 1){
      lastLayer[i] = 0.9999999999999999;
    }
    if(lastLayer[i] == 0){
      lastLayer[i] = 1e-323;
    }
  }
  return;
}

void NeuralNetwork::Softmax(vec1D &lastLayer, vec1D const &beforeActF){
  auto it_max = std::max_element(beforeActF.begin(), beforeActF.end());
  double denominator = 0;
  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    denominator += std::exp(beforeActF[i] - *it_max);
  }

  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    double numerator = std::exp(beforeActF[i] - *it_max);
    if(numerator == 0) numerator = 1e-323;
    lastLayer[i] = numerator / denominator;
  }
  return;
}

void NeuralNetwork::Identity(vec1D &lastLayer, vec1D const &beforeActF){

  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    lastLayer[i] = beforeActF[i];
  }
  return;
}

void NeuralNetwork::SetActivationFunction_Hidden(std::string actFuncHidden){
  if(actFuncHidden == "Sigmoid"){
    hidf_ptr = &NeuralNetwork::Sigmoid;
  }else if(actFuncHidden == "ReLU"){
    hidf_ptr = &NeuralNetwork::ReLU;
  }else if(actFuncHidden == "Identity"){
    hidf_ptr = &NeuralNetwork::Identity;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::SetActivationFunction_Output(std::string actFuncOutput){
  if(actFuncOutput == "Sigmoid"){
    outf_ptr = &NeuralNetwork::Sigmoid;
  }else if(actFuncOutput == "Softmax"){
    outf_ptr = &NeuralNetwork::Softmax;
  }else if(actFuncOutput == "Identity"){
    outf_ptr = &NeuralNetwork::Identity;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::SetLossFunction(std::string nameLossFunction_){
  if(nameLossFunction_ == "MSE"){
    loss_ptr = &NeuralNetwork::MeanSquaredError;
  }else if(nameLossFunction_ == "BCE"){
    loss_ptr = &NeuralNetwork::BinaryCrossEntropy;
  }else if(nameLossFunction_ == "CCE"){
    loss_ptr = &NeuralNetwork::CategoricalCrossEntropy;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::CalcuHiddenLayer(){
  int nConnection = w.size();
  for(int ith_connection = 0 ; ith_connection < nConnection - 1 ; ith_connection++){
    for(int i = 0 ; i < (int)w[ith_connection].size() ; i++){
      double sum = 0;
      for(int j = 0 ; j < (int)w[ith_connection][i].size() ; j++){
	sum += nLayerNeurons[ith_connection][j] * w[ith_connection][i][j];
      }
      sum += b[ith_connection][i];
      nLayerNeurons[ith_connection + 1][i] = hidf_ptr(sum);
    }
  }
  return;
}

void NeuralNetwork::CalculationAllStageOfLayer(){
  // --- Hidden Layer -- //
  CalcuHiddenLayer();
  // --- Last Layer --//
  int last_con = (int)w.size() - 1;
  vec1D beforeActFunc;
  for(int i = 0 ; i < (int)w[last_con].size() ; i++){ 
    double sum = 0;
    for(int j = 0 ; j < (int)w[last_con][i].size() ; j++){
      sum += nLayerNeurons[last_con][j] * w[last_con][i][j];
    }
    sum += b[last_con][i];
    beforeActFunc.push_back(sum);
    //nLayerNeurons[ith_connection + 1][i] = hidf_ptr(sum);
  }
  outf_ptr(nLayerNeurons[last_con + 1], beforeActFunc);
  
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
	error += loss_ptr( nLayerNeurons[last], answerDataSet[iEntry]) / numLastNeurons;
      }// end of Data Entry
#ifdef DBPR      
      if(iLearn % 100 == 0 && iCreature == 0)
	std::cout << "Error (Average) = " << error / nEntries << std::endl;
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
  	error += loss_ptr( nLayerNeurons[last], answerDataSet[iEntry]) / numLastNeurons;
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

double NeuralNetwork::MeanSquaredError(vec1D const &lastNeurons, vec1D const &answerData){
  if(lastNeurons.size() != answerData.size()){
    std::cout << "-- Error !! Discrepancy between number of neurons in last layer and answer data -- " << std::endl;
    return -1;
  }
  double acc = 0;
  auto Square = [](double x){return x * x;};
  for(int i = 0 ; i < (int)lastNeurons.size() ; i++){
    acc += Square(lastNeurons[i] - answerData[i]);
  }
  return acc;
}


double NeuralNetwork::BinaryCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData){
  if((int)lastNeurons.size() != 1){
    std::cout << "Error ! The number of neurons in Last Layer should be 1 if you use Binary Cross Entropy for loss function." << std::endl;
  }
  if(lastNeurons[0] == 1){
    double bce = - answerData[0] * std::log(0.9999999999999999)
      - (1 - answerData[0]) * std::log(1 -  0.9999999999999999);
    return bce;
  }
  double BCE = - answerData[0] * std::log(lastNeurons[0])
    - (1 - answerData[0]) * std::log(1 - lastNeurons[0]);
  return BCE; 
}

double NeuralNetwork::CategoricalCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData){
  if(lastNeurons.size() != answerData.size()){
    std::cout << "-- Error !! Discrepancy between number of neurons in last layer and answer data -- " << std::endl;
    return -1;
  }
  double acc = 0;
  for(int i = 0 ; i < (int)answerData.size() ; i++){
    if(lastNeurons[i] == 1){
      double bce = - answerData[i] * std::log(0.9999999999999999)
	- (1 - answerData[i]) * std::log(1 -  0.9999999999999999);
      acc += bce;

      continue;
    }
    acc +=
      - answerData[i] * std::log(lastNeurons[i])
      - (1 - answerData[i]) * std::log(1. - lastNeurons[i]) ;

  }
  return acc;
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
