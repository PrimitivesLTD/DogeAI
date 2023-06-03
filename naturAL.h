#ifndef naturAL_H // NaturAL layers header
#define naturAL_H // Set of functions for training NaturAL in pure C

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define Tensor1D double*
#define Tensor2D double**

typedef struct naturAL naturAL;

struct percept {
	int _index;
	double _nettoInput;
	double _activation;
	double _output;
	void (*activationFunction) (struct percept* percept); 
	void (*outputFunction) (struct percept* percept);
	double (*derivative)(naturAL* network, int i, int j);
};

typedef struct percept percept;

percept* perceptInit(int index, double nettoInput, void (*actf) (percept* percept), void (*outf) (percept* percept)) {
	percept* Percept = static_cast<percept*>(malloc(sizeof(percept)));
	if (Percept == NULL) {
		fprintf(stderr, "Memory error");
		return NULL;
	}
	Percept->_index = index;
	Percept->_nettoInput = nettoInput;
	Percept->activationFunction = actf;
	Percept->outputFunction = outf;
	return Percept;
}

struct objective {
	Tensor1D _inputVector;
	Tensor1D _targetVector;
	Tensor1D _actualOutput;
};

typedef struct objective objective;

objective newObjective(Tensor1D input, Tensor1D target) {

	objective* pattern = static_cast<objective*>(malloc(sizeof(objective)));

	pattern->_inputVector = static_cast<double*>(malloc(sizeof(input)));
	pattern->_targetVector = static_cast<double*>(malloc(sizeof(target)));
	pattern->_actualOutput = static_cast<double*>(malloc(sizeof(target)));
	pattern->_inputVector = input;
	pattern->_targetVector = target;

	return *pattern;
}

void objectiveClr(objective* pattern) {
	free(pattern);
}

Tensor2D reverseTensor2D(Tensor2D toreverse, int vecsize) {
	int i;
	Tensor2D reversed = static_cast<double**>(malloc(vecsize * sizeof(double)));
	
	for (i = 0; i < vecsize; i++) {
		reversed[vecsize - 1 - i] = toreverse[i];
	}
	return reversed;
}

double derivativeSigmoid(naturAL* network, int i, int j);

double derivativeReLU(naturAL* network, int i, int j);

double derivativeLeakyReLU(naturAL* network, int i, int j);

double derivativeLinear(naturAL* network, int i, int j);

double derivativeThres(naturAL* network, int i, int j);

void SIGMOID(percept* percept) {
	percept->_activation = 1 / (1 + exp(-1 * percept->_nettoInput));
	percept->derivative = derivativeSigmoid;
}

void LINEAR(percept* percept) {
	double netinput = percept->_nettoInput;
	if (netinput >= -1 && netinput <= 1) percept->_activation = netinput;
	else if (netinput < -1) percept->_activation = -1.0;
	else if (netinput > 1) percept->_activation = 1.0;
	percept->derivative = derivativeLinear;
}

void RELU(percept* percept) {
	if (percept->_nettoInput >= 0) {
		percept->_activation = percept->_nettoInput;
	}
	else {
		percept->_activation = 0;
	}
	percept->derivative = derivativeReLU;
}

void LEAKY_RELU(percept* percept) {
	if (percept->_nettoInput >= 0) {
		percept->_activation = percept->_nettoInput;
	}
	else {
		percept->_activation = 0.33 * percept->_nettoInput;
	}
	percept->derivative = derivativeLeakyReLU;
}

void THRESHOLD_OUT(percept* percept) {
	if (percept->_activation > 0.5) {
		percept->_output = 1.0;
	}
	else {
		percept->_output = 0.0;
	}
}

void SIGMOID_OUT(percept* percept) {
	percept->_output = 1 / (1 + exp(-1 * percept->_activation));
}

void LINEAR_OUT(percept* percept) {
	double activ = percept->_nettoInput;
	if (activ >= -1 && activ <= 1) percept->_output = activ;
	else if (activ < -1) percept->_output = -1.0;
	else if (activ > 1) percept->_output = 1.0;
}

void RELU_OUT(percept* percept) {
	if (percept->_activation >= 0) {
		percept->_output = percept->_activation;
	}
	else {
		percept->_output = 0;
	}
}

void IDENTITY(percept* percept) {
	percept->_output = percept->_activation;
}

struct weightsTensor {
	int _columns;
	int _rows;
	double **_matrix;
};

typedef struct weightsTensor weightsTensor;

weightsTensor* weightsTensor_init(int columns, int rows) {

	int i, j;
	weightsTensor* weightsBatch = static_cast<weightsTensor*>(malloc(sizeof(weightsTensor)));
	if (weightsBatch == NULL) {
		fprintf(stderr, "No memory for tensor init");
		return NULL;
	}
	weightsBatch->_columns = columns;
	weightsBatch->_rows = rows;
	weightsBatch->_matrix = static_cast<double**>(malloc(rows*columns*sizeof(double*)));
	if (weightsBatch->_matrix == NULL) {
		fprintf(stderr, "Tensor allocation error");
		free(weightsBatch);
		return NULL;
	}
	for (i = 0; i < rows; i++) {
		weightsBatch->_matrix[i] = static_cast<double*>(malloc(columns*columns * sizeof(double)));
		if (weightsBatch->_matrix[i] == NULL) {
			fprintf(stderr, "Tensor allocation error");
			free(weightsBatch->_matrix);
			free(weightsBatch);
			return NULL;
		}
		for (j = 0; j < columns; j++) {
			weightsBatch->_matrix[i][j] = 0.0;
		}
	}
	return weightsBatch;
}

int saveWeights(weightsTensor matrix, const char* outfilename, const char* precision) {
	int i, j;
	FILE* outfile;

	outfile = fopen(outfilename, "w");
	if (outfile == NULL) {
		perror("Save error");
		return 0;
	}

	for (i = 0; i < matrix._rows; i++) {
		for (j = 0; j < matrix._columns; j++) {
			fprintf(outfile, precision , matrix._matrix[i][j]);
		}
		fprintf(outfile, "\n");
	}
	fclose(outfile);

	return 1;
}

int loadWeights(weightsTensor matrix, const char* infilename) {
	int i, j;
	FILE* infile;
	double read;

	infile = fopen(infilename, "r");
	if (infile == NULL) {
		perror("Load error");
		return 0;
	}
	
	for (i = 0; i < matrix._rows; i++) {
		for (j = 0; j < matrix._columns; j++) {
			fscanf(infile, "%lf ", &read);
			matrix._matrix[i][j] = read;
		}
	}
	fclose(infile);
	return 1;
}

int saveWeightsSinglePrecision(weightsTensor matrix, const char* outfilename) {
	FILE* outfile;

	outfile = fopen(outfilename, "wb");
	if (outfile == NULL) {
		perror("Save error");
		return 0;
	}

	for (int i = 0; i < matrix._rows; i++) {
		for (int j = 0; j < matrix._columns; j++) {
			float temp = (float)matrix._matrix[i][j];
			fwrite(&temp, sizeof(float), 1, outfile);
		}
	}
	fclose(outfile);

	return 1;
}

int loadWeightsSinglePrecision(weightsTensor matrix, const char* infilename) {
	FILE* infile;

	infile = fopen(infilename, "rb");
	if (infile == NULL) {
		perror("Load error");
		return 0;
	}
	
	float temp;
	for (int i = 0; i < matrix._rows; i++) {
		for (int j = 0; j < matrix._columns; j++) {
			fread(&temp, sizeof(float), 1, infile);
			matrix._matrix[i][j] = (double)temp;
		}
	}
	fclose(infile);
	return 1;
}

struct naturAL {
	int _numberOfLayers;
	int* _perceptsPerLayer;
	percept** _perceptLayers;
	weightsTensor _weighingmatrix;
};

Tensor1D predict(naturAL* network, Tensor1D inputVector) {

	int i, j, a;
	int countLayers = network->_numberOfLayers;

	int countLastLayer = network->_perceptsPerLayer[countLayers-1];

	Tensor1D output = static_cast<double*>(malloc(countLastLayer*sizeof(double)));

	for (i = 0; i < network->_perceptsPerLayer[0]; i++) {
		network->_perceptLayers[0][i]._nettoInput = inputVector[i];
		network->_perceptLayers[0][i].activationFunction(&network->_perceptLayers[0][i]);
		network->_perceptLayers[0][i].outputFunction(&network->_perceptLayers[0][i]);
	}

		for (i = 1; i < countLayers; i++) {
		for (j = 0; j < network->_perceptsPerLayer[i]; j++) {
			
			network->_perceptLayers[i][j]._nettoInput = 0;
			for (a = 0; a < network->_perceptsPerLayer[i-1]; a++) {
				
				network->_perceptLayers[i][j]._nettoInput += network->_perceptLayers[i - 1][a]._output 
					* network->_weighingmatrix._matrix[network->_perceptLayers[i - 1][a]._index][network->_perceptLayers[i][j]._index];
				
			}
			
			network->_perceptLayers[i][j].activationFunction(&network->_perceptLayers[i][j]);
			network->_perceptLayers[i][j].outputFunction(&network->_perceptLayers[i][j]);
			
			if (i == countLayers - 1) {
				
				output[j] = network->_perceptLayers[i][j]._output;
			}
		}
	}

	return output;
}

naturAL* initModel(int numberOfLayers, int* perceptsPerLayer, void (*actinner)(percept* percept), void (*actoutput)(percept* percept), void (*outinner)(percept* percept), void (*outoutput)(percept* percept)) {
	
	naturAL* network = static_cast<naturAL*>(malloc(sizeof(naturAL)));
	network->_perceptLayers = static_cast<percept**>(malloc(numberOfLayers * numberOfLayers * sizeof(percept*)));

	network->_numberOfLayers = numberOfLayers;
	network->_perceptsPerLayer = perceptsPerLayer;

	int a, b, c, x, y;
	int layer, pl;
	int x2, y2;

	int index = 0;

	srand(time(NULL));

	for (layer = 0; layer < numberOfLayers; layer++) {
		network->_perceptLayers[layer] = static_cast<percept*>(malloc(perceptsPerLayer[layer]*perceptsPerLayer[layer] * sizeof(percept)));

		for (pl = 0; pl < perceptsPerLayer[layer]; pl++) {
			if (layer == numberOfLayers - 1) {
				network->_perceptLayers[layer][pl] = * perceptInit(index++, 0, actoutput, outoutput);
			}
			else {
				network->_perceptLayers[layer][pl] = *perceptInit(index++, 0, actinner, outinner);
			}
		}
	}

	network->_weighingmatrix = *weightsTensor_init(index, index);

	for (a = 0; a < (numberOfLayers - 1); a++) {
		for (b = 0; b < perceptsPerLayer[a]; b++) {
			for (c = 0; c < perceptsPerLayer[a + 1]; c++) {
				x = network->_perceptLayers[a][b]._index;
				y = network->_perceptLayers[a + 1][c]._index;
				network->_weighingmatrix._matrix[x][y] = 1.0;
			}
		}
	}

	for (x2 = 0; x2 < index-perceptsPerLayer[numberOfLayers-1]; x2++) {
		for (y2 = 0; y2 < index; y2++) {
			if (network->_weighingmatrix._matrix[x2][y2] == 1.0) {

				double random = (double)rand() / (double)(RAND_MAX)-(double).5;

				network->_weighingmatrix._matrix[x2][y2] = random;
			}
		}
	}

	return network;
}

void naturALClr(naturAL* network) {
	free(network->_perceptLayers);
	free(network->_weighingmatrix._matrix);
	free(network);
}

int trainModel(naturAL * network, int repetitionsNum, double learningRate, double tolerance, int samplesNum, objective * trainingSamples) {
	int rep,tp,out,i,j,z,d;
	int count = 0;
	int trainingSec = 0;

	for (rep = 0; rep < repetitionsNum; rep++) {

		count++;
		trainingSec = 0;

		for (tp = 0; tp < samplesNum; tp++) {
			
			Tensor2D netDeltas = static_cast<double**>(malloc(network->_numberOfLayers * network->_numberOfLayers * network->_numberOfLayers * sizeof(double*)));
			int netDeltaCount = 0;

			Tensor1D output = predict(network, trainingSamples[tp]._inputVector);

			int out_s = network->_numberOfLayers - 1;

			int out_n = network->_perceptsPerLayer[out_s];


			Tensor1D deltas = static_cast<double*>(malloc(out_n * out_n * sizeof(double)));
			int deltaCount = 0;

			for (out = 0; out < out_n; out++) {

				double deriv = network->_perceptLayers[out_s][out].derivative(network, out_s, out);

				double this_delta = deriv*(trainingSamples[tp]._targetVector[out] - network->_perceptLayers[out_s][out]._activation);

				deltas[deltaCount++] = this_delta;

				if (fabs(this_delta) > tolerance) {
					trainingSec = 1;
				}
			}
			netDeltas[netDeltaCount++] = deltas;


			if (trainingSec == 1) {

				int p = 0;


				for (i = network->_numberOfLayers - 2; i > 0; i--) {

					Tensor1D deltas_i = static_cast<double*>(malloc(100 * network->_numberOfLayers * network->_numberOfLayers * sizeof(double)));
					int icount = 0;

					for (j = 0; j < network->_numberOfLayers; j++) {

						double deriv = network->_perceptLayers[i][j].derivative(network, i, j);

						double sum = 0.0;
						int row = network->_perceptLayers[i][j]._index;

						for (z = 0; z < network->_perceptsPerLayer[i + 1]; z++) {

							int column = network->_perceptLayers[i + 1][z]._index;
							sum += netDeltas[p][z] * network->_weighingmatrix._matrix[row][column];

						}
						deltas_i[icount++] = deriv * sum;

					}
					netDeltas[netDeltaCount++] = deltas_i;
					p++;

				}

				netDeltas = reverseTensor2D(netDeltas, netDeltaCount);

				for (i = network->_numberOfLayers - 1; i > 0; i--) {

					for (j = 0; j < network->_perceptsPerLayer[i]; j++) {

						double delta = netDeltas[(i - 1)][j];
						int column = network->_perceptLayers[i][j]._index;

						for (z = 0; z < network->_perceptsPerLayer[i - 1]; z++) {

							int row = network->_perceptLayers[i - 1][z]._index;
							double weightDelta = learningRate * delta * network->_perceptLayers[i - 1][z]._activation;
							network->_weighingmatrix._matrix[row][column] += weightDelta;

						}
					}
				}

			}

			for (d = 0; d < netDeltaCount; d++) {
				free(netDeltas[d]);
			}
			free(netDeltas);
		}
		if (trainingSec == 0) {
			break;
		}
	}
	if (trainingSec == 0) {
		return count;
	}
	else {
		return -repetitionsNum;
	}
}


double derivativeSigmoid(naturAL* network, int i, int j) {
	return network->_perceptLayers[i][j]._activation * (1 - network->_perceptLayers[i][j]._activation);
}

double derivativeReLU(naturAL* network, int i, int j) {
	if (network->_perceptLayers[i][j]._activation > 0) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

double derivativeLeakyReLU(naturAL* network, int i, int j) {
	if (network->_perceptLayers[i][j]._activation > 0) {
		return 1.0;
	}
	else {
		return 0.33;
	}
}

double derivativeLinear(naturAL* network, int i, int j) {
	if ((network->_perceptLayers[i][j]._activation <= 1) && (network->_perceptLayers[i][j]._activation >= -1)) {
		return 1.0;
	}
	else {
		return 0;
	}
}

#endif
