// chap1.cpp
// This was pretty useful:
// https://eigen.tuxfamily.org/dox-devel/AsciiQuickReference.txt

#include "chap1.hpp"

#include <iostream>
#include <functional>
#include <utility>
#include <cstdlib>
#include <vector>
#include <iomanip> 
#include <fstream>
#include <random>
#include <algorithm>
#include <iterator>
#include <cstdint>

using namespace std;

#include <eigen3/Eigen/Dense>


using namespace Eigen;

#define _swapend __builtin_bswap32

class Example {
public:

  Example(uint8_t *input, int inSize, uint8_t *output, int outSize, int y);
  Example(const Example &e) { throw domain_error("can't copy example"); }

  int getSize() { return data.size(); }
  void printGrid(int rows, int cols);

  const VectorXd &getInput() const { return data; }
  const VectorXd &getLabel() const { return label; }

  int y;

private:
  int inSize, outSize;
  VectorXd data, label;
};

// Read example input from array of bytes.
Example::Example(uint8_t *cs, int inSize, uint8_t *output, int outSize, int y)
: inSize(inSize), outSize(outSize), y(y) {
  data = VectorXd(inSize);
  for (int i=0; i<inSize; ++i) {
    data[i] = int(cs[i])/256.0;
  }
  label = VectorXd(outSize);
  for (int i=0; i<outSize; ++i) {
    label[i] = output[i];
  }
}

void Example::printGrid(int w, int h) {
  cout << y << endl;
  for (int y=0; y<h; ++y) {
    for (int x=0; x<w; ++x) {
      printf("%d ", int(data[y*w+x]));
    }
    cout << endl;
  }
}


class DataLoader {
public:
  Example getNext();
  const vector<Example> getNext(int num);

  DataLoader(const string& p, const string& p2);
  int imageHeight, imageWidth, gridSize;
private:
  ifstream fileLabels, fileImages;
};

DataLoader::DataLoader(const string& p, const string& p2) {
    // Make sure fs exist.
    fileLabels.open(p.c_str(), ios::in|ios::binary|ios::ate);
    fileImages.open(p2.c_str(), ios::in|ios::binary|ios::ate);
    if (!fileLabels.good()) {
      throw domain_error("Labels file doesn't exist.");
    }
    if (!fileImages.good()) {
      throw domain_error("Images files doesn't exist.");
    }

    if (!fileLabels.is_open()) {
      throw runtime_error("Failed to open labels file.");
    } else if (!fileImages.is_open()) {
      throw runtime_error("Failed to open images file.");
    }
    
    // Read headers for each file
    
    int header1[2], header2[4];

    fileLabels.seekg(0, ios::beg);
    fileLabels.read((char *) &header1, sizeof(header1));
    if (_swapend(header1[0]) != 2049) {
      throw runtime_error("Labels file has wrong magic number.");
    }
    
    fileImages.seekg(0, ios::beg);
    fileImages.read((char *) &header2, sizeof(header2));
    if (_swapend(header2[0]) != 2051) {
      throw runtime_error("Images file has wrong magic number.");
    }

    if (header1[1] != header2[1]) {
      throw runtime_error("Different number of labels and images.");
    }

    imageWidth = _swapend(header2[2]);
    imageHeight = _swapend(header2[3]);
    gridSize = imageWidth*imageHeight;
}

Example DataLoader::getNext() {
  int label = 0;
  uint8_t input[gridSize], output[10] = {0};
  // Read next label and next image.
  fileLabels.read((char *) &label, 1);
  if (label < 0 || label > 9) {
    throw domain_error("Label byte with invalid value detected.");
  }
  output[label] = 1;
  fileImages.read((char *) input, gridSize);
  return Example(input, gridSize, output, 10, label);
}

const vector<Example> DataLoader::getNext(int num) {
  vector<Example> es;
  for (int i=0; i<num; i++) {
    es.push_back(getNext());
  }
  return es;
}

class NeuralNetwork {
public:
  NeuralNetwork(int *layout, int lsize);

  void printWeights();
  const vector<MatrixXd> backprop(const Example &e);
  void SGD(DataLoader &dl);
  const vector<MatrixXd> calcFromBatch(vector<Example *> &es);
private:
  int depth; 
	vector<int> layout;
	vector<VectorXd> biases;
  vector<MatrixXd> weights;

  VectorXd costDerivative(VectorXd, VectorXd);
  VectorXd feedForward(const VectorXd);
};

void NeuralNetwork::printWeights() {
  cout << string(60, '-') << endl;

  cout << "\n> Weights: " << endl;
  for (int i=0; i<depth; ++i) {
    cout << "Layer " << i << ":\n" << weights[i] << "\n";
  }
  cout << string(60, '-') << endl;
}


NeuralNetwork::NeuralNetwork(int *_layout, int size) : depth(size-1) {
	layout = vector<int>(_layout, _layout+size);

	for (int i=1; i<depth; ++i) { // We don't need biases for the first layer.
	  biases.push_back(VectorXd::Random(layout[i]+1));
	}

  for (int i=0; i<depth-1; ++i) {
    // +1 for biases. Adding +1 on the columns (weight output) also, even though
    // the bias faux-neuron shouldn't take inputs (its value must remain 1),
    // because this makes the multiplications easier in backprop.
    MatrixXd matrix = MatrixXd::Random(layout[i]+1, layout[i+1]+1);
    matrix.col(0).setZero();
    weights.push_back(matrix);
  }
  // Special case for last layer: output doesn't need a bias neuron, and having
  // one wouldn't make things easier.
  MatrixXd matrix = MatrixXd::Random(layout[depth-1]+1, layout[depth]);
  weights.push_back(matrix);
}

VectorXd sigmoid(const VectorXd &base) {
  return 1.0 / (1.0 + (-base).array().exp());
}

VectorXd sigmoidPrime(const VectorXd &base) {
  auto sig = sigmoid(base);
  return sig.array()*(1-sig.array());
}

VectorXd NeuralNetwork::feedForward(const VectorXd input) {
  if (input.size() != layout[0]) {
    throw domain_error("Wrong input dimension.");
  }

  VectorXd A(1+input.size());
  A << 1, input; // Add bias neuron (set to 1);

  for (int i=0; i<depth; ++i) {
    A[0] = 1; // Set bias neuron to 1.
    // A[0] = 1 mustn't be executed for last activation (aka. the nn output)
    // because there's no bias neuron exists in the last layer.
    A = sigmoid(weights[i].transpose()*A);
  }

  return A;
}

VectorXd NeuralNetwork::costDerivative(VectorXd f_x, VectorXd f_y) {
  return f_x-f_y;
}

const vector<MatrixXd> NeuralNetwork::backprop(const Example &e) {
  if (e.getInput().size() != layout[0]) {
    throw domain_error("Wrong input dimension.");
  }

  if (e.getLabel().size() != layout[depth]) {
    throw domain_error("Wrong output dimension.");
  }

  vector<MatrixXd> grad(depth); // Store final weight gradient, to be returned.
  vector<VectorXd> zs(depth); // Store z vectors for all layers.

  // Fill up gradient matrices with zeroes.
  for (int i=0; i<depth; ++i) {
    auto &w = weights[i];
    grad[i] = MatrixXd::Zero(w.rows(), w.cols());
  }

  // Keep track of activations, including input from example.
  vector<VectorXd> actvs(depth+1);
  actvs[0] = VectorXd(layout[0]+1);
  actvs[0] << 1, e.getInput(); // Add bias neuron with value 1.

  // Feedforward.
  for (int i=0; i<depth; i++) {
    // !! We're making unnecessary copies here. ??
    VectorXd z = weights[i].transpose()*actvs[i];
    zs[i] = z;
    actvs[i+1] = sigmoid(z);
    if (i != depth-1) {
      actvs[i+1][0] = 1; // Bias neuron.
    }
  }

  // Backprop. Here's the math-intense part.
  VectorXd delta = costDerivative(actvs[depth], e.getLabel())
    .cwiseProduct(sigmoidPrime(zs.back()));
  grad[depth-1] = actvs[depth-1]*delta.transpose();
  

  // Loop layers, from depth-2 to 0, finding the nablas and partial derivatives.
  for (int i=depth-2; i >= 0; --i) {
    delta = (weights[i+1]*delta).cwiseProduct(sigmoidPrime(zs[i]));
    grad[i] = actvs[i] * delta.transpose();
  }

  cout << "Right? " << e.y << endl << actvs[depth] << endl;

  return grad;
}

void NeuralNetwork::SGD(DataLoader &dl) {

  double batchSize = 10, lrate = 3;
  int count = 0;

  for (int i=0; i<1000; ++i) {
    vector<Example *> es;
    for (int i=0; i<batchSize; i++) {
      es.push_back(new Example(dl.getNext()));
    }
    const vector<MatrixXd> deltas = calcFromBatch(es);

    // Use sum of gradients to modify weights.
    for (int i=0; i<depth; i++) {
      weights[i] -= lrate/batchSize*deltas[i];
    }

    for (int i=0; i<batchSize; i++) {
      delete es[i];
    }
  }
}

const vector<MatrixXd> NeuralNetwork::calcFromBatch(vector<Example *> &es) {
  // Initialize sum of gradients to 0.
  vector<MatrixXd> deltas(depth);
  for (int i=0; i<depth; i++) {
    deltas[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
  }

  for (int i2=0; i2<es.size(); ++i2) {
    //system("clear");
    //cout << es[i2].getInput() << endl;
    //printf("oiem %d %d\n", i2, es[i2].getInput().size());
    vector<MatrixXd> dd = backprop(*(es[i2]));
    // Accumulate nablas.
    for (int i=0; i<dd.size(); ++i) {
      deltas[i] += dd[i];
    }
  }

  return deltas;
}

int main() {
  DataLoader dl("train-labels-idx1-ubyte", "train-images-idx3-ubyte");

	int layout[] = {784, 30, 10};
  NeuralNetwork net(layout, sizeof(layout)/sizeof(*layout));
  net.SGD(dl);

  return 0;
}
