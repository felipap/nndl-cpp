// chap1.cpp

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

  Example(uint8_t *input, int inSize, uint8_t *output, int outSize);
  Example(const Example &e) { throw domain_error("can't copy example"); }

  int getSize() { return data.size(); }
  void printGrid(int rows, int cols);

  const VectorXd &getInput() { return data; }
  const VectorXd &getLabel() { return label; }

private:
  int inSize, outSize;
  VectorXd data, label;
};

// Read example input from array of bytes.
Example::Example(uint8_t *cs, int inSize, uint8_t *output, int outSize)
: inSize(inSize), outSize(outSize) {
  data = VectorXd(inSize);
  for (int i=0; i<inSize; ++i) {
    data[i] = int(cs[i]);
  }
  label = VectorXd(outSize);
  for (int i=0; i<outSize; ++i) {
    label[i] = output[i];
  }
}

void Example::printGrid(int w, int h) {
  cout << label << endl;
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
  return Example(input, gridSize, output, 10);
}

class NeuralNetwork {
public:
  NeuralNetwork(int *layout, int lsize);

  void printWeights();
  const vector<MatrixXd> backprop(Example &e);
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

  cout << "> Biases: " << endl;
  for (int i=1; i<layout.size(); ++i) {
    cout << "Layer :" << i << ": ";
    for (int d=0; d<layout[i]; ++d) {
      cout << setw(4) << biases[i-1][d] << " ";
    }
    cout << endl;
  }

  cout << "\n> Weights: " << endl;
  for (int i=0; i<layout.size(); ++i) {
    cout << "Layer " << i << ":\n" << weights[0] << "\n";
  }
  cout << string(60, '-') << endl;
}


NeuralNetwork::NeuralNetwork(int *_layout, int size) : depth(size-1) {
	layout = vector<int>(_layout, _layout+size);

	for (int i=1; i<depth; ++i) { // We don't need biases for the first layer.
	  biases.push_back(VectorXd::Random(layout[i]));
	}

  for (int i=0; i<depth; ++i) {
    MatrixXd matrix = MatrixXd::Random(layout[i], layout[i+1]);
    weights.push_back(matrix);
  }

  //MatrixXd l1(3,2);
  //l1 << 1, 2, 3, 4, 5, 6;
  //weights.push_back(l1);
  //MatrixXd l2(2,3);
  //l2 << .1, .2, .3, .4, .5, .6;
  //weights.push_back(l2);

  //cout << feedForward(MatrixXd::Ones(3, 1));
}


VectorXd NeuralNetwork::feedForward(const VectorXd input) {
  MatrixXd A(input);
  for (int i=0; i<depth; ++i) {
    A = sigmoid(weights[i].transpose()*A);
  }
  return A;
}

VectorXd sigmoid(const VectorXd &base) {
  return base.unaryExpr([](double x) { return 1 / (1 + exp(-x) ); });
}

VectorXd sigmoidPrime(const VectorXd &base) {
  VectorXd sig = sigmoid(base);
  //cout << sig << endl;
  //return VectorXd::Ones(base.size());
  return sig.cwiseProduct(VectorXd(1-sig.array()));
}

VectorXd NeuralNetwork::costDerivative(VectorXd f_x, VectorXd f_y) {
  return f_x-f_y;
}

const vector<MatrixXd> NeuralNetwork::backprop(Example &e) {
  vector<MatrixXd> grad(depth); // Store final weight gradient, to be returned.
  vector<VectorXd> zs(depth); // Store z vectors for all layers.

  // Fill up grad matrices with zeroes.
  for (int i=0; i<depth; ++i) {
    auto &w = weights[i];
    grad[i] = MatrixXd::Zero(w.rows(), w.cols());
  }

  // Keep track of activations, including input from example.
  vector<VectorXd> actvs(depth+1);
  actvs[0] = e.getInput();

  // Feedforward.
  for (int i=0; i<depth; i++) {
    // !! We're making unnecessary copies here.
    VectorXd z = weights[i].transpose()*actvs[i];
    zs[i] = z;
    actvs[i+1] = sigmoid(z);
  }
  
  // Backprop. Here's the math-intense part.
  VectorXd delta = costDerivative(actvs[depth], e.getLabel())
    .cwiseProduct(sigmoidPrime(zs.back()));
  
  grad[depth-1] = delta * actvs[depth].transpose();
  cout << "Grad: \n" << grad[depth-1] << endl;

  // Loop layers, from depth-2 to 0, finding the nablas and partial derivatives.
  for (int i=depth-2; i >= 0; --i) {
    printf("i=%d\n", i);
    //cout << "Porra: \n" << sigmoidPrime(zs[i]) << endl;
    delta = (weights[i+1]*delta).cwiseProduct(sigmoidPrime(zs[i]));
    //cout << "Delta: \n" << delta << endl;
    grad[i] = delta * actvs[i].transpose();
    //cout << "Actvs " << i << endl << actvs.back() << endl;
  }

  cout << "Grad " << grad[1] << endl;

  return grad;
}

int main() {
  DataLoader dl("train-labels-idx1-ubyte", "train-images-idx3-ubyte");

	int layout[] = {784, 30, 10};
  NeuralNetwork net(layout, sizeof(layout)/sizeof(*layout));

  for (int i=0; i<100; ++i) {
    system("clear");
    cout << "let's get next\n";
    Example example = dl.getNext();
    cout << "let's backprop\n";
    // example.printGrid(28, 28);
    net.backprop(example);
  }

  return 0;
}
