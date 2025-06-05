
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

// функция активации ReLU
double relu(double x) {
    return max(0.0, x);
}

// Производная функции ReLU
double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

class MLP {
private:
    vector<vector<double>> weights_ih; // Веса между входным и скрытым слоем
    vector<vector<double>> weights_ho; // Веса между скрытым и выходным слоем
    vector<double> bias_h; // Смещения для скрытого слоя
    vector<double> bias_o; // Смещения для выходного слоя
    
    double learning_rate;
    
public:
    MLP(int input_size, int hidden_size, int output_size, double lr) 
        : learning_rate(lr) {
        // Инициализация весов случайными значениями
        srand(13);
        
        // Веса между входным и скрытым слоем
        weights_ih.resize(hidden_size, vector<double>(input_size));
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights_ih[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
            }
        }
        
        // Веса между скрытым и выходным слоем
        weights_ho.resize(output_size, vector<double>(hidden_size));
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights_ho[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1, 1]
            }
        }
        
        // Инициализация смещений
        bias_h.resize(hidden_size);
        bias_o.resize(output_size);
        for (int i = 0; i < hidden_size; ++i) {
            bias_h[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        for (int i = 0; i < output_size; ++i) {
            bias_o[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
    }
    
    // Прямое распространение
    vector<double> forward(const vector<double>& inputs) {
        // Активация скрытого слоя
        vector<double> hidden(weights_ih.size());
        for (int i = 0; i < weights_ih.size(); ++i) {
            hidden[i] = 0.0;
            for (int j = 0; j < inputs.size(); ++j) {
                hidden[i] += inputs[j] * weights_ih[i][j];
            }
            hidden[i] += bias_h[i];
            hidden[i] = relu(hidden[i]);
        }
        
        // Активация выходного слоя
        vector<double> output(weights_ho.size());
        for (int i = 0; i < weights_ho.size(); ++i) {
            output[i] = 0.0;
            for (int j = 0; j < hidden.size(); ++j) {
                output[i] += hidden[j] * weights_ho[i][j];
            }
            output[i] += bias_o[i];
            output[i] = relu(output[i]);
        }
        
        return output;
    }
    
    // Обратное распространение ошибки
    void backward(const vector<double>& inputs, const vector<double>& targets) {
        // Прямой проход для получения активаций
        vector<double> hidden(weights_ih.size());
        for (int i = 0; i < weights_ih.size(); ++i) {
            hidden[i] = 0.0;
            for (int j = 0; j < inputs.size(); ++j) {
                hidden[i] += inputs[j] * weights_ih[i][j];
            }
            hidden[i] += bias_h[i];
            hidden[i] = relu(hidden[i]);
        }
        
        vector<double> output(weights_ho.size());
        for (int i = 0; i < weights_ho.size(); ++i) {
            output[i] = 0.0;
            for (int j = 0; j < hidden.size(); ++j) {
                output[i] += hidden[j] * weights_ho[i][j];
            }
            output[i] += bias_o[i];
            output[i] = relu(output[i]);
        }
        
        // Вычисление ошибок выходного слоя
        vector<double> output_errors(output.size());
        for (int i = 0; i < output.size(); ++i) {
            output_errors[i] = targets[i] - output[i];
        }
        
        // Вычисление градиентов для выходного слоя
        vector<double> output_deltas(output.size());
        for (int i = 0; i < output.size(); ++i) {
            output_deltas[i] = output_errors[i] * relu_derivative(output[i]);
        }
        
        // Вычисление ошибок скрытого слоя
        vector<double> hidden_errors(hidden.size());
        for (int j = 0; j < hidden.size(); ++j) {
            hidden_errors[j] = 0.0;
            for (int i = 0; i < output.size(); ++i) {
                hidden_errors[j] += output_deltas[i] * weights_ho[i][j];
            }
        }
        
        // Вычисление градиентов для скрытого слоя
        vector<double> hidden_deltas(hidden.size());
        for (int j = 0; j < hidden.size(); ++j) {
            hidden_deltas[j] = hidden_errors[j] * relu_derivative(hidden[j]);
        }
        
        // Обновление весов между скрытым и выходным слоем
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < hidden.size(); ++j) {
                weights_ho[i][j] += learning_rate * output_deltas[i] * hidden[j];
            }
            bias_o[i] += learning_rate * output_deltas[i];
        }
        
        // Обновление весов между входным и скрытым слоем
        for (int i = 0; i < hidden.size(); ++i) {
            for (int j = 0; j < inputs.size(); ++j) {
                weights_ih[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
            }
            bias_h[i] += learning_rate * hidden_deltas[i];
        }
    }
    
    // Функция обучения
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
        for (int e = 0; e < epochs; ++e) {
            double error = 0.0;
            for (int i = 0; i < inputs.size(); ++i) {
                backward(inputs[i], targets[i]);
                vector<double> output = forward(inputs[i]);
                for (int j = 0; j < output.size(); ++j) {
                    error += 0.5 * pow(targets[i][j] - output[j], 2);
                }
            }
            if (e % 1000 == 0) {
                cout << "Epoch " << e << ", error: " << error << endl;
            }
        }
    }

    void save_weights(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return;
    }
    
    // Сохраняем веса input-hidden
    for (const auto& row : weights_ih) {
        for (double w : row) file << w << " ";
        file << endl;
    }
    
    // Сохраняем веса hidden-output
    for (const auto& row : weights_ho) {
        for (double w : row) file << w << " ";
        file << endl;
    }
    
    // Сохраняем смещения
    for (double b : bias_h) file << b << " ";
    file << endl;
    for (double b : bias_o) file << b << " ";
    
    file.close();
}

void load_weights(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for reading!" << endl;
        return;
    }
    
    // Загружаем веса input-hidden
    for (auto& row : weights_ih) {
        for (double& w : row) file >> w;
    }
    
    // Загружаем веса hidden-output
    for (auto& row : weights_ho) {
        for (double& w : row) file >> w;
    }
    
    // Загружаем смещения
    for (double& b : bias_h) file >> b;
    for (double& b : bias_o) file >> b;
    
    file.close();
}
};

int main() {
    // Создаем MLP с 2 входами, 2 нейронами в скрытом слое и 1 выходом
    MLP mlp(2, 4, 1, 0.001);
    
    // Данные для обучения XOR
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
    
    // Обучаем сеть
    mlp.train(inputs, targets, 100000);
    
    // Тестируем сеть
    cout << "Testing trained network:" << endl;
    for (const auto& input : inputs) {
        vector<double> output = mlp.forward(input);
        cout << input[0] << " XOR " << input[1] << " = " << output[0] << " (expected: " << (input[0] != input[1]) << ")" << endl;
    }
    
    mlp.save_weights("xor_weights.txt");
    
    return 0;  
}
