#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int INPUT = 28 * 28;
constexpr int HIDDEN = 256;
constexpr int OUTPUT = 10;
constexpr int BATCH = 1024;
constexpr int TRAINING = 60'000;
constexpr int TEST = 10'000;
constexpr int THREADS = 256;

void cuda_check(cudaError_t status, const char *call, const char *file,
                int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error{std::string{file} + ':' +
                                 std::to_string(line) + ": " + call + ": " +
                                 cudaGetErrorString(status)};
    }
}

void cublas_check(cublasStatus_t status, const char *call, const char *file,
                  int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error{std::string{file} + ':' +
                                 std::to_string(line) + ": " + call +
                                 " failed (" + std::to_string(status) + ')'};
    }
}

#define CUDA(call) cuda_check((call), #call, __FILE__, __LINE__)
#define CUBLAS(call) cublas_check((call), #call, __FILE__, __LINE__)

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_{size} {
        CUDA(cudaMalloc(reinterpret_cast<void **>(&data_), size * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    T *get() noexcept { return data_; }
    const T *get() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    T *data_{};
    std::size_t size_{};
};

class BlasHandle {
public:
    BlasHandle() { CUBLAS(cublasCreate(&handle_)); }
    ~BlasHandle() { cublasDestroy(handle_); }
    operator cublasHandle_t() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{};
};

template <class T>
std::vector<T> read_binary(const std::filesystem::path &path,
                           std::size_t count) {
    const auto bytes = count * sizeof(T);
    if (std::filesystem::file_size(path) != bytes) {
        throw std::runtime_error{path.string() + " has an unexpected size"};
    }
    std::vector<T> values(count);
    std::ifstream file{path, std::ios::binary};
    if (!file.read(reinterpret_cast<char *>(values.data()), bytes)) {
        throw std::runtime_error{"cannot read " + path.string()};
    }
    return values;
}

void initialize_weight(float *device, int rows, int columns,
                       std::mt19937 &generator) {
    std::vector<float> host(static_cast<std::size_t>(rows) * columns);
    const float bound = std::sqrt(6.0F / static_cast<float>(rows));
    std::uniform_real_distribution<float> distribution{-bound, bound};
    std::generate(host.begin(), host.end(),
                  [&] { return distribution(generator); });
    CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                    cudaMemcpyHostToDevice));
}

struct Network {
    DeviceBuffer<float> w1{INPUT * HIDDEN}, b1{HIDDEN};
    DeviceBuffer<float> w2{HIDDEN * OUTPUT}, b2{OUTPUT};
    DeviceBuffer<float> dw1{INPUT * HIDDEN}, db1{HIDDEN};
    DeviceBuffer<float> dw2{HIDDEN * OUTPUT}, db2{OUTPUT};

    Network() {
        std::mt19937 generator{42};
        initialize_weight(w1.get(), INPUT, HIDDEN, generator);
        initialize_weight(w2.get(), HIDDEN, OUTPUT, generator);
        CUDA(cudaMemset(b1.get(), 0, b1.size() * sizeof(float)));
        CUDA(cudaMemset(b2.get(), 0, b2.size() * sizeof(float)));
    }
};

__global__ void add_bias(float *values, const float *bias, int count,
                         int features, bool relu) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = values[index] + bias[index % features];
        values[index] = relu ? fmaxf(value, 0.0F) : value;
    }
}

__global__ void relu_backward(float *gradient, const float *activation,
                              int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && activation[index] <= 0.0F) {
        gradient[index] = 0.0F;
    }
}

__global__ void bias_backward(float *result, const float *gradient, int rows,
                              int columns) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < columns) {
        float sum = 0.0F;
        for (int row = 0; row < rows; ++row) {
            sum += gradient[row * columns + column];
        }
        result[column] = sum;
    }
}

__global__ void softmax_loss_backward(float *logits,
                                      const std::int32_t *labels, int rows,
                                      int columns, float *loss, int *correct) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float *values = logits + row * columns;
    float maximum = values[0];
    int best = 0;
    for (int column = 1; column < columns; ++column) {
        if (values[column] > maximum) {
            maximum = values[column];
            best = column;
        }
    }
    float sum = 0.0F;
    for (int column = 0; column < columns; ++column) {
        values[column] = expf(values[column] - maximum);
        sum += values[column];
    }
    const int label = labels[row];
    atomicAdd(loss, -logf(fmaxf(values[label] / sum, 1.0e-12F)));
    atomicAdd(correct, best == label);
    for (int column = 0; column < columns; ++column) {
        values[column] =
            (values[column] / sum - (column == label)) / rows;
    }
}

__global__ void count_correct(const float *logits,
                              const std::int32_t *labels, int rows,
                              int columns, int *correct) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int best = 0;
        for (int column = 1; column < columns; ++column) {
            if (logits[row * columns + column] >
                logits[row * columns + best]) {
                best = column;
            }
        }
        atomicAdd(correct, best == labels[row]);
    }
}

__global__ void adamw_update(float *parameter, const float *gradient, float *m,
                             float *v, int size, float learning_rate,
                             float decay, float correction1,
                             float correction2) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        const float g = gradient[index];
        m[index] = 0.9F * m[index] + 0.1F * g;
        v[index] = 0.95F * v[index] + 0.05F * g * g;
        parameter[index] *= 1.0F - learning_rate * decay;
        parameter[index] -=
            learning_rate * (m[index] / correction1) /
            (sqrtf(v[index] / correction2) + 1.0e-8F);
    }
}

class AdamW {
public:
    AdamW(int size, float learning_rate, float decay)
        : size_{size},
          learning_rate_{learning_rate},
          decay_{decay},
          m_{static_cast<std::size_t>(size)},
          v_{static_cast<std::size_t>(size)} {
        CUDA(cudaMemset(m_.get(), 0, size * sizeof(float)));
        CUDA(cudaMemset(v_.get(), 0, size * sizeof(float)));
    }

    void step(float *parameter, const float *gradient) {
        ++step_;
        const float c1 = 1.0F - std::pow(0.9F, step_);
        const float c2 = 1.0F - std::pow(0.95F, step_);
        adamw_update<<<(size_ + THREADS - 1) / THREADS, THREADS>>>(
            parameter, gradient, m_.get(), v_.get(), size_, learning_rate_,
            decay_, c1, c2);
        CUDA(cudaGetLastError());
    }

private:
    int size_;
    int step_{};
    float learning_rate_;
    float decay_;
    DeviceBuffer<float> m_, v_;
};

__global__ void muon_prepare(const float *gradient, float *momentum,
                             float *work, int rows, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns) {
        const float g = gradient[index];
        const float m = momentum[index] = 0.95F * momentum[index] + 0.05F * g;
        const float update = 0.05F * g + 0.95F * m;
        const int row = index / columns;
        const int column = index % columns;
        work[rows > columns ? column * rows + row : index] = update;
    }
}

__global__ void muon_polynomial(const float *gram, const float *squared,
                                float *polynomial, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        polynomial[index] =
            -4.7750F * gram[index] + 2.0315F * squared[index];
    }
}

__global__ void muon_finish_iteration(float *next, const float *current,
                                      int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        next[index] += 3.4445F * current[index];
    }
}

__global__ void muon_apply(float *parameter, const float *direction, int rows,
                           int columns, float learning_rate, float scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns) {
        const int row = index / columns;
        const int column = index % columns;
        const int source =
            rows > columns ? column * rows + row : index;
        parameter[index] *= 1.0F - learning_rate * 0.01F;
        parameter[index] -= learning_rate * scale * direction[source];
    }
}

class Muon {
public:
    Muon(int rows, int columns, float learning_rate = 0.02F)
        : rows_{rows},
          columns_{columns},
          short_{std::min(rows, columns)},
          long_{std::max(rows, columns)},
          learning_rate_{learning_rate},
          momentum_{static_cast<std::size_t>(rows) * columns},
          first_{static_cast<std::size_t>(rows) * columns},
          second_{static_cast<std::size_t>(rows) * columns},
          gram_{static_cast<std::size_t>(short_) * short_},
          squared_{static_cast<std::size_t>(short_) * short_},
          polynomial_{static_cast<std::size_t>(short_) * short_} {
        CUDA(cudaMemset(momentum_.get(), 0,
                        momentum_.size() * sizeof(float)));
    }

    void step(cublasHandle_t handle, float *parameter,
              const float *gradient) {
        const int count = rows_ * columns_;
        muon_prepare<<<(count + THREADS - 1) / THREADS, THREADS>>>(
            gradient, momentum_.get(), first_.get(), rows_, columns_);
        CUDA(cudaGetLastError());

        float norm = 0.0F;
        CUBLAS(cublasSnrm2(handle, count, first_.get(), 1, &norm));
        const float inverse = 1.0F / (norm + 1.0e-7F);
        CUBLAS(cublasSscal(handle, count, &inverse, first_.get(), 1));

        constexpr float one = 1.0F;
        constexpr float zero = 0.0F;
        float *current = first_.get();
        float *next = second_.get();
        for (int iteration = 0; iteration < 5; ++iteration) {
            CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, short_,
                               short_, long_, &one, current, long_, current,
                               long_, &zero, gram_.get(), short_));
            CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, short_,
                               short_, short_, &one, gram_.get(), short_,
                               gram_.get(), short_, &zero, squared_.get(),
                               short_));
            const int square = short_ * short_;
            muon_polynomial<<<(square + THREADS - 1) / THREADS, THREADS>>>(
                gram_.get(), squared_.get(), polynomial_.get(), square);
            CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, long_,
                               short_, short_, &one, current, long_,
                               polynomial_.get(), short_, &zero, next, long_));
            muon_finish_iteration<<<(count + THREADS - 1) / THREADS,
                                      THREADS>>>(next, current, count);
            std::swap(current, next);
        }
        CUDA(cudaGetLastError());
        const float scale =
            std::sqrt(std::max(1.0F, static_cast<float>(rows_) / columns_));
        muon_apply<<<(count + THREADS - 1) / THREADS, THREADS>>>(
            parameter, current, rows_, columns_, learning_rate_, scale);
        CUDA(cudaGetLastError());
    }

private:
    int rows_, columns_, short_, long_;
    float learning_rate_;
    DeviceBuffer<float> momentum_, first_, second_;
    DeviceBuffer<float> gram_, squared_, polynomial_;
};

void linear(cublasHandle_t handle, const float *input, const float *weight,
            float *output, int rows, int in, int out) {
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out, rows, in,
                       &one, weight, out, input, in, &zero, output, out));
}

void linear_backward(cublasHandle_t handle, const float *input,
                     const float *weight, const float *output_gradient,
                     float *input_gradient, float *weight_gradient,
                     float *bias_gradient, int rows, int in, int out) {
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out, in, rows,
                       &one, output_gradient, out, input, in, &zero,
                       weight_gradient, out));
    if (input_gradient) {
        CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in, rows, out,
                           &one, weight, out, output_gradient, out, &zero,
                           input_gradient, in));
    }
    bias_backward<<<(out + THREADS - 1) / THREADS, THREADS>>>(
        bias_gradient, output_gradient, rows, out);
    CUDA(cudaGetLastError());
}

float evaluate(cublasHandle_t handle, Network &network, const float *images,
               const std::int32_t *labels, float *hidden, float *logits,
               int samples, int *device_correct) {
    CUDA(cudaMemset(device_correct, 0, sizeof(int)));
    for (int offset = 0; offset < samples; offset += BATCH) {
        const int rows = std::min(BATCH, samples - offset);
        linear(handle, images + offset * INPUT, network.w1.get(), hidden,
               rows, INPUT, HIDDEN);
        add_bias<<<(rows * HIDDEN + THREADS - 1) / THREADS, THREADS>>>(
            hidden, network.b1.get(), rows * HIDDEN, HIDDEN, true);
        linear(handle, hidden, network.w2.get(), logits, rows, HIDDEN,
               OUTPUT);
        add_bias<<<(rows * OUTPUT + THREADS - 1) / THREADS, THREADS>>>(
            logits, network.b2.get(), rows * OUTPUT, OUTPUT, false);
        count_correct<<<(rows + THREADS - 1) / THREADS, THREADS>>>(
            logits, labels + offset, rows, OUTPUT, device_correct);
    }
    CUDA(cudaGetLastError());
    int correct = 0;
    CUDA(cudaMemcpy(&correct, device_correct, sizeof(int),
                    cudaMemcpyDeviceToHost));
    return 100.0F * correct / samples;
}

int run(int epochs) {
    const std::filesystem::path directory{"data"};
    auto train_images =
        read_binary<float>(directory / "X_train.bin", TRAINING * INPUT);
    auto train_labels =
        read_binary<std::int32_t>(directory / "y_train.bin", TRAINING);
    auto test_images =
        read_binary<float>(directory / "X_test.bin", TEST * INPUT);
    auto test_labels =
        read_binary<std::int32_t>(directory / "y_test.bin", TEST);
    const auto valid_label = [](std::int32_t value) {
        return value >= 0 && value < OUTPUT;
    };
    if (!std::ranges::all_of(train_labels, valid_label) ||
        !std::ranges::all_of(test_labels, valid_label)) {
        throw std::runtime_error{"label outside [0, 9]"};
    }

    DeviceBuffer<float> d_train_images(train_images.size());
    DeviceBuffer<std::int32_t> d_train_labels(train_labels.size());
    DeviceBuffer<float> d_test_images(test_images.size());
    DeviceBuffer<std::int32_t> d_test_labels(test_labels.size());
    CUDA(cudaMemcpy(d_train_images.get(), train_images.data(),
                    train_images.size() * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_train_labels.get(), train_labels.data(),
                    train_labels.size() * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_test_images.get(), test_images.data(),
                    test_images.size() * sizeof(float),
                    cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_test_labels.get(), test_labels.data(),
                    test_labels.size() * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice));

    Network network;
    BlasHandle handle;
    DeviceBuffer<float> hidden(BATCH * HIDDEN), logits(BATCH * OUTPUT);
    DeviceBuffer<float> hidden_gradient(BATCH * HIDDEN);
    DeviceBuffer<float> epoch_loss(1);
    DeviceBuffer<int> correct(1);
    Muon muon(INPUT, HIDDEN);
    AdamW output_weight(HIDDEN * OUTPUT, 0.003F, 0.01F);
    AdamW hidden_bias(HIDDEN, 0.003F, 0.0F);
    AdamW output_bias(OUTPUT, 0.003F, 0.0F);

    std::cout << "loaded " << TRAINING << " train / " << TEST
              << " test | CUDA cuBLAS | Muon + AdamW | batch " << BATCH
              << '\n';
    for (int epoch = 0; epoch < epochs; ++epoch) {
        CUDA(cudaMemset(epoch_loss.get(), 0, sizeof(float)));
        CUDA(cudaMemset(correct.get(), 0, sizeof(int)));
        for (int offset = 0; offset < TRAINING; offset += BATCH) {
            const int rows = std::min(BATCH, TRAINING - offset);
            const float *images = d_train_images.get() + offset * INPUT;
            const std::int32_t *labels = d_train_labels.get() + offset;

            linear(handle, images, network.w1.get(), hidden.get(), rows,
                   INPUT, HIDDEN);
            add_bias<<<(rows * HIDDEN + THREADS - 1) / THREADS, THREADS>>>(
                hidden.get(), network.b1.get(), rows * HIDDEN, HIDDEN, true);
            linear(handle, hidden.get(), network.w2.get(), logits.get(), rows,
                   HIDDEN, OUTPUT);
            add_bias<<<(rows * OUTPUT + THREADS - 1) / THREADS, THREADS>>>(
                logits.get(), network.b2.get(), rows * OUTPUT, OUTPUT, false);
            softmax_loss_backward<<<(rows + THREADS - 1) / THREADS, THREADS>>>(
                logits.get(), labels, rows, OUTPUT, epoch_loss.get(),
                correct.get());
            CUDA(cudaGetLastError());

            linear_backward(handle, hidden.get(), network.w2.get(),
                            logits.get(), hidden_gradient.get(),
                            network.dw2.get(), network.db2.get(), rows, HIDDEN,
                            OUTPUT);
            relu_backward<<<(rows * HIDDEN + THREADS - 1) / THREADS,
                              THREADS>>>(hidden_gradient.get(), hidden.get(),
                                        rows * HIDDEN);
            linear_backward(handle, images, network.w1.get(),
                            hidden_gradient.get(), nullptr, network.dw1.get(),
                            network.db1.get(), rows, INPUT, HIDDEN);

            muon.step(handle, network.w1.get(), network.dw1.get());
            output_weight.step(network.w2.get(), network.dw2.get());
            hidden_bias.step(network.b1.get(), network.db1.get());
            output_bias.step(network.b2.get(), network.db2.get());
        }
        float loss = 0.0F;
        int hits = 0;
        CUDA(cudaMemcpy(&loss, epoch_loss.get(), sizeof(float),
                        cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(&hits, correct.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
        std::cout << "epoch " << epoch + 1 << " | loss " << loss / TRAINING
                  << " | acc " << 100.0F * hits / TRAINING << "%\n";
    }

    const float accuracy =
        evaluate(handle, network, d_test_images.get(), d_test_labels.get(),
                 hidden.get(), logits.get(), TEST, correct.get());
    std::cout << "---\ntest accuracy: " << accuracy << "%\n";
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc > 2) {
            throw std::invalid_argument{"usage: mnist [epochs]"};
        }
        const int epochs = argc > 1 ? std::stoi(argv[1]) : 20;
        if (epochs < 0 || epochs > 10'000) {
            throw std::invalid_argument{"epochs must be in [0, 10000]"};
        }
        return run(epochs);
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
