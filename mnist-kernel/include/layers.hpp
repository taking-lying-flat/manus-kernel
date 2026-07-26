#pragma once

#include <tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace mnist {

class Linear {
public:
    Linear(std::size_t input_size, std::size_t output_size,
           std::size_t batch_size, std::mt19937 &generator)
        : input_size_{input_size},
          output_size_{output_size},
          batch_size_{batch_size},
          weight_{input_size, output_size},
          bias_(output_size),
          output_{batch_size, output_size},
          weight_gradient_{input_size, output_size},
          bias_gradient_(output_size),
          input_gradient_{batch_size, input_size} {
        const float limit = std::sqrt(
            6.0F / static_cast<float>(input_size + output_size));
        std::uniform_real_distribution<float> distribution{-limit, limit};
        std::generate(weight_.data().begin(), weight_.data().end(),
                      [&] { return distribution(generator); });
    }

    const Matrix &forward(std::span<const float> input) {
        if (input.size() != batch_size_ * input_size_) {
            throw std::invalid_argument{"Linear received an invalid input"};
        }
        cached_input_ = input;
        matmul(input, batch_size_, input_size_, weight_, output_);
        for (std::size_t row = 0; row < batch_size_; ++row) {
            for (std::size_t column = 0; column < output_size_; ++column) {
                output_(row, column) += bias_[column];
            }
        }
        return output_;
    }

    const Matrix &forward(const Matrix &input) {
        return forward(input.data());
    }

    const Matrix &backward(const Matrix &output_gradient,
                           bool need_input_gradient = true) {
        if (output_gradient.rows() != batch_size_ ||
            output_gradient.columns() != output_size_ ||
            cached_input_.empty()) {
            throw std::invalid_argument{"Linear backward state is invalid"};
        }

        compute_parameter_gradients(output_gradient);
        if (need_input_gradient) {
            compute_input_gradient(output_gradient);
        }
        return input_gradient_;
    }

    Matrix &weight() noexcept { return weight_; }
    std::span<float> bias() noexcept { return bias_; }
    const Matrix &weight_gradient() const noexcept { return weight_gradient_; }
    std::span<const float> bias_gradient() const noexcept {
        return bias_gradient_;
    }

private:
    void compute_parameter_gradients(const Matrix &gradient) {
        for (std::size_t column = 0; column < output_size_; ++column) {
            float sum = 0.0F;
            for (std::size_t row = 0; row < batch_size_; ++row) {
                sum += gradient(row, column);
            }
            bias_gradient_[column] = sum;
        }

        auto weight_gradient = weight_gradient_.data();
        const auto output_gradient = gradient.data();
#pragma omp parallel for if (batch_size_ >= 64)
        for (std::ptrdiff_t signed_column = 0;
             signed_column < static_cast<std::ptrdiff_t>(input_size_);
             ++signed_column) {
            const auto input_column =
                static_cast<std::size_t>(signed_column);
            auto gradient_row = weight_gradient.subspan(
                input_column * output_size_, output_size_);
            std::fill(gradient_row.begin(), gradient_row.end(), 0.0F);
            for (std::size_t row = 0; row < batch_size_; ++row) {
                const float value =
                    cached_input_[row * input_size_ + input_column];
                const std::size_t offset = row * output_size_;
                for (std::size_t column = 0; column < output_size_; ++column) {
                    gradient_row[column] +=
                        value * output_gradient[offset + column];
                }
            }
        }
    }

    void compute_input_gradient(const Matrix &gradient) {
#pragma omp parallel for if (batch_size_ >= 64)
        for (std::ptrdiff_t signed_row = 0;
             signed_row < static_cast<std::ptrdiff_t>(batch_size_);
             ++signed_row) {
            const auto row = static_cast<std::size_t>(signed_row);
            for (std::size_t input = 0; input < input_size_; ++input) {
                float sum = 0.0F;
                for (std::size_t output = 0; output < output_size_; ++output) {
                    sum += gradient(row, output) * weight_(input, output);
                }
                input_gradient_(row, input) = sum;
            }
        }
    }

    std::size_t input_size_;
    std::size_t output_size_;
    std::size_t batch_size_;
    Matrix weight_;
    std::vector<float> bias_;
    Matrix output_;
    Matrix weight_gradient_;
    std::vector<float> bias_gradient_;
    Matrix input_gradient_;
    std::span<const float> cached_input_;
};

class ReLU {
public:
    ReLU(std::size_t batch_size, std::size_t features)
        : output_{batch_size, features}, gradient_{batch_size, features} {}

    const Matrix &forward(const Matrix &input) {
        if (input.rows() != output_.rows() ||
            input.columns() != output_.columns()) {
            throw std::invalid_argument{"ReLU received an invalid input"};
        }
        std::transform(input.data().begin(), input.data().end(),
                       output_.data().begin(),
                       [](float value) { return std::max(0.0F, value); });
        return output_;
    }

    const Matrix &backward(const Matrix &output_gradient) {
        if (output_gradient.rows() != output_.rows() ||
            output_gradient.columns() != output_.columns()) {
            throw std::invalid_argument{"ReLU received an invalid gradient"};
        }
        const auto incoming = output_gradient.data();
        const auto activation = output_.data();
        auto result = gradient_.data();
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] =
                activation[index] > 0.0F ? incoming[index] : 0.0F;
        }
        return gradient_;
    }

private:
    Matrix output_;
    Matrix gradient_;
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss(std::size_t batch_size, std::size_t classes)
        : probabilities_{batch_size, classes},
          gradient_{batch_size, classes} {}

    float forward(const Matrix &logits,
                  std::span<const std::int32_t> labels) {
        if (logits.rows() != probabilities_.rows() ||
            logits.columns() != probabilities_.columns() ||
            labels.size() != logits.rows()) {
            throw std::invalid_argument{"loss received invalid inputs"};
        }
        labels_ = labels;

        float total = 0.0F;
        for (std::size_t row = 0; row < logits.rows(); ++row) {
            float maximum = std::numeric_limits<float>::lowest();
            for (std::size_t column = 0; column < logits.columns(); ++column) {
                maximum = std::max(maximum, logits(row, column));
            }

            float sum = 0.0F;
            for (std::size_t column = 0; column < logits.columns(); ++column) {
                const float value = std::exp(logits(row, column) - maximum);
                probabilities_(row, column) = value;
                sum += value;
            }
            for (std::size_t column = 0; column < logits.columns(); ++column) {
                probabilities_(row, column) /= sum;
            }
            const auto label = static_cast<std::size_t>(labels[row]);
            total -=
                std::log(std::max(probabilities_(row, label), 1.0e-12F));
        }
        return total / static_cast<float>(labels.size());
    }

    const Matrix &backward() {
        if (labels_.empty()) {
            throw std::logic_error{"loss backward called before forward"};
        }
        const float scale = 1.0F / static_cast<float>(labels_.size());
        std::transform(probabilities_.data().begin(),
                       probabilities_.data().end(), gradient_.data().begin(),
                       [scale](float value) { return value * scale; });
        for (std::size_t row = 0; row < labels_.size(); ++row) {
            gradient_(row, static_cast<std::size_t>(labels_[row])) -= scale;
        }
        return gradient_;
    }

private:
    Matrix probabilities_;
    Matrix gradient_;
    std::span<const std::int32_t> labels_;
};

class MLP {
public:
    MLP(std::size_t input_size, std::size_t hidden_size,
        std::size_t output_size, std::size_t batch_size)
        : generator_{42},
          first_{input_size, hidden_size, batch_size, generator_},
          activation_{batch_size, hidden_size},
          second_{hidden_size, output_size, batch_size, generator_} {}

    const Matrix &forward(std::span<const float> input) {
        return second_.forward(activation_.forward(first_.forward(input)));
    }

    void backward(const Matrix &gradient) {
        const Matrix &hidden_gradient = second_.backward(gradient);
        first_.backward(activation_.backward(hidden_gradient), false);
    }

    Linear &first() noexcept { return first_; }
    Linear &second() noexcept { return second_; }

private:
    std::mt19937 generator_;
    Linear first_;
    ReLU activation_;
    Linear second_;
};

} // namespace mnist
