#pragma once

#include <layers.hpp>

#include <cmath>

namespace mnist {

class AdamW {
public:
    AdamW(std::size_t size, float learning_rate, float weight_decay)
        : first_(size),
          second_(size),
          learning_rate_{learning_rate},
          weight_decay_{weight_decay} {}

    void step(std::span<float> parameters,
              std::span<const float> gradients) {
        if (parameters.size() != first_.size() ||
            gradients.size() != parameters.size()) {
            throw std::invalid_argument{"invalid AdamW parameter sizes"};
        }
        ++steps_;
        const float correction1 = 1.0F - std::pow(0.9F, steps_);
        const float correction2 = 1.0F - std::pow(0.95F, steps_);
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            const float gradient = gradients[index];
            first_[index] = 0.9F * first_[index] + 0.1F * gradient;
            second_[index] =
                0.95F * second_[index] + 0.05F * gradient * gradient;
            parameters[index] *= 1.0F - learning_rate_ * weight_decay_;
            parameters[index] -=
                learning_rate_ * (first_[index] / correction1) /
                (std::sqrt(second_[index] / correction2) + 1.0e-8F);
        }
    }

private:
    std::vector<float> first_;
    std::vector<float> second_;
    float learning_rate_;
    float weight_decay_;
    std::size_t steps_{};
};

/*
 * Muon reference algorithm: https://github.com/KellerJordan/Muon
 */
class Muon {
public:
    explicit Muon(const Matrix &parameter, float learning_rate = 0.02F)
        : rows_{parameter.rows()},
          columns_{parameter.columns()},
          learning_rate_{learning_rate},
          momentum_{rows_, columns_},
          update_{rows_, columns_},
          orthogonal_{std::min(rows_, columns_), std::max(rows_, columns_)},
          next_{orthogonal_.rows(), orthogonal_.columns()},
          gram_{orthogonal_.rows(), orthogonal_.rows()},
          gram_squared_{gram_.rows(), gram_.columns()},
          polynomial_{gram_.rows(), gram_.columns()} {}

    void step(Matrix &parameter, const Matrix &gradient) {
        if (parameter.rows() != rows_ || parameter.columns() != columns_ ||
            gradient.rows() != rows_ || gradient.columns() != columns_) {
            throw std::invalid_argument{"invalid Muon parameter shape"};
        }

        auto momentum = momentum_.data();
        auto update = update_.data();
        const auto gradients = gradient.data();
        for (std::size_t index = 0; index < gradients.size(); ++index) {
            momentum[index] = 0.95F * momentum[index] +
                              0.05F * gradients[index];
            update[index] =
                0.05F * gradients[index] + 0.95F * momentum[index];
        }
        orthogonalize();

        const float scale = std::sqrt(std::max(
            1.0F, static_cast<float>(rows_) / static_cast<float>(columns_)));
        auto values = parameter.data();
        const auto direction = orthogonal_.data();
#pragma omp parallel for
        for (std::ptrdiff_t signed_row = 0;
             signed_row < static_cast<std::ptrdiff_t>(rows_); ++signed_row) {
            const auto row = static_cast<std::size_t>(signed_row);
            for (std::size_t column = 0; column < columns_; ++column) {
                const std::size_t position = row * columns_ + column;
                const std::size_t direction_position =
                    rows_ > columns_ ? column * rows_ + row : position;
                values[position] *= 1.0F - learning_rate_ * 0.01F;
                values[position] -=
                    learning_rate_ * scale * direction[direction_position];
            }
        }
    }

private:
    void orthogonalize() {
        const auto update = update_.data();
        double squared_norm = 0.0;
#pragma omp parallel for reduction(+ : squared_norm)
        for (std::ptrdiff_t index = 0;
             index < static_cast<std::ptrdiff_t>(update.size()); ++index) {
            const double value = update[static_cast<std::size_t>(index)];
            squared_norm += value * value;
        }
        const float inverse_norm =
            1.0F / (static_cast<float>(std::sqrt(squared_norm)) + 1.0e-7F);

#pragma omp parallel for
        for (std::ptrdiff_t signed_row = 0;
             signed_row < static_cast<std::ptrdiff_t>(rows_); ++signed_row) {
            const auto row = static_cast<std::size_t>(signed_row);
            for (std::size_t column = 0; column < columns_; ++column) {
                if (rows_ > columns_) {
                    orthogonal_(column, row) =
                        update_(row, column) * inverse_norm;
                } else {
                    orthogonal_(row, column) =
                        update_(row, column) * inverse_norm;
                }
            }
        }

        constexpr float a = 3.4445F;
        constexpr float b = -4.7750F;
        constexpr float c = 2.0315F;
        const auto short_side = orthogonal_.rows();
        const auto long_side = orthogonal_.columns();

        for (int iteration = 0; iteration < 5; ++iteration) {
#pragma omp parallel for
            for (std::ptrdiff_t signed_row = 0;
                 signed_row < static_cast<std::ptrdiff_t>(short_side);
                 ++signed_row) {
                const auto row = static_cast<std::size_t>(signed_row);
                for (std::size_t column = row; column < short_side;
                     ++column) {
                    float sum = 0.0F;
                    for (std::size_t index = 0; index < long_side; ++index) {
                        sum += orthogonal_(row, index) *
                               orthogonal_(column, index);
                    }
                    gram_(row, column) = gram_(column, row) = sum;
                }
            }

            matmul(gram_, gram_, gram_squared_);
            for (std::size_t index = 0; index < polynomial_.size(); ++index) {
                polynomial_.data()[index] =
                    b * gram_.data()[index] +
                    c * gram_squared_.data()[index];
            }
            matmul(polynomial_, orthogonal_, next_);
            for (std::size_t index = 0; index < next_.size(); ++index) {
                next_.data()[index] += a * orthogonal_.data()[index];
            }
            std::swap(orthogonal_, next_);
        }
    }

    std::size_t rows_;
    std::size_t columns_;
    float learning_rate_;
    Matrix momentum_;
    Matrix update_;
    Matrix orthogonal_;
    Matrix next_;
    Matrix gram_;
    Matrix gram_squared_;
    Matrix polynomial_;
};

class MuonOptimizer {
public:
    explicit MuonOptimizer(MLP &model)
        : model_{model},
          muon_{model.first().weight()},
          second_weight_{model.second().weight().size(), 0.003F, 0.01F},
          first_bias_{model.first().bias().size(), 0.003F, 0.0F},
          second_bias_{model.second().bias().size(), 0.003F, 0.0F} {}

    void step() {
        muon_.step(model_.first().weight(),
                   model_.first().weight_gradient());
        second_weight_.step(model_.second().weight().data(),
                            model_.second().weight_gradient().data());
        first_bias_.step(model_.first().bias(),
                         model_.first().bias_gradient());
        second_bias_.step(model_.second().bias(),
                          model_.second().bias_gradient());
    }

private:
    MLP &model_;
    Muon muon_;
    AdamW second_weight_;
    AdamW first_bias_;
    AdamW second_bias_;
};

} // namespace mnist
