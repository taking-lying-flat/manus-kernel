#pragma once

// Lightweight matrix storage and CPU kernels.

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mnist {

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t columns)
        : rows_{rows}, columns_{columns}, values_(rows * columns) {}

    Matrix(std::size_t rows, std::size_t columns, std::vector<float> values)
        : rows_{rows}, columns_{columns}, values_{std::move(values)} {
        if (values_.size() != rows * columns) {
            throw std::invalid_argument{"matrix data does not match shape"};
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    float &operator()(std::size_t row, std::size_t column) noexcept {
        return values_[row * columns_ + column];
    }
    const float &operator()(std::size_t row,
                            std::size_t column) const noexcept {
        return values_[row * columns_ + column];
    }

    [[nodiscard]] std::span<float> data() noexcept { return values_; }
    [[nodiscard]] std::span<const float> data() const noexcept {
        return values_;
    }

    [[nodiscard]] std::span<const float>
    rows_view(std::size_t first, std::size_t count) const {
        if (first > rows_ || count > rows_ - first) {
            throw std::out_of_range{"matrix row view is out of bounds"};
        }
        return data().subspan(first * columns_, count * columns_);
    }

private:
    std::size_t rows_;
    std::size_t columns_;
    std::vector<float> values_;
};

inline void matmul(std::span<const float> left, std::size_t rows,
                   std::size_t inner, const Matrix &right, Matrix &output) {
    if (left.size() != rows * inner || right.rows() != inner ||
        output.rows() != rows || output.columns() != right.columns()) {
        throw std::invalid_argument{"invalid matrix multiplication shapes"};
    }

    const auto right_data = right.data();
    auto result = output.data();
    const auto columns = right.columns();

#pragma omp parallel for if (rows >= 64)
    for (std::ptrdiff_t signed_row = 0;
         signed_row < static_cast<std::ptrdiff_t>(rows); ++signed_row) {
        const auto row = static_cast<std::size_t>(signed_row);
        auto output_row = result.subspan(row * columns, columns);
        std::fill(output_row.begin(), output_row.end(), 0.0F);
        for (std::size_t index = 0; index < inner; ++index) {
            const float value = left[row * inner + index];
            const std::size_t offset = index * columns;
            for (std::size_t column = 0; column < columns; ++column) {
                output_row[column] += value * right_data[offset + column];
            }
        }
    }
}

inline void matmul(const Matrix &left, const Matrix &right, Matrix &output) {
    matmul(left.data(), left.rows(), left.columns(), right, output);
}

} // namespace mnist
