#include <optimizers.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mnist {

constexpr std::size_t kInputSize = 28 * 28;
constexpr std::size_t kHiddenSize = 256;
constexpr std::size_t kOutputSize = 10;
constexpr std::size_t kBatchSize = 1'000;
constexpr std::size_t kTrainingSamples = 60'000;
constexpr std::size_t kTestSamples = 10'000;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(std::int32_t) == 4);
static_assert(std::endian::native == std::endian::little);

struct Dataset {
    Matrix images;
    std::vector<std::int32_t> labels;

    [[nodiscard]] std::size_t size() const noexcept {
        return images.rows();
    }
};

struct Options {
    int epochs{20};
    std::filesystem::path data_directory;
};

template <typename T>
std::vector<T> read_binary(const std::filesystem::path &path,
                           std::size_t count) {
    const auto expected_bytes = count * sizeof(T);
    std::error_code error;
    const auto actual_bytes = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error{"cannot inspect " + path.string() + ": " +
                                 error.message()};
    }
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error{
            path.string() + " has " + std::to_string(actual_bytes) +
            " bytes; expected " + std::to_string(expected_bytes)};
    }

    std::ifstream input{path, std::ios::binary};
    std::vector<T> values(count);
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error{"cannot read " + path.string()};
    }
    return values;
}

Dataset load_dataset(const std::filesystem::path &directory,
                     std::string_view image_file,
                     std::string_view label_file, std::size_t count) {
    auto images =
        read_binary<float>(directory / image_file, count * kInputSize);
    auto labels =
        read_binary<std::int32_t>(directory / label_file, count);

    if (std::ranges::any_of(labels, [](std::int32_t label) {
            return label < 0 ||
                   label >= static_cast<std::int32_t>(kOutputSize);
        })) {
        throw std::runtime_error{"dataset label is outside [0, 9]"};
    }
    return {Matrix{count, kInputSize, std::move(images)}, std::move(labels)};
}

std::size_t count_correct(const Matrix &logits,
                          std::span<const std::int32_t> labels) {
    std::size_t correct = 0;
    for (std::size_t row = 0; row < labels.size(); ++row) {
        std::size_t best = 0;
        for (std::size_t column = 1; column < logits.columns(); ++column) {
            if (logits(row, column) > logits(row, best)) {
                best = column;
            }
        }
        correct += best == static_cast<std::size_t>(labels[row]);
    }
    return correct;
}

int parse_epochs(std::string_view argument) {
    int epochs{};
    const auto [position, error] = std::from_chars(
        argument.data(), argument.data() + argument.size(), epochs);
    if (error != std::errc{} ||
        position != argument.data() + argument.size() || epochs < 0 ||
        epochs > 10'000) {
        throw std::invalid_argument{
            "epochs must be an integer from 0 to 10000"};
    }
    return epochs;
}

Options parse_options(int argc, char **argv) {
    if (argc > 3) {
        throw std::invalid_argument{
            "usage: " + std::string{argv[0]} +
            " [epochs] [data_directory]"};
    }

    Options options;
#ifdef MNIST_DEFAULT_DATA_DIRECTORY
    options.data_directory = MNIST_DEFAULT_DATA_DIRECTORY;
#else
    options.data_directory =
        std::filesystem::absolute(argv[0]).parent_path() / "data";
#endif
    if (argc >= 2) {
        options.epochs = parse_epochs(argv[1]);
    }
    if (argc == 3) {
        options.data_directory = argv[2];
    }
    return options;
}

float evaluate(MLP &model, const Dataset &dataset,
               std::size_t current_batch_size) {
    std::size_t correct = 0;
    for (std::size_t offset = 0; offset < dataset.size();
         offset += current_batch_size) {
        const auto images =
            dataset.images.rows_view(offset, current_batch_size);
        const auto labels =
            std::span<const std::int32_t>{dataset.labels}.subspan(
                offset, current_batch_size);
        correct += count_correct(model.forward(images), labels);
    }
    return 100.0F * static_cast<float>(correct) /
           static_cast<float>(dataset.size());
}

int run(const Options &options) {
    Dataset training =
        load_dataset(options.data_directory, "X_train.bin", "y_train.bin",
                     kTrainingSamples);
    Dataset test =
        load_dataset(options.data_directory, "X_test.bin", "y_test.bin",
                     kTestSamples);
    if (training.size() % kBatchSize != 0 ||
        test.size() % kBatchSize != 0) {
        throw std::runtime_error{"batch size does not divide dataset"};
    }

    MLP model{kInputSize, kHiddenSize, kOutputSize, kBatchSize};
    CrossEntropyLoss criterion{kBatchSize, kOutputSize};
    MuonOptimizer optimizer{model};

    std::cout << "loaded " << training.size() << " train / " << test.size()
              << " test samples | optimizer muon+adamw | batch "
              << kBatchSize << '\n';
    const auto started = std::chrono::steady_clock::now();

    const std::size_t batches = training.size() / kBatchSize;
    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        float total_loss = 0.0F;
        std::size_t correct = 0;
        for (std::size_t batch = 0; batch < batches; ++batch) {
            const std::size_t offset = batch * kBatchSize;
            const auto images =
                training.images.rows_view(offset, kBatchSize);
            const auto labels =
                std::span<const std::int32_t>{training.labels}.subspan(
                    offset, kBatchSize);

            // PyTorch-style: forward -> loss -> backward -> optimizer.step().
            const Matrix &logits = model.forward(images);
            total_loss += criterion.forward(logits, labels);
            correct += count_correct(logits, labels);
            model.backward(criterion.backward());
            optimizer.step();
        }

        std::cout << "epoch " << std::setw(2) << epoch + 1 << " | loss "
                  << std::fixed << std::setprecision(4)
                  << total_loss / static_cast<float>(batches) << " | acc "
                  << std::setprecision(1)
                  << 100.0F * static_cast<float>(correct) /
                         static_cast<float>(training.size())
                  << "%\n";
    }

    const float accuracy = evaluate(model, test, kBatchSize);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - started;
    std::cout << "---\ntest accuracy: " << std::fixed << std::setprecision(2)
              << accuracy << "%\nelapsed: " << elapsed.count() << " s\n";
    return 0;
}

} // namespace mnist

int main(int argc, char **argv) {
    try {
        return mnist::run(mnist::parse_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
