#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define LEARNING_RATE 0.01f
#define NUM_TRAIN 60000
#define NUM_TEST 10000

static void *xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "ERROR: unable to allocate %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void matmul_forward(const float *A, const float *B, float *C,
                           int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

/* C = A^T B, where A is m x n and B is m x k. */
static void matmul_at_b(const float *A, const float *B, float *C,
                        int m, int n, int k)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < m; l++) {
                sum += A[l * n + i] * B[l * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

/* C = A B^T, where A is m x n and B is k x n. */
static void matmul_a_bt(const float *A, const float *B, float *C,
                        int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += A[i * n + l] * B[j * n + l];
            }
            C[i * k + j] = sum;
        }
    }
}

static void relu_forward(float *x, int size)
{
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

static void relu_backward(const float *dY, const float *activation,
                          float *dX, int size)
{
    for (int i = 0; i < size; i++) {
        dX[i] = activation[i] > 0.0f ? dY[i] : 0.0f;
    }
}

static void bias_forward(float *Z, const float *b, int m, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            Z[i * k + j] += b[j];
        }
    }
}

static void softmax(float *logits, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float max_val = -INFINITY;
        for (int c = 0; c < cols; c++) {
            if (logits[r * cols + c] > max_val) {
                max_val = logits[r * cols + c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float value = expf(logits[r * cols + c] - max_val);
            logits[r * cols + c] = value;
            sum += value;
        }
        for (int c = 0; c < cols; c++) {
            logits[r * cols + c] /= sum;
        }
    }
}

static float cross_entropy_loss(const float *probs, const int *y,
                                int rows, int cols)
{
    float total_loss = 0.0f;
    for (int r = 0; r < rows; r++) {
        float probability = fmaxf(probs[r * cols + y[r]], 1.0e-12f);
        total_loss -= logf(probability);
    }
    return total_loss / (float)rows;
}

static void sgd_update(float *params, const float *grads, int size)
{
    for (int i = 0; i < size; i++) {
        params[i] -= LEARNING_RATE * grads[i];
    }
}

static void softmax_ce_grad(const float *probs, const int *y,
                            float *dlogits, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dlogits[r * cols + c] = probs[r * cols + c] / (float)rows;
        }
        dlogits[r * cols + y[r]] -= 1.0f / (float)rows;
    }
}

static void linear_backward(const float *X, const float *W, const float *dY,
                            float *dX, float *dW, float *db,
                            int rows, int in, int out)
{
    for (int j = 0; j < out; j++) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            sum += dY[r * out + j];
        }
        db[j] = sum;
    }
    matmul_at_b(X, dY, dW, rows, in, out);
    matmul_a_bt(dY, W, dX, rows, out, in);
}

static void init_weights(float *W, int fan_in, int fan_out)
{
    const float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        float u = (float)rand() / (float)RAND_MAX;
        W[i] = (u * 2.0f - 1.0f) * limit;
    }
}

static void load_floats(const char *path, float *dst, int count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "ERROR: cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t got = fread(dst, sizeof(*dst), (size_t)count, file);
    if (got != (size_t)count) {
        fprintf(stderr, "ERROR: %s short read %zu/%d\n", path, got, count);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

static void load_ints(const char *path, int *dst, int count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "ERROR: cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    size_t got = fread(dst, sizeof(*dst), (size_t)count, file);
    if (got != (size_t)count) {
        fprintf(stderr, "ERROR: %s short read %zu/%d\n", path, got, count);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

static int count_correct(const float *probs, const int *labels, int rows)
{
    int correct = 0;
    for (int r = 0; r < rows; r++) {
        int best = 0;
        for (int c = 1; c < OUTPUT_SIZE; c++) {
            if (probs[r * OUTPUT_SIZE + c] >
                probs[r * OUTPUT_SIZE + best]) {
                best = c;
            }
        }
        if (best == labels[r]) {
            correct++;
        }
    }
    return correct;
}

int main(int argc, char **argv)
{
    int epochs = 20;
    if (argc > 2) {
        fprintf(stderr, "usage: %s [epochs]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        char *end = NULL;
        long value = strtol(argv[1], &end, 10);
        if (*argv[1] == '\0' || *end != '\0' || value < 0 || value > 10000) {
            fprintf(stderr, "ERROR: epochs must be an integer from 0 to 10000\n");
            return EXIT_FAILURE;
        }
        epochs = (int)value;
    }

    srand(42);

    float *W1 = xmalloc(sizeof(*W1) * INPUT_SIZE * HIDDEN_SIZE);
    float *b1 = calloc(HIDDEN_SIZE, sizeof(*b1));
    float *W2 = xmalloc(sizeof(*W2) * HIDDEN_SIZE * OUTPUT_SIZE);
    float *b2 = calloc(OUTPUT_SIZE, sizeof(*b2));
    if (b1 == NULL || b2 == NULL) {
        fprintf(stderr, "ERROR: unable to allocate bias arrays\n");
        return EXIT_FAILURE;
    }
    init_weights(W1, INPUT_SIZE, HIDDEN_SIZE);
    init_weights(W2, HIDDEN_SIZE, OUTPUT_SIZE);

    float *dW1 = xmalloc(sizeof(*dW1) * INPUT_SIZE * HIDDEN_SIZE);
    float *db1 = xmalloc(sizeof(*db1) * HIDDEN_SIZE);
    float *dW2 = xmalloc(sizeof(*dW2) * HIDDEN_SIZE * OUTPUT_SIZE);
    float *db2 = xmalloc(sizeof(*db2) * OUTPUT_SIZE);

    float *z1 = xmalloc(sizeof(*z1) * BATCH_SIZE * HIDDEN_SIZE);
    float *z2 = xmalloc(sizeof(*z2) * BATCH_SIZE * OUTPUT_SIZE);
    float *dlogits = xmalloc(sizeof(*dlogits) * BATCH_SIZE * OUTPUT_SIZE);
    float *da1 = xmalloc(sizeof(*da1) * BATCH_SIZE * HIDDEN_SIZE);
    float *dz1 = xmalloc(sizeof(*dz1) * BATCH_SIZE * HIDDEN_SIZE);
    float *dXin = xmalloc(sizeof(*dXin) * BATCH_SIZE * INPUT_SIZE);

    float *X = xmalloc(sizeof(*X) * NUM_TRAIN * INPUT_SIZE);
    int *y = xmalloc(sizeof(*y) * NUM_TRAIN);
    float *Xt = xmalloc(sizeof(*Xt) * NUM_TEST * INPUT_SIZE);
    int *yt = xmalloc(sizeof(*yt) * NUM_TEST);

    load_floats("data/X_train.bin", X, NUM_TRAIN * INPUT_SIZE);
    load_ints("data/y_train.bin", y, NUM_TRAIN);
    load_floats("data/X_test.bin", Xt, NUM_TEST * INPUT_SIZE);
    load_ints("data/y_test.bin", yt, NUM_TEST);
    printf("loaded %d train / %d test samples\n", NUM_TRAIN, NUM_TEST);

    const int num_batches = NUM_TRAIN / BATCH_SIZE;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct = 0;

        for (int b = 0; b < num_batches; b++) {
            float *Xb = X + b * BATCH_SIZE * INPUT_SIZE;
            int *yb = y + b * BATCH_SIZE;

            matmul_forward(Xb, W1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
            bias_forward(z1, b1, BATCH_SIZE, HIDDEN_SIZE);
            relu_forward(z1, BATCH_SIZE * HIDDEN_SIZE);

            matmul_forward(z1, W2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            bias_forward(z2, b2, BATCH_SIZE, OUTPUT_SIZE);
            softmax(z2, BATCH_SIZE, OUTPUT_SIZE);

            epoch_loss +=
                cross_entropy_loss(z2, yb, BATCH_SIZE, OUTPUT_SIZE);
            correct += count_correct(z2, yb, BATCH_SIZE);

            softmax_ce_grad(z2, yb, dlogits, BATCH_SIZE, OUTPUT_SIZE);
            linear_backward(z1, W2, dlogits, da1, dW2, db2,
                            BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            relu_backward(da1, z1, dz1, BATCH_SIZE * HIDDEN_SIZE);
            linear_backward(Xb, W1, dz1, dXin, dW1, db1,
                            BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);

            sgd_update(W1, dW1, INPUT_SIZE * HIDDEN_SIZE);
            sgd_update(b1, db1, HIDDEN_SIZE);
            sgd_update(W2, dW2, HIDDEN_SIZE * OUTPUT_SIZE);
            sgd_update(b2, db2, OUTPUT_SIZE);
        }

        printf("epoch %2d | loss %.4f | acc %.1f%%\n",
               epoch + 1, epoch_loss / (float)num_batches,
               100.0f * (float)correct / (float)(num_batches * BATCH_SIZE));
    }

    int test_correct = 0;
    const int test_batches = NUM_TEST / BATCH_SIZE;
    for (int b = 0; b < test_batches; b++) {
        float *Xb = Xt + b * BATCH_SIZE * INPUT_SIZE;
        int *yb = yt + b * BATCH_SIZE;

        matmul_forward(Xb, W1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
        bias_forward(z1, b1, BATCH_SIZE, HIDDEN_SIZE);
        relu_forward(z1, BATCH_SIZE * HIDDEN_SIZE);
        matmul_forward(z1, W2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        bias_forward(z2, b2, BATCH_SIZE, OUTPUT_SIZE);
        softmax(z2, BATCH_SIZE, OUTPUT_SIZE);
        test_correct += count_correct(z2, yb, BATCH_SIZE);
    }

    printf("---\ntest accuracy: %.2f%%\n",
           100.0f * (float)test_correct /
               (float)(test_batches * BATCH_SIZE));

    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(dW1);
    free(db1);
    free(dW2);
    free(db2);
    free(z1);
    free(z2);
    free(dlogits);
    free(da1);
    free(dz1);
    free(dXin);
    free(X);
    free(y);
    free(Xt);
    free(yt);
    return EXIT_SUCCESS;
}
