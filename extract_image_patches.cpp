#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
// Tensorflow NHWC
vector<int> ExtractImagePatch(const vector<int> &input, int output_shape,
                              int *ksize, int *stride, int *rate,
                              int *input_dims, string padding_type);

ostream &operator<<(ostream &os, vector<int> &v) {
    stringstream ss;
    ss << "[";
    for (auto &i : v) {
        ss << i << ",";
    }
    ss.seekp(-1, ss.cur);
    ss << "]";
    os << ss.str();
    return os;
}

ostream &operator<<(ostream &os, vector<vector<int>> &v) {
    stringstream ss;
    ss << "[";
    for (auto &i : v) {
        ss << "[";
        for (auto &j : i) {
            ss << j << ",";
        }
        ss.seekp(-1, ss.cur);
        ss << "]," << endl;
    }
    ss.seekp(-2, ss.cur);
    ss << "]";
    os << ss.str();
    return os;
}

int main() {
    vector<int> input;
    for (int i = 0; i < 600; i++) {
        input.push_back(i);
    }
    int output_shape = 3 * 2 * 2 * 32;
    int input_dims[] = {3, 10, 10, 2};
    int ksize[] = {4, 4};
    int stride[] = {7, 7};
    int rate[] = {3, 3};
    auto output = ExtractImagePatch(input, output_shape, ksize, stride, rate,
                                    input_dims, "SAME");
    cout << input[540] << endl;
    cout << output[330] << endl;
}

vector<int> ExtractImagePatch(const vector<int> &input, int output_shape,
                              int *ksize, int *stride, int *rate,
                              int *input_dims, string padding_type) {
    int inputDepth = input_dims[3];
    int inputColSize = input_dims[2];
    int inputRowSize = input_dims[1];

    int ksize_row = ksize[0];
    int ksize_col = ksize[1];
    int row_rate = rate[0];
    int col_rate = rate[1];
    int row_strides = stride[0];
    int col_strides = stride[1];

    int patch_rows_eff = ksize_row + (ksize_row - 1) * (row_rate - 1);
    int patch_cols_eff = ksize_col + (ksize_col - 1) * (col_rate - 1);

    int outputRows = 0;
    int outputCols = 0;
    int rowPaddingTop = 0;
    int colPaddingLeft = 0;

    if (padding_type == "VALID") {
        outputRows = ceil((inputRowSize - patch_rows_eff + 1.f) /
                          static_cast<float>(row_strides));
        outputCols = ceil((inputColSize - patch_cols_eff + 1.f) /
                          static_cast<float>(col_strides));
        rowPaddingTop = max(0, ((outputRows - 1) * row_strides +
                                patch_rows_eff - inputRowSize) /
                                   2);
        colPaddingLeft = max(0, ((outputCols - 1) * col_strides +
                                 patch_cols_eff - inputColSize) /
                                    2);
    } else {
        outputRows = ceil(inputRowSize / static_cast<float>(row_strides));
        outputCols = ceil(inputColSize / static_cast<float>(col_strides));
        rowPaddingTop =
            ((outputRows - 1) * row_strides + patch_rows_eff - inputRowSize) /
            2;
        colPaddingLeft =
            ((outputCols - 1) * col_strides + patch_cols_eff - inputColSize) /
            2;
    }

    int rowStride = ksize_col;
    int patchStride = rowStride * ksize_row * inputDepth;
    int otherStride = patchStride * outputRows * outputCols;

    int colInputStride = inputDepth;
    int rowInputStride = inputDepth * inputColSize;
    int patchInputStride = inputDepth * inputColSize * inputRowSize;
    int OutputDepth = inputDepth;

    vector<int> output(output_shape, 0);
    int needBatch = (output.size() - 1) / otherStride;

    for (int i = 0; i < output.size(); i++) {
        int batchIndex = needBatch ? (i / otherStride) : 0;
        int innerIndex = needBatch ? (i - batchIndex * otherStride) : i;
        // inner index
        int patchIndex = innerIndex / patchStride;
        int patchOffset = (innerIndex - patchIndex * patchStride) / OutputDepth;
        // row
        int rowIndex = patchIndex / outputCols;
        int rowOffset = patchOffset / rowStride;
        int inputRow =
            rowIndex * row_strides + rowOffset * row_rate - rowPaddingTop;
        if (inputRow < 0 || inputRow >= inputRowSize) {
            output[i] = 0;
            continue;
        }
        // col
        int colIndex = patchIndex - rowIndex * outputCols;
        int colOffset = patchOffset - rowOffset * rowStride;
        int inputCol =
            colIndex * col_strides + colOffset * col_rate - colPaddingLeft;
        if (inputCol < 0 || inputCol >= inputColSize) {
            output[i] = 0;
            continue;
        }
        // depth
        int depth = innerIndex - (innerIndex / OutputDepth) * OutputDepth;
        // input index
        int inputIndex = depth + inputCol * colInputStride +
                         inputRow * rowInputStride +
                         batchIndex * patchInputStride;
        output[i] = input[inputIndex];
    }
    return output;
}
