#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;
// Tensorflow NHWC 
vector<int> ExtractImagePatch(const vector<int> &input, int output_shape, int ksize, int stride, int rate, const vector<int> &input_dims, string padding_type);

ostream &
operator<<(ostream &os, vector<int> &v)
{
    stringstream ss;
    ss << "[";
    for (auto &i : v)
    {
        ss << i << ",";
    }
    ss.seekp(-1, ss.cur);
    ss << "]";
    os << ss.str();
    return os;
}

ostream &
operator<<(ostream &os, vector<vector<int>> &v)
{
    stringstream ss;
    ss << "[";
    for (auto &i : v)
    {
        ss << "[";
        for (auto &j : i)
        {
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

int main()
{
    vector<int> input;

    for (int i = 1; i < 101; i++)
    {
        input.push_back(i);
    }
    int output_shape = 4 * 16;
    vector<int> input_dims = {1, 10, 10, 1};

    auto output = ExtractImagePatch(input, output_shape, 4, 7, 1, input_dims, "SAME");
    cout << output << endl;
}

vector<int> ExtractImagePatch(const vector<int> &input, int output_shape, int ksize, int stride, int rate, const vector<int> &input_dims, string padding_type)
{

    int m_inputDepth = input_dims[3];
    int m_inputRows = input_dims[2];
    int m_inputCols = input_dims[1];
    int ksize_col = ksize;
    int ksize_row = ksize;
   
    int m_in_row_strides = rate;
    int m_in_col_strides = rate;

    int m_col_strides = stride;
    int m_row_strides = stride;

    int m_row_inflate_strides = 1;
    int m_col_inflate_strides = 1;

    int m_input_rows_eff = (m_inputRows - 1) * m_row_inflate_strides + 1;
    int m_input_cols_eff = (m_inputCols - 1) * m_col_inflate_strides + 1;
    int m_patch_rows_eff = ksize_row + (ksize_row - 1) * (m_in_row_strides - 1);
    int m_patch_cols_eff = ksize_col + (ksize_col - 1) * (m_in_col_strides - 1);
    int m_outputRows = 0;
    int m_outputCols = 0;
    int m_rowPaddingTop = 0;
    int m_colPaddingLeft = 0;
    if (padding_type == "VALID")
    {
        m_outputRows = ceil((m_input_rows_eff - m_patch_rows_eff + 1.f) / static_cast<float>(m_row_strides));
        m_outputCols = ceil((m_input_cols_eff - m_patch_cols_eff + 1.f) / static_cast<float>(m_col_strides));
        m_rowPaddingTop = max(0, ((m_outputRows - 1) * m_row_strides + m_patch_rows_eff - m_input_rows_eff) / 2);
        m_colPaddingLeft = max(0, ((m_outputCols - 1) * m_col_strides + m_patch_cols_eff - m_input_cols_eff) / 2);
    }
    else
    {
        m_outputRows = ceil(m_input_rows_eff / static_cast<float>(m_row_strides));
        m_outputCols = ceil(m_input_cols_eff / static_cast<float>(m_col_strides));
        m_rowPaddingTop = ((m_outputRows - 1) * m_row_strides + m_patch_rows_eff - m_input_rows_eff) / 2;
        m_colPaddingLeft = ((m_outputCols - 1) * m_col_strides + m_patch_cols_eff - m_input_cols_eff) / 2;
    }
    int m_dimensions[4];
    m_dimensions[0] = input_dims[3];
    m_dimensions[1] = ksize_row;
    m_dimensions[2] = ksize_col;
    m_dimensions[3] = m_outputRows * m_outputCols;

    int m_colStride = m_dimensions[1];
    int m_patchStride = m_colStride * m_dimensions[2] * m_dimensions[0];
    int m_otherStride = m_patchStride * m_dimensions[3];

    int m_rowInputStride = m_inputDepth;
    int m_colInputStride = m_inputDepth * m_inputRows;
    int m_patchInputStride = m_inputDepth * m_inputRows * m_inputCols;
    int m_fastOutputDepth = m_dimensions[0];

    vector<int> output(output_shape, 0);
    int needBatch = (output.size() - 1) / m_otherStride;
    for (int i = 0; i < output.size(); i++)
    {
        int batchIndex = needBatch ? (i / m_otherStride) : 0;
        int innerIndex = needBatch ? (i - batchIndex * m_otherStride) : i;
        // inner index
        int patchIndex = innerIndex / m_patchStride;
        int patchOffset = (innerIndex - patchIndex * m_patchStride) / m_fastOutputDepth;

        int colIndex = patchIndex / m_outputRows;
        int colOffset = patchOffset / m_colStride;

        int inputCol = colIndex * m_col_strides + colOffset * m_in_col_strides - m_colPaddingLeft;
        if (inputCol < 0 || inputCol >= m_input_cols_eff)
        {
            output[i] = 0;
            continue;
        }

        int rowIndex = patchIndex - colIndex * m_outputRows;
        int rowOffset = patchOffset - colOffset * m_colStride;

        int inputRow = rowIndex * m_row_strides + rowOffset * m_in_row_strides - m_rowPaddingTop;
        if (inputRow < 0 || inputRow >= m_input_rows_eff)
        {
            output[i] = 0;
            continue;
        }
        int depth = innerIndex - (innerIndex / m_fastOutputDepth) * m_fastOutputDepth;

        int inputIndex = depth + inputRow * m_rowInputStride + inputCol * m_colInputStride + batchIndex * m_patchInputStride;
        output[i] = input[inputIndex];
    }
    return output;
}
