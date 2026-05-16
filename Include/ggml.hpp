#pragma once

#include "ggmlPocket.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

std::string weightPathLocation = "/mnt/c/Users/shiva/Desktop/IISC/temp/ExampleTemp/LLAMA/Weight3BInstruct"; // Add weight folder location

std::string ReadTxtFile(std::string fileName)
{
    std::string path = weightPathLocation + "/" + fileName;
    std::fstream newFile;
    std::string temp;
    newFile.open(path, std::ios::in);
    if (!newFile.is_open())
    {
        std::cerr << "Error: Could not open file " << path << std::endl;
        exit(1);
    }
    std::string res = "";
    while (getline(newFile, temp))
    {
        if (temp != "")
            res += temp;
    }
    return res;
}

std::vector<std::string> SplitString(const std::string &str, const std::string &spliter)
{
    std::vector<std::string> res;

    if (str.empty() || spliter.empty())
        return res;

    size_t start = 0;
    size_t end = str.find(spliter);

    while (end != std::string::npos)
    {
        if (end > start)
        {
            res.emplace_back(str.substr(start, end - start));
        }
        start = end + spliter.length();
        end = str.find(spliter, start);
    }

    if (start < str.length())
    {
        res.emplace_back(str.substr(start));
    }

    return res;
}

pocket *InitPocket(std::vector<int64_t> shape, ggml_backend *mem, ggml_type type)
{
    return new pocket(mem, ggml_new_tensor(mem->ctx, type, shape.size(), shape.data()), shape.size());
}

void LoadBin(std::string filename, float *arr, size_t numElement)
{
    filename = weightPathLocation + "/" + filename;

    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file");

    file.read(reinterpret_cast<char *>(arr), numElement * sizeof(float));
}

void Loadpocket(std::string filename, pocket *tensor)
{
    /*
        -> Load pocket from bin file.
        -> supported datatypes are
        {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q3_0, GGML_TYPE_Q3_1}
    */
    ggml_type type = tensor->ten->type;
    std::string fullPath = weightPathLocation + "/" + filename;

    std::ifstream file(fullPath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file: " + fullPath);

    if (type == GGML_TYPE_F32)
    {
        std::vector<float> tempVector(tensor->size);
        size_t sizeInBytes = ggml_nbytes(tensor->ten);

        file.read((char *)tempVector.data(), sizeInBytes);
        ggml_backend_tensor_set(tensor->ten, tempVector.data(), 0, sizeInBytes);
    }

    // Used for Quantized and Float 16
    else
    {
        // Load the raw F32 data from disk into a temporary vector
        std::vector<float> tempFloat32(tensor->size);
        file.read((char *)tempFloat32.data(), tensor->size * sizeof(float));

        if (type == GGML_TYPE_F16)
        {
            // 1. Create a temporary CPU staging buffer for the F16 data
            std::vector<ggml_fp16_t> tempF16(tensor->size);

            // 2. Convert F32 -> F16 in standard CPU RAM
            for (size_t i = 0; i < tensor->size; i++)
            {
                tempF16[i] = ggml_fp32_to_fp16(tempFloat32[i]);
            }

            // 3. Push the converted F16 staging buffer to the Backend
            ggml_backend_tensor_set(tensor->ten, tempF16.data(), 0, ggml_nbytes(tensor->ten));
        }

        else // Quantized Types
        {
            int64_t nPerRow = tensor->ten->ne[0];
            int64_t nrows = tensor->size / nPerRow;

            size_t sizeInBytes = ggml_nbytes(tensor->ten);
            std::vector<char> tempVector(sizeInBytes);

            // Quantize the temporary FP32 array into the CPU staging buffer
            ggml_quantize_chunk(
                type,               // The target quantization type
                tempFloat32.data(), // Source: Standard float array
                tempVector.data(),  // Destination: CPU Staging buffer
                0,                  // Start row index (0)
                nrows,              // Total number of rows
                nPerRow,            // Number of elements per row (ne[0])
                nullptr);

            // Push the quantized staging buffer to the Backend
            ggml_backend_tensor_set(tensor->ten, tempVector.data(), 0, sizeInBytes);
        }
    }
}

void contiguous(pocket *inp, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : inp->mem;
    inp->ten = ggml_cont(mem->ctx, inp->ten);
    inp->isNotContiguous = false;
}

pocket *matmul(pocket *input, pocket *weights, ggml_backend *targetMem = nullptr)
{
    /*
        -> Always use the input context to store the output metadata
        input = [d, h, s, b] X weight = [d, t]
        output = [t, h, s, b]
        -> starting 2 dim will be same as before.
    */

    ggml_backend *mem = targetMem ? targetMem : input->mem;
    // Weights can not be permute
    if (input->isNotContiguous)
        contiguous(input, mem);

    // ggml_mul_mat(ctx, Weights, Input) = Input @ Weights.T()
    return new pocket(mem, ggml_mul_mat(mem->ctx, weights->ten, input->ten), std::max(weights->dim, input->dim));
}

pocket *permute(pocket *inp, std::vector<int64_t> shape, ggml_backend *targetMem = nullptr)
{
    /*
        inp:  (qkvDim X head X s X b)
        permute(inp, {0, 2, 1, 3})
        qkvDim X head X s X b -> qkvDim X s X head X b
    */
   ggml_backend *mem = targetMem ? targetMem : inp->mem;
   if(inp->isNotContiguous)
    contiguous(inp, mem);
    struct ggml_tensor *res_tensor = ggml_permute(mem->ctx, inp->ten,
                                                  shape[0],
                                                  shape[1],
                                                  shape[2],
                                                  shape[3]);
    pocket *out = new pocket(mem, res_tensor, inp->dim);
    out->isNotContiguous = true;

    return out;
}

pocket *reshape(pocket *inp, std::vector<int64_t> ne, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : inp->mem;
    if (inp->isNotContiguous)
        contiguous(inp);

    struct ggml_tensor *newTensor;

    switch (ne.size())
    {
    case 1:
        newTensor = ggml_reshape_1d(mem->ctx, inp->ten,
                                    ne[0]);
        break;
    case 2:
        newTensor = ggml_reshape_2d(mem->ctx, inp->ten, ne[0], ne[1]);
        break;
    case 3:
        newTensor = ggml_reshape_3d(mem->ctx, inp->ten,
                                    ne[0], ne[1], ne[2]);
        break;
    case 4:
        newTensor = ggml_reshape_4d(mem->ctx, inp->ten,
                                    ne[0], ne[1], ne[2], ne[3]);
        break;
    default:
        std::cerr << "Error: reshape only supports 1 to 4 dimensions " << ne.size() << "." << std::endl;
        exit(1);
    }
    pocket *out = new pocket(mem, newTensor, ne.size());
    out->isNotContiguous = true;
    return out;
}

pocket *slice(pocket *inp, std::vector<int64_t> start, std::vector<int64_t> end, ggml_backend *targetMem = nullptr)
{
    /*
        -> inp: a pocket pointer, which needs to sliced
        -> start: starting indeces of slice pocket
        -> end: ending indeces of slice pocket
        Example:
            slice(inp, {0, 0, 0, 10}, {101, 102, 103, 104})
            For comparison same opeartion in torch will look like
                inp[10:104, 0:103, 0:102, 0:101]
                (ggml used reversed indeces compare to torch.)

    */
    ggml_backend *mem = targetMem ? targetMem : inp->mem;
    if (inp->isNotContiguous)
        contiguous(inp, mem);
    std::vector<int64_t> endGGML(end.begin(), end.end());

    int64_t type_size = ggml_type_size(inp->ten->type);
    int64_t offset = (start[0] * type_size);

    for (int i = 1; i < start.size(); i++)
        offset += start[i] * inp->ten->nb[i];

    for (int i = 0; i < start.size(); i++)
        endGGML[i] = end[i] - start[i];

    ggml_tensor *newTensor;
    switch (start.size())
    {
    case 1:
        newTensor = ggml_view_1d(mem->ctx,
                                 inp->ten,
                                 endGGML[0],
                                 offset);
        break;
    case 2:
        newTensor = ggml_view_2d(mem->ctx, inp->ten,
                                 endGGML[0], endGGML[1],
                                 inp->ten->nb[1],
                                 offset);
        break;
    case 3:
        newTensor = ggml_view_3d(mem->ctx, inp->ten,
                                 endGGML[0], endGGML[1], endGGML[2],
                                 inp->ten->nb[1], inp->ten->nb[2],
                                 offset);
        break;
    case 4:
        newTensor = ggml_view_4d(mem->ctx, inp->ten,
                                 endGGML[0], endGGML[1], endGGML[2], endGGML[3],
                                 inp->ten->nb[1], inp->ten->nb[2], inp->ten->nb[3],
                                 offset);
        break;
    default:
        std::cerr << "Error: view only supports 1 to 4 dimensions but got " << start.size() << "." << std::endl;
        exit(1);
    }

    pocket *out = new pocket(mem, newTensor, start.size());
    out->isNotContiguous = true;
    return out;
}

pocket *mulTen(pocket *a, pocket *b, ggml_backend *targetMem = nullptr)
{
    /*
        -> Element-wise multiplication: a * b
        -> 'b' will be broadcasted to match 'a' if its dimensions are smaller.
        -> Always use the context of 'a' to store the new metadata.
    */
    ggml_backend *mem = targetMem ? targetMem : a->mem;
    if (a->isNotContiguous)
        contiguous(a, mem);
    if (b->isNotContiguous)
        contiguous(b, mem);

    struct ggml_tensor *res_tensor = ggml_mul(mem->ctx, a->ten, b->ten);

    return new pocket(mem, res_tensor, std::max(a->dim, b->dim));
}

pocket *addTen(pocket *a, pocket *b, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : a->mem;

    // Ensure contiguous memory for optimal backend computation
    if (a->isNotContiguous)
        contiguous(a, mem);
    if (b->isNotContiguous)
        contiguous(b, mem);

    // Create the addition node: a + b
    struct ggml_tensor *res_tensor = ggml_add(mem->ctx, a->ten, b->ten);

    return new pocket(mem, res_tensor, std::max(a->dim, b->dim));
}

pocket *subTen(pocket *a, pocket *b, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : a->mem;

    // Ensure contiguous memory
    if (a->isNotContiguous)
        contiguous(a, mem);
    if (b->isNotContiguous)
        contiguous(b, mem);

    // Create the subtraction node: a - b
    struct ggml_tensor *res_tensor = ggml_sub(a->mem->ctx, a->ten, b->ten);

    return new pocket(a->mem, res_tensor, std::max(a->dim, b->dim));
}

pocket *mulScal(pocket *a, float scalar_value, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : a->mem;

    if (a->isNotContiguous)
        contiguous(a, mem);

    struct ggml_tensor *res_tensor = ggml_scale(mem->ctx, a->ten, scalar_value);

    return new pocket(mem, res_tensor, a->dim);
}

// Updated copypocket in ggml.hpp
pocket* copypocket(pocket *inp, pocket *cpy, std::vector<int64_t> startCpyIndex, std::vector<int64_t> endCopyIndex, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : inp->mem;
    
    if(inp->isNotContiguous) contiguous(inp, mem);
    if(cpy->isNotContiguous) contiguous(cpy, mem); 

    pocket *val = slice(inp, startCpyIndex, endCopyIndex, mem);

    struct ggml_tensor *res_ten = ggml_cpy(mem->ctx, cpy->ten, val->ten);    

    pocket *res = new pocket(mem, res_ten, inp->dim);
    
    delete val;
    return res;
}

pocket *repeat(pocket *small, std::vector<int64_t> shape, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : small->mem;
    struct ggml_tensor *target_dummy = ggml_new_tensor(
        mem->ctx,
        small->ten->type,
        shape.size(),
        shape.data());

    struct ggml_tensor *res_tensor = ggml_repeat(
        mem->ctx,
        small->ten,
        target_dummy);

    pocket *out = new pocket(mem, res_tensor, shape.size());
    out->isNotContiguous = true;
    return out;
}

pocket *flashAttention(pocket *Q, pocket *K, pocket *V, int64_t &nPast, int64_t &seqLenQ, float &scale, ggml_backend *targetMem = nullptr)
{
    /*
        Q(query): qkvDim X s X 24 X b
        K(key) : qkvDim X (nPast + s) X 24 X b
        V(value): qkvDim X (nPast + s) X 24 X b

        Out: qkvDim X s X 24 X b
    */
        if (Q->isNotContiguous) contiguous(Q);
        if (K->isNotContiguous) contiguous(K);
        if (V->isNotContiguous) contiguous(V);
    ggml_backend *mem = targetMem ? targetMem : Q->mem;

    struct ggml_tensor *mask = nullptr;

    if (seqLenQ > 1)
    {
        int64_t seqLenKV = seqLenQ + nPast;

        struct ggml_tensor *one_elem = ggml_view_1d(mem->ctx, Q->ten, 1, 0);

        struct ggml_tensor *zero = ggml_scale(mem->ctx, one_elem, 0.0f);

        int64_t maskNE[2] = {seqLenKV, seqLenQ};
        struct ggml_tensor *targShape = ggml_new_tensor(mem->ctx, GGML_TYPE_F32, 2, maskNE);
        struct ggml_tensor *zPocket = ggml_repeat(mem->ctx, zero, targShape);

        mask = ggml_diag_mask_inf(mem->ctx, zPocket, nPast);

        mask = ggml_cast(mem->ctx, mask, GGML_TYPE_F16);
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;

    struct ggml_tensor *out = ggml_flash_attn_ext(
        mem->ctx,
        Q->ten, K->ten, V->ten,
        mask,
        scale, max_bias, logit_softcap);

    return new pocket(mem, out, Q->dim);
}

inline void transferPocket(pocket *src, pocket *&dst)
{
    /*
        -> src pointer transfer to dst, and free dst original memory
    */
    delete dst;
    dst = src;
}

inline pocket *RMSNorm(pocket *inp, pocket *weight, float eps, ggml_backend *targetMem = nullptr)
{
    ggml_backend *mem = targetMem ? targetMem : inp->mem;
    pocket *normed = new pocket(inp->mem, ggml_rms_norm(mem->ctx, inp->ten, eps), inp->dim);

    pocket *out = mulTen(normed, weight, mem);

    delete normed;
    return out;
}