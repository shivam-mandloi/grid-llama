#pragma once

#include "ggml.hpp"

#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>

void CreateSinAndCos(std::vector<float> &theta,
                     pocket *sinTheta,
                     pocket *cosTheta,
                     int crnSeqLen,
                     int numOfNewEle)
{
    // Create using gemini
    /*
        -> Used by llama class to calculate RoPE frequencies
        -> Dynamically updates the pre-allocated GGML tensors (sinTheta, cosTheta)
        -> Calculates values for `numOfNewEle` and patches them at the `crnSeqLen` offset
    */
    int d = theta.size();

    // 1. Create temporary CPU staging buffers for the new elements
    int totalNewElements = numOfNewEle * d;
    std::vector<float> tempSin(totalNewElements);
    std::vector<float> tempCos(totalNewElements);

    // 2. Calculate the values for the new sequence positions
    for (int i = 0; i < numOfNewEle; i++)
    {
        int seqPos = crnSeqLen + i; // The absolute position in the sequence
        for (int j = 0; j < d; j++)
        {
            float val = seqPos * theta[j];
            tempSin[i * d + j] = std::sin(val);
            tempCos[i * d + j] = std::cos(val);
        }
    }

    // 3. Calculate the memory offset and size in bytes
    // size of F32 is 4 bytes. We skip 'crnSeqLen * d' elements to insert at the correct row.
    size_t offsetInBytes = (size_t)crnSeqLen * d * sizeof(float);
    size_t sizeInBytes = (size_t)totalNewElements * sizeof(float);

    // 4. Push the staging buffers to the GGML backend (CPU or CUDA)
    ggml_backend_tensor_set(sinTheta->ten, tempSin.data(), offsetInBytes, sizeInBytes);
    ggml_backend_tensor_set(cosTheta->ten, tempCos.data(), offsetInBytes, sizeInBytes);
}

pocket *getEmbeddings(pocket *embd, const std::vector<int> &tokens,
                      ggml_backend *graphMem, // for metadata/graph ops
                      ggml_backend *dataMem)  // for actual input data
{
    pocket *inpIndex = InitPocket({static_cast<int64_t>(tokens.size())}, dataMem, GGML_TYPE_I32);
    dataMem->AllocMemory(); // only allocates inpIndex, small & immediate

    size_t sizeInBytes = tokens.size() * sizeof(int);
    ggml_backend_tensor_set(inpIndex->ten, tokens.data(), 0, sizeInBytes);

    // get_rows op is a graph node, lives in graphMem's ctx
    struct ggml_tensor *res_tensor = ggml_get_rows(graphMem->ctx, embd->ten, inpIndex->ten);

    pocket *outEmbd = new pocket(graphMem, res_tensor, 2);
    delete inpIndex; // just pocket metadata; dataMem owns the buffer
    return outEmbd;
}

int SelectIndex(pocket *logits, float temperature = 0.7f, float top_p = 0.9f)
{
    int vocab = logits->ten->ne[0];
    int s = logits->ten->ne[1];

    std::vector<float> cpu_logits(vocab * s);
    ggml_backend_tensor_get(logits->ten, cpu_logits.data(), 0, vocab * s * sizeof(float));

    float *last = cpu_logits.data() + (s - 1) * vocab;

    for (int i = 0; i < vocab; i++)
        last[i] /= temperature;

    float maxVal = *std::max_element(last, last + vocab);
    float sum = 0.0f;
    std::vector<float> probs(vocab);
    for (int i = 0; i < vocab; i++)
    {
        probs[i] = std::exp(last[i] - maxVal);
        sum += probs[i];
    }
    for (int i = 0; i < vocab; i++)
        probs[i] /= sum;

    std::vector<int> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    int cutoff = vocab;
    for (int i = 0; i < vocab; i++)
    {
        cumsum += probs[indices[i]];
        if (cumsum >= top_p)
        {
            cutoff = i + 1;
            break;
        }
    }

    // 5. Renormalize over the kept tokens
    float keptSum = 0.0f;
    for (int i = 0; i < cutoff; i++)
        keptSum += probs[indices[i]];

    // 6. Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, keptSum);
    float r = dist(gen);

    float acc = 0.0f;
    for (int i = 0; i < cutoff; i++)
    {
        acc += probs[indices[i]];
        if (acc >= r)
            return indices[i];
    }
    return indices[0]; // fallback
}

class tokenizer
{

public:
    std::string spaceEnc = "Ġ", newLine = "Ċ";
    std::vector<std::string> vocabArr;
    std::unordered_map<std::string, int> mergeMap, vocabMap;
    tokenizer()
    {
        // Read vocab and merge file
        std::string vocabStr = ReadTxtFile("vocabCPP.txt");
        std::string mergeStr = ReadTxtFile("mergesCPP.txt");

        // Make vocab and merge vector
        vocabArr = SplitString(vocabStr, " ");
        std::vector<std::string> mergeArr = SplitString(mergeStr, "<^>");

        // Create map for faster access to elements
        for (int i = 0; i < mergeArr.size(); i++)
            mergeMap[mergeArr[i]] = i;
        for (int i = 0; i < vocabArr.size(); i++)
            vocabMap[vocabArr[i]] = i;
    }

    std::vector<int> encode(std::string text)
    {
        std::vector<std::string> splitedText;

        // split text by char and replace space and newLine by the Llama token
        for (int i = 0; i < text.size(); i++)
        {
            if (text[i] == ' ')
                splitedText.push_back(spaceEnc);
            else if (text[i] == '\n')
                splitedText.push_back(newLine);
            else
                splitedText.emplace_back(1, text[i]);
        }

        // loop until there is not any char left to merge
        while (true)
        {
            int mergeIndex = mergeMap.size(), index = -1;

            // Find two element which can be merged and have high priority
            for (int i = 0; i < splitedText.size() - 1; i++)
            {
                std::string tempStr = splitedText.at(i) + " " + splitedText.at(i + 1);
                if (mergeMap.find(tempStr) != mergeMap.end())
                {
                    if (mergeIndex > mergeMap[tempStr])
                    {
                        mergeIndex = mergeMap[tempStr];
                        index = i;
                    }
                }
            }

            // If there is not any merge left
            if (index == -1)
                break;

            std::vector<std::string> tempText;
            int i = 0;
            std::string targetString = splitedText.at(index) + " " + splitedText.at(index + 1);

            // merge all the token
            while (i < splitedText.size() - 1)
            {
                if (splitedText.at(i) == splitedText.at(index) && splitedText.at(i + 1) == splitedText.at(index + 1))
                {
                    tempText.push_back(splitedText.at(i) + splitedText.at(i + 1));
                    i++;
                }
                else
                    tempText.push_back(splitedText.at(i));
                i++;
            }
            if (i == splitedText.size() - 1)
                tempText.push_back(splitedText.at(i));
            splitedText = std::move(tempText);
        }

        std::vector<int> res(splitedText.size(), 0);

        // Convert to token to index
        for (int i = 0; i < splitedText.size(); i++)
            res[i] = vocabMap[splitedText[i]];
        return res;
    }

    std::string decode(std::vector<int> encd)
    {
        std::string decdString = "";
        for (int i = 0; i < encd.size(); i++)
            decdString += vocabArr[encd[i]];

        size_t pos;

        while ((pos = decdString.find("Ġ")) != std::string::npos)
            decdString.replace(pos, 2, " ");

        while ((pos = decdString.find("Ċ")) != std::string::npos)
            decdString.replace(pos, 2, "\n");
        return decdString;
    }
};

class Attention
{

public:
    pocket *query, *key, *value, *output, *kCache, *vCache;
    int qkvDim = 128; // for 3B
    float qkvDimRoot = 1.0 / std::sqrt(qkvDim);
    Attention(int index, std::vector<int64_t> querySize,
              std::vector<int64_t> keySize,
              std::vector<int64_t> valueSize,
              std::vector<int64_t> outputSize,
              ggml_backend *_weightMem,
              int batchSize,
              int maxSeqLen,
              ggml_type weightType)
    {
        query = InitPocket(querySize, _weightMem, weightType);
        key = InitPocket(keySize, _weightMem, weightType);
        value = InitPocket(valueSize, _weightMem, weightType);
        output = InitPocket(outputSize, _weightMem, weightType);

        // kCache and vChache has GGML_TYPE_F32 type, for easy calculation
        kCache = InitPocket({128, 8, maxSeqLen, batchSize}, _weightMem, GGML_TYPE_F32);
        vCache = InitPocket({128, 8, maxSeqLen, batchSize}, _weightMem, GGML_TYPE_F32);
    }

    void LoadParameter(int index)
    {
        // Assume memory has been allocated
        // torch.tensor(3072 X 3072) -> ggml(3072, 3072)
        Loadpocket("model.layers." + std::to_string(index) + ".self_attn.q_proj.weight.bin", query);
        // torch.tensor(1024 X 3072) -> ggml(3072, 1024)
        Loadpocket("model.layers." + std::to_string(index) + ".self_attn.k_proj.weight.bin", key);
        // torch.tensor(1024 X 3072) -> ggml(3072, 1024)
        Loadpocket("model.layers." + std::to_string(index) + ".self_attn.v_proj.weight.bin", value);
        // torch.tensor(3072 X 3072) -> ggml(3072, 3072)
        Loadpocket("model.layers." + std::to_string(index) + ".self_attn.o_proj.weight.bin", output);
    }

    pocket *forward(pocket *inp, pocket *sinTheta, pocket *cosTheta, int64_t crntSeqSize, std::vector<pocket *> &requiredOps)
    {
        /*
            (3072 X 1 X b)
            n = 3072
            x: (n X s X b) | (n X 1 X b)
            sinTheta: ((qkvDim/2) X MaxSeqSize)
            cosTheta:((qkvDim/2) X MaxSeqSize)
            crntSeqSize: current sequence size
            requiredOps: This make sure that every node add on ggml graph are added, I push all the node which can be missed while creating graph, all the element in this vector will be added forcefully by "ggml_build_forward_expand" this function.

            sinTheta and cosTheta have MAX_SEQ_LEN
        */

        // Get the bath, seqLen and dim
        int64_t b = inp->ten->ne[2], s = inp->ten->ne[1], n = inp->ten->ne[0];

        // inp: (n X s X b)
        // weight: (n X (head * qkvDim))
        // Out Dim: (head * qkvDim) X s X b

        pocket *q = matmul(inp, query), *k = matmul(inp, key), *v = matmul(inp, value);

        // qkvDim X head X s X b
        transferPocket(reshape(q, {qkvDim, 24, s, b}), q);
        transferPocket(reshape(k, {qkvDim, 8, s, b}), k);
        transferPocket(reshape(v, {qkvDim, 8, s, b}), v);

        // Dim: (qkvDim / 2) X 2 X head X (b*s)
        transferPocket(reshape(q, {qkvDim / 2, 2, 24, b * s}), q);
        transferPocket(reshape(k, {qkvDim / 2, 2, 8, b * s}), k);

        /*
            ROPE Implementation
        */
        // Get the right sin and cos
        pocket *sin = slice(sinTheta, {0, crntSeqSize}, {(qkvDim / 2), crntSeqSize + s}, inp->mem);
        pocket *cos = slice(cosTheta, {0, crntSeqSize}, {(qkvDim / 2), crntSeqSize + s}, inp->mem);

        transferPocket(reshape(sin, {static_cast<int64_t>(qkvDim / 2), 1, s, 1}), sin);
        transferPocket(reshape(cos, {static_cast<int64_t>(qkvDim / 2), 1, s, 1}), cos);
        // ROPE for the query

        // Dim: (qkvDim/2) X 1 X 24 X b*s
        pocket *x = slice(q, {0, 0, 0, 0}, {(qkvDim / 2), 1, 24, (b * s)});
        pocket *y = slice(q, {0, 1, 0, 0}, {(qkvDim / 2), 2, 24, (b * s)});

        transferPocket(reshape(x, {(qkvDim / 2), 24, s, b}), x);
        transferPocket(reshape(y, {(qkvDim / 2), 24, s, b}), y);

        // put sin/cos as second parameter for the broadcast
        // newX = x * tCos - y * tSin;
        pocket *mSin = mulTen(y, sin);
        pocket *mCos = mulTen(x, cos);
        pocket *xNew = subTen(mCos, mSin);

        // newY = x * tSin + y * tCos;
        transferPocket(mulTen(x, sin), mSin);
        transferPocket(mulTen(y, cos), mCos);
        pocket *yNew = addTen(mSin, mCos);
        // Dim: (qkvDim / 2) X 2 X head X (b*s)
        // Out Dim: qkvDim X head X s X b

        transferPocket(reshape(q, {qkvDim, 24, s, b}), q);

        // first half of query replaced with x and second by y
        // q Out Dim: qkvDim X 24 X s X b

        // copypocket(q, xNew, {0, 0, 0, 0}, {(qkvDim / 2), 24, s, b}, requiredOps);
        // copypocket(q, yNew, {(qkvDim / 2), 0, 0, 0}, {qkvDim, 24, s, b}, requiredOps);
        transferPocket(new pocket(q->mem, ggml_concat(q->mem->ctx, xNew->ten, yNew->ten, 0), q->dim), q);

        // ROPE for the key

        // Dim: (qkvDim/2) X 1 X 8 X b*s
        transferPocket(slice(k, {0, 0, 0, 0}, {(qkvDim / 2), 1, 8, (b * s)}), x);
        transferPocket(slice(k, {0, 1, 0, 0}, {(qkvDim / 2), 2, 8, (b * s)}), y);

        // Dim: (qkvDim/2) X 8 X b X s
        transferPocket(reshape(x, {(qkvDim / 2), 8, s, b}), x);
        transferPocket(reshape(y, {(qkvDim / 2), 8, s, b}), y);

        // put sin/cos as second parameter for the broadcast
        transferPocket(mulTen(y, sin), mSin);
        transferPocket(mulTen(x, cos), mCos);
        transferPocket(subTen(mCos, mSin), xNew);

        // newY = x * tSin + y * tCos;
        transferPocket(mulTen(x, sin), mSin);
        transferPocket(mulTen(y, cos), mCos);
        transferPocket(addTen(mSin, mCos), yNew);

        transferPocket(reshape(k, {qkvDim, 8, s, b}), k);
        // first half of query replaced with x and second by y
        // k Out Dim: qkvDim X 8 X s X b
        transferPocket(new pocket(k->mem, ggml_concat(k->mem->ctx, xNew->ten, yNew->ten, 0), k->dim), k);
        /*
            Rope Done
        */
        // // Update kCache and vCache
        requiredOps.push_back(copypocket(kCache, k, {0, 0, crntSeqSize, 0}, {qkvDim, 8, crntSeqSize + s, b}, inp->mem));
        requiredOps.push_back(copypocket(vCache, v, {0, 0, crntSeqSize, 0}, {qkvDim, 8, crntSeqSize + s, b}, inp->mem));

        // k/v Dim: qkvDim X 8 X (crntSeqSize + s) X b
        transferPocket(slice(kCache, {0, 0, 0, 0}, {qkvDim, 8, crntSeqSize + s, b}, k->mem), k);
        transferPocket(slice(vCache, {0, 0, 0, 0}, {qkvDim, 8, crntSeqSize + s, b}, v->mem), v);

        // Flash attention require d X s X h X b
        // qkvDim X 24 X s X b -> qkvDim X s X 24 X b
        transferPocket(permute(q, {0, 2, 1, 3}), q);

        // b X (crntSeqSize + s) X 8 X qkvDim -> b X 8 X (crntSeqSize + s) X qkvDim
        // qkvDim X 8 X (crntSeqSize + s) X b -> qkvDim X (crntSeqSize + s) X 8 X b
        transferPocket(permute(k, {0, 2, 1, 3}), k);

        // qkvDim X 8 X (crntSeqSize + s) X b -> qkvDim X (crntSeqSize + s) X 8 X b
        transferPocket(permute(v, {0, 2, 1, 3}), v);

        // Out: qkvDim X 24 X s X b
        pocket *resTensor = flashAttention(q, k, v, crntSeqSize, s, qkvDimRoot);


        // qkvDim X 24 X s X b -> qkvDim*24 X s X b
        // transferPocket(permute(resTensor, {0, 2, 1, 3}), resTensor); // The goat error took me 3 days to find
        transferPocket(reshape(resTensor, {qkvDim * 24, s, b}), resTensor);

        pocket *finalOut = matmul(resTensor, output);

        delete mSin;
        delete mCos;
        delete xNew;
        delete yNew;

        delete q;
        delete k;
        delete v;
        delete sin;
        delete cos;
        delete x;
        delete y;
        delete resTensor;

        return finalOut;
    }
};

class MLP
{
    pocket *down, *up, *gate;

public:
    MLP(std::vector<int64_t> downSize,
        std::vector<int64_t> upSize,
        std::vector<int64_t> gateSize,
        ggml_backend *_weightMem,
        ggml_type type)
    {
        down = InitPocket(downSize, _weightMem, type);
        up = InitPocket(upSize, _weightMem, type);
        gate = InitPocket(gateSize, _weightMem, type);
    }

    void LoadParameter(int index)
    {
        // [8192, n]
        Loadpocket("model.layers." + std::to_string(index) + ".mlp.down_proj.weight.bin", down);
        // [n, 8192]
        Loadpocket("model.layers." + std::to_string(index) + ".mlp.up_proj.weight.bin", up);
        // [n, 8192]
        Loadpocket("model.layers." + std::to_string(index) + ".mlp.gate_proj.weight.bin", gate);
    }

    pocket *forward(pocket *inp)
    {
        /*
            x: n X s X b
            b = batch size | s = sequence | n = input dim
        */

        // out: 8192 X s X b
        pocket *gateOut = matmul(inp, gate);
        pocket *upOut = matmul(inp, up);

        pocket *siluOut = new pocket(inp->mem, ggml_silu(inp->mem->ctx, gateOut->ten), inp->dim);

        // SwiGLU: SiLU(gate) * up
        transferPocket(mulTen(siluOut, upOut), siluOut);

        pocket *res = matmul(siluOut, down);

        delete gateOut;
        delete upOut;
        delete siluOut;

        return res;
    }
};

class Transformer
{

public:
    Attention *attn;
    MLP *mlp;
    pocket *inputRMSNorm, *postAttnRMSNorm;
    Transformer(int index,
                std::vector<int64_t> querySize,
                std::vector<int64_t> keySize,
                std::vector<int64_t> valueSize,
                std::vector<int64_t> outputSize,
                std::vector<int64_t> downSize,
                std::vector<int64_t> upSize,
                std::vector<int64_t> gateSize,
                std::vector<int64_t> inputRMSNormSize,
                std::vector<int64_t> postAttnRMSNormSize,
                int maxBatchSize,
                int maxSeqLen,
                ggml_backend *weightMem,
                ggml_type weightType)
    {
        attn = new Attention(index, querySize, keySize, valueSize, outputSize, weightMem, maxBatchSize, maxSeqLen, weightType);
        mlp = new MLP(downSize, upSize, gateSize, weightMem, weightType);

        // Elementwise multiplication in ggml not work for quantized weight
        inputRMSNorm = InitPocket(inputRMSNormSize, weightMem, GGML_TYPE_F32);
        postAttnRMSNorm = InitPocket(postAttnRMSNormSize, weightMem, GGML_TYPE_F32);
    }

    void LoadParameter(int index)
    {
        Loadpocket("model.layers." + std::to_string(index) + ".input_layernorm.weight.bin", inputRMSNorm);
        Loadpocket("model.layers." + std::to_string(index) + ".post_attention_layernorm.weight.bin", postAttnRMSNorm);
        attn->LoadParameter(index);
        mlp->LoadParameter(index);
    }

    // change the input iteself
    void forward(pocket *&inp,
                 pocket *sinTheta,
                 pocket *cosTheta,
                 int64_t crntSeqSize,
                 std::vector<pocket *> &requiredOps)
    {
        /*
            inp: n X s X b
            b = batch size | s = sequence | n = input dim
        */

        pocket *attInp = RMSNorm(inp, inputRMSNorm, 1e-5);
        transferPocket(attn->forward(attInp, sinTheta, cosTheta, crntSeqSize, requiredOps), attInp);
        // attInp->printShape();
        transferPocket(addTen(attInp, inp), inp);
        // std::cout << "start transformer post rms" << std::endl;
        transferPocket(RMSNorm(inp, postAttnRMSNorm, 1e-5), attInp);

        // std::cout << "start mlp " << std::endl;
        transferPocket(mlp->forward(attInp), attInp);
        transferPocket(addTen(inp, attInp), inp);
        delete attInp;
    }
};

class llama
{
    pocket *sinTheta, *cosTheta, *embd, *remsNorm;
    int qkvDim = 128, crnSeqLen = 0, resetSeqLen, maxSeqLen = 0, lastGeneratedEle = -1;
    float ropeTheta = 500000.0;
    std::vector<Transformer> layer;
    tokenizer tkn;
    std::vector<float> theta;
    Graph *grp;
    ggml_backend *weightMem, *inputMem, *graphMem;
    device dvc;

public:
    llama(int batchSize, int _maxSeqLen, std::unordered_map<std::string, std::string> cmnd, device _dvc = cpu) : maxSeqLen(_maxSeqLen), dvc(_dvc)
    {
        /*
            -> Need to update for batch size more then 1
            cmnd = {"header": "header of the model", "text": "question or system command"}
        */
        theta = std::vector<float>(qkvDim / 2, 0);
        for (int i = 0; i < qkvDim / 2; i++)
            theta[i] = pow(ropeTheta, (-2.0f * i) / qkvDim);

        weightMem = new ggml_backend(10, dvc); // 10MB

        /*
            -> Define only for the llama 3 (3B)
        */
        std::vector<int64_t> queryOutSize = {3072, 3072},
                             keyValueSize = {3072, 1024},
                             downSize = {8192, 3072},
                             upSize = {3072, 8192},
                             gateSize = {3072, 8192},
                             inputPostRMSNormSize = {3072},
                             embdSize = {3072, 128256};

        // RMS norm use elementwise multiplication, and ggml element wise multiplication not support quantized version of tensor's
        remsNorm = InitPocket(inputPostRMSNormSize, weightMem, GGML_TYPE_F32);
        embd = InitPocket(embdSize, weightMem, GGML_TYPE_F32);
        sinTheta = InitPocket({qkvDim / 2, maxSeqLen}, weightMem, GGML_TYPE_F32);
        cosTheta = InitPocket({qkvDim / 2, maxSeqLen}, weightMem, GGML_TYPE_F32);

        for (int i = 0; i < 28; i++)
        {
            layer.emplace_back(i,
                               queryOutSize,
                               keyValueSize,
                               keyValueSize,
                               queryOutSize,
                               downSize,
                               upSize,
                               gateSize,
                               inputPostRMSNormSize,
                               inputPostRMSNormSize,
                               batchSize,
                               maxSeqLen,
                               weightMem,
                               GGML_TYPE_Q4_0);
        }

        // allocate memory
        weightMem->AllocMemory();

        std::cout << "[*] Allocate memory, start loadding weights" << std::endl;

        Loadpocket("model.embed_tokens.weight.bin", embd);
        Loadpocket("model.norm.weight.bin", remsNorm);

        CreateSinAndCos(theta, sinTheta, cosTheta, 0, maxSeqLen);
        for (int i = 0; i < 28; i++)
        {
            layer[i].LoadParameter(i);
        }

        std::cout << "[*] Parameter loaded" << std::endl;

        // Only call when
        if (cmnd["text"].size() != 0)
        {
            // std::cout << "check " << 1 << std::endl;
            graphMem = new ggml_backend(50, dvc); // 10MB
            inputMem = new ggml_backend(10, dvc);
            grp = new Graph(graphMem);
            std::vector<pocket *> requiredOps;

            std::vector<int> index = tkn.encode(cmnd["text"]);

            /*
                {
                    "<|end_header_id|>": 128007,
                    "<|start_header_id|>": 128006,
                    "<|eot_id|>": 128009,
                    "<|begin_of_text|>": 128000,
                    "\n\n": 271
                }
            */
            int headerId = 0;
            if (cmnd["header"] == "system")
                headerId = 9125;
            else if (cmnd["header"] == "user")
                headerId = 882;
            else if (cmnd["header"] == "assistant")
                headerId = 78191;
            else
                headerId = tkn.vocabMap[cmnd["header"]];
            std::vector<int> extraHeaders = {128000, 128006, headerId, 128007, 271};
            index.insert(index.begin(), extraHeaders.begin(), extraHeaders.end());
            index.push_back(128009);
            crnSeqLen += index.size();

            if (crnSeqLen >= maxSeqLen)
            {
                std::cerr << "[!] You can't just put more then " << maxSeqLen << " your current fucking sequence lenght is " << crnSeqLen << std::endl;
                exit(0);
            }

            // std::cout << "check " << 2 << std::endl;
            pocket *inp = getEmbeddings(embd, index, graphMem, inputMem);

            // include batch
            transferPocket(reshape(inp, {3072, static_cast<int64_t>(index.size()), 1}), inp);

            // std::cout << "check " << 3 << std::endl;
            for (int i = 0; i < 28; i++)
            {
                layer[i].forward(inp, sinTheta, cosTheta, 0, requiredOps);
            }
            // std::cout << "check " << 4 << std::endl;
            grp->AddRemainingNode(requiredOps);
            grp->AddNode(inp);

            grp->Compute();

            // for each iteration we need to create new graph, because input, cache will change.
            delete inputMem;
            delete graphMem;
            delete grp;
            delete inp;
            for (auto node : requiredOps)
                delete node;
        }
        resetSeqLen = crnSeqLen;
        std::cout << "[*] Done with prompt" << std::endl;
    }

    void ask(std::unordered_map<std::string, std::string> que)
    {
        /*
            que = {"header": "header of the model", "text": "question or system command"}
        */
        inputMem = new ggml_backend(10, dvc); // 1MB
        graphMem = new ggml_backend(50, dvc);
        grp = new Graph(graphMem);

        std::vector<int> index = tkn.encode(que["text"]);

        /*
            {
                "<|end_header_id|>": 128007,
                "<|start_header_id|>": 128006,
                "<|eot_id|>": 128009,
                "<|begin_of_text|>": 128000,
                "\n\n": 271
            }
        */
        int headerId = 0;
        if (que["header"] == "system")
            headerId = 9125;
        else if (que["header"] == "user")
            headerId = 882;
        else if (que["header"] == "assistant")
            headerId = 78191;
        else
            headerId = tkn.vocabMap[que["header"]];
        std::vector<int> extraHeaders = {128006, headerId, 128007, 271};

        index.insert(index.begin(), extraHeaders.begin(), extraHeaders.end());
        index.push_back(128009);
        index.push_back(128006); // <|start_header_id|>
        index.push_back(78191);  // The hardcoded ID for "assistant"
        index.push_back(128007); // <|end_header_id|>
        index.push_back(271);

        std::vector<pocket *> requiredOps;

        if (crnSeqLen + index.size() >= maxSeqLen)
        {
            std::cerr << "[!] You can't just put more then " << maxSeqLen << " your current fucking sequence lenght is " << crnSeqLen << std::endl;
            exit(1);
        }

        pocket *inp = getEmbeddings(embd, index, graphMem, inputMem);
        transferPocket(reshape(inp, {3072, static_cast<int64_t>(index.size()), 1}), inp);
        for (int i = 0; i < 28; i++)
        {
            layer[i].forward(inp, sinTheta, cosTheta, crnSeqLen, requiredOps);
        }
        transferPocket(RMSNorm(inp, remsNorm, 1e-5), inp);
        transferPocket(matmul(inp, embd), inp);

        grp->AddRemainingNode(requiredOps);
        grp->AddNode(inp);

        grp->Compute();

        int nextEle = SelectIndex(inp);
        crnSeqLen += index.size();
        lastGeneratedEle = nextEle;

        delete grp;
        delete inp;
        delete inputMem;
        delete graphMem;
        for (auto node : requiredOps)
            delete node;
    }

    std::string GetNext()
    {
        if (crnSeqLen >= maxSeqLen)
        {
            std::cerr << "[!] You can't just put more then " << maxSeqLen << " your current fucking sequence lenght is " << crnSeqLen << std::endl;
            return "endOfText"; // Stop generation;
        }

        inputMem = new ggml_backend(10, dvc); // 1MB
        graphMem = new ggml_backend(50, dvc);
        grp = new Graph(graphMem);
        std::vector<pocket *> requiredOps;

        if (lastGeneratedEle == 128001 || lastGeneratedEle == 128009)
            return "endOfText";

        std::string out = tkn.decode({lastGeneratedEle});
        pocket *inp = getEmbeddings(embd, {lastGeneratedEle}, graphMem, inputMem);
        transferPocket(reshape(inp, {3072, 1, 1}), inp);

        for (int i = 0; i < 28; i++)
        {
            layer[i].forward(inp, sinTheta, cosTheta, crnSeqLen, requiredOps);
        }
        transferPocket(RMSNorm(inp, remsNorm, 1e-5), inp);
        transferPocket(matmul(inp, embd), inp);

        grp->AddRemainingNode(requiredOps);
        grp->AddNode(inp);

        grp->Compute();

        int nextEle = SelectIndex(inp);
        crnSeqLen += 1;
        lastGeneratedEle = nextEle;

        delete grp;
        delete inp;
        delete inputMem;
        delete graphMem;
        for (auto node : requiredOps)
            delete node;

        return out;
    }

    void reset()
    {
        crnSeqLen = resetSeqLen;
    }
};