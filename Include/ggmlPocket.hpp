#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <iostream>
#include <vector>
#include <cmath>

/*
    This class Not support the broadcasting for more then GGML_MAX_DIMS tensor.
    we can use dimension folding to make more then 5-dim tensor.
*/

enum device
{
    cpu,
    cuda
};

class ggml_backend
{
    /*
        Parameter:
            size: in mb
    */
public:
    struct ggml_context *ctx;
    ggml_backend_t backend;
    std::vector<ggml_backend_buffer_t> buffers;
    ggml_backend(int size, device type = cpu)
    {
        // Initialized memory to store graph (tensor metadata) information.
        struct ggml_init_params params = {
            .mem_size = (size_t)size * 1024 * 1024, // in Bytes
            .mem_buffer = NULL,                     // If we want to provide our own memory by malloc
            .no_alloc = true                        // if we don't want to allocate memory to tensor, just allocate meta data, and graph informations
        };
        ctx = ggml_init(params);

        if (type == cpu)
            backend = ggml_backend_cpu_init();
        else if (type == cuda)
            backend = ggml_backend_cuda_init(0);
    }

    ~ggml_backend()
    {
        for (auto buf : buffers) {
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
        }
        
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }

        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    void AllocMemory()
    {
        /*
            -> This function allocate memory if there is any tensor initialized with this ggml_backend
            -> All the meta data for grpah or tensor it has been allocated to memory initialized before
        */
        ggml_backend_buffer_t new_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        
        if (new_buffer != nullptr) {
            buffers.push_back(new_buffer);
        }
    }

    // Delete copy semantics to prevent double-freeing the ggml_context
    ggml_backend(const ggml_backend &) = delete;
    ggml_backend &operator=(const ggml_backend &) = delete;

private:
};

class pocket
{
    /*
        -> Used as a pocket to have everything at one place
    */
public:
    ggml_backend *mem;
    struct ggml_tensor *ten;
    size_t dim = 0, size;
    bool isNotContiguous = false;
    pocket(ggml_backend *_mem, ggml_tensor *_ten, size_t _dim)
        : mem(_mem), ten(_ten), dim(_dim), size(ggml_nelements(_ten))
    {
    }
    ~pocket()
    {        
    }

    void printShape()
    {
        std::cout << "{";
        for(int index : ten->ne)
            std::cout << index << " ";
        std::cout << "}" << std::endl;
    }
};

class Graph
{
    ggml_backend *mem;
    struct ggml_cgraph *graph;
    ggml_gallocr_t allocr;

public:
    Graph(ggml_backend *_mem) : mem(_mem)
    {
        graph = ggml_new_graph_custom(mem->ctx, 8192, false);
    }

    ~Graph()
    {
        if (allocr)
        {
            ggml_gallocr_free(allocr);
        }
    }

    void AddNode(pocket *tensor)
    {
        ggml_build_forward_expand(graph, tensor->ten);
    }

    void AddRemainingNode(std::vector<pocket *> requiredops)
    {
        for (auto node : requiredops)
            ggml_build_forward_expand(graph, node->ten);
    }

    void Compute()
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mem->backend));
        ggml_gallocr_alloc_graph(allocr, graph);
        ggml_backend_graph_compute(mem->backend, graph);
    }
};