
llama_cpp installation:

```bash
conda install nvidia/label/cuda-12.4.1::cuda-toolkit -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --verbose
```

