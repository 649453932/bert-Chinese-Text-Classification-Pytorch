#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

using namespace std;

const static std::vector<std::string> key = {                                                                                                                                                                                                                     
    "finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment"
};

template <typename T>
int argmax(T a, T b) {
    return std::max_element(a, b) - a;
}

class Predictor {
public:
    ~Predictor() {
        delete ses_;
        delete tokenizer_;
    }

    Predictor(const std::string& model_path, const std::string& vocab_path) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;

        OrtCUDAProviderOptions cuda_options; //= {
    //          0,
    //          //OrtCudnnConvAlgoSearch::EXHAUSTIVE,
    //          OrtCudnnConvAlgoSearchExhaustive,
    //          std::numeric_limits<size_t>::max(),
    //          0,
    //          true
    //      };

        session_options.AppendExecutionProvider_CUDA(cuda_options);
        ses_ = new Ort::Session(env, model_path, session_options);
        auto& session = *ses_;
        size_t num_input_nodes = session.GetInputCount();
        std::cout<< num_input_nodes <<std::endl;
        std::cout<< session.GetOutputCount() <<std::endl;

        tokenizer_ = new FullTokenizer(vocab_path);
    }

    std::vector<Ort::Value> build_input(const std::string& text) {
        auto tokens = tokenizer.tokenize(text);
        auto ids = tokenizer.convertTokensToIds(tokens);
        std::vector<int64_t> mask;
        ids.resize(pad_size_);
        mask.resize(pad_size_);

        for (int i = 0; i < pad_size_; ++i) {
            if (0 == ids[i]) {
                break;
            }
            mask[i] = 1;
            //mask[i] = ids[i] > 0;
            std::cout << ids[i] << "\t" << std::endl;
        }

        std::vector<int64_t> input_node_dims = {1, 32};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, ids.data(),
                                                                pad_size_, input_node_dims.data(), 2);

        Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask.data(),
                                                                pad_size_, input_node_dims.data(), 2);

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(mask_tensor));
        return ort_inputs;
    }

    size_t predict(const std::string& text) {
        auto ort_inputs = build_input(text);
        std::vector<const char*> input_node_names = {"ids", "mask"};
        std::vector<const char*> output_node_names = {"output"};

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                        ort_inputs.size(), output_node_names.data(), 1);

        float* output = output_tensors[0].GetTensorMutableData<float>();

        return argmax(output, outpus+10);
    }

private:
    Ort::session* ses_ = nullptr;
    FullTokenizer tokenizer_ = nullptr;
    int pad_size_ = 32;
};

int main()
{
    const char* model_path = "/home/guodong/github/Bert-Chinese-Text-Classification-Pytorch/model.onnx";
    const char* vocab_path = "/home/guodong/bert_pretrain/vocab.txt";

    Predictor pred(model_path, vocab_path);

    std::string line;
    while (std::getline(std::cin, line)) {
        int idx = pred.predict(line);
        std::cout << line << " is " << key[idx] << std::endl;
    }

    return 0;
}
