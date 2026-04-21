/*
 * flow_model.hpp
 * --------------
 * ResNet-18 model adapted for optical flow input, implemented with LibTorch.
 * Also defines the UCF101FlowDataset class.
 *
 * Input:  stacked u/v flow maps  →  Tensor (2 * NUM_FRAMES, H, W)
 * Output: action class logits    →  Tensor (num_classes,)
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace fs = std::filesystem;

// ── Config ───────────────────────────────────────────────────────────────────
// Implementation note: both u and v channels are stacked for each of the
// NUM_FRAMES sampled flow maps, yielding 2*NUM_FRAMES = 32 input channels.
// The paper (§III.B) refers to "10-channel flow magnitude maps" as a simplified
// description; using u+v retains directional information and is standard in the
// two-stream literature.
constexpr int NUM_FRAMES   = 16;
constexpr int NUM_CLASSES  = 101;
constexpr int IMG_H        = 224;
constexpr int IMG_W        = 224;
// ─────────────────────────────────────────────────────────────────────────────


// ── ResNet-18 building blocks ─────────────────────────────────────────────────

/*
 * BasicBlock
 * ----------
 * Standard ResNet basic block: two 3x3 convolutions with a residual connection.
 */
struct BasicBlockImpl : torch::nn::Module {
    torch::nn::Conv2d     conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},   bn2{nullptr};
    torch::nn::Sequential  downsample{nullptr};

    BasicBlockImpl(int in_ch, int out_ch, int stride = 1) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3)
                              .stride(stride).padding(1).bias(false)));
        bn1   = register_module("bn1", torch::nn::BatchNorm2d(out_ch));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3)
                              .stride(1).padding(1).bias(false)));
        bn2   = register_module("bn2", torch::nn::BatchNorm2d(out_ch));

        // Downsample when spatial size or channel count changes
        if (stride != 1 || in_ch != out_ch) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 1)
                                  .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_ch)
            ));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        if (downsample) identity = downsample->forward(identity);
        return torch::relu(x + identity);
    }
};
TORCH_MODULE(BasicBlock);


/*
 * FlowResNet18
 * ------------
 * ResNet-18 with the first conv layer replaced to accept (2 * NUM_FRAMES)
 * input channels (stacked u and v optical flow maps).
 */
struct FlowResNet18Impl : torch::nn::Module {
    torch::nn::Conv2d      conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential  layer1{nullptr}, layer2{nullptr},
                           layer3{nullptr}, layer4{nullptr};
    torch::nn::Linear      fc{nullptr};

    FlowResNet18Impl(int numFrames = NUM_FRAMES, int numClasses = NUM_CLASSES) {
        int in_ch = 2 * numFrames;

        // First conv: accepts stacked flow channels instead of RGB
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, 64, 7)
                              .stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        // ResNet layers
        layer1 = register_module("layer1", makeLayer(64,  64,  2, 1));
        layer2 = register_module("layer2", makeLayer(64,  128, 2, 2));
        layer3 = register_module("layer3", makeLayer(128, 256, 2, 2));
        layer4 = register_module("layer4", makeLayer(256, 512, 2, 2));

        fc = register_module("fc", torch::nn::Linear(512, numClasses));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));
        x = torch::max_pool2d(x, 3, 2, 1);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = torch::adaptive_avg_pool2d(x, {1, 1});
        x = x.flatten(1);   // (B, 512)
        return fc(x);        // (B, num_classes)
    }

private:
    // Helper: build a ResNet layer with `numBlocks` BasicBlocks
    torch::nn::Sequential makeLayer(int in_ch, int out_ch,
                                    int numBlocks, int stride) {
        torch::nn::Sequential seq;
        seq->push_back(BasicBlock(in_ch, out_ch, stride));
        for (int i = 1; i < numBlocks; ++i)
            seq->push_back(BasicBlock(out_ch, out_ch, 1));
        return seq;
    }
};
TORCH_MODULE(FlowResNet18);


// ── Dataset ───────────────────────────────────────────────────────────────────

/*
 * UCF101FlowDataset
 * -----------------
 * Loads pre-extracted optical flow maps (.yml) for UCF-101.
 *
 * Each sample returns:
 *   data   : FloatTensor of shape (2 * NUM_FRAMES, H, W)
 *   target : LongTensor scalar (class index)
 */
class UCF101FlowDataset : public torch::data::Dataset<UCF101FlowDataset> {
public:
    struct Sample {
        fs::path flowDir;
        int      label;
    };

    UCF101FlowDataset(const std::string& flowRoot,
                      const std::string& splitFile,
                      const std::string& classIndexFile,
                      int numFrames = NUM_FRAMES)
        : flowRoot_(flowRoot), numFrames_(numFrames)
    {
        // Parse classInd.txt: "<1-based index> <ClassName>"
        std::ifstream cls(classIndexFile);
        std::string line;
        while (std::getline(cls, line)) {
            std::istringstream ss(line);
            int idx; std::string name;
            ss >> idx >> name;
            classToIdx_[name] = idx - 1;   // 0-based
        }

        // Parse split file: "ClassName/v_xxx.avi [label]"
        std::ifstream sf(splitFile);
        while (std::getline(sf, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string relPath;
            ss >> relPath;

            fs::path p(relPath);
            std::string className = p.parent_path().string();
            std::string clipStem  = p.stem().string();
            int label = classToIdx_.at(className);

            fs::path flowDir = fs::path(flowRoot) / className / clipStem;
            if (fs::exists(flowDir))
                samples_.push_back({flowDir, label});
        }

        std::cout << "  Loaded " << samples_.size() << " clips from " << splitFile << "\n";
    }

    // Return one sample as {data tensor, label tensor}
    torch::data::Example<> get(size_t idx) override {
        const Sample& s = samples_[idx];

        // Collect sorted .yml files
        std::vector<fs::path> flowFiles;
        for (const auto& entry : fs::directory_iterator(s.flowDir)) {
            if (entry.path().extension() == ".yml")
                flowFiles.push_back(entry.path());
        }
        std::sort(flowFiles.begin(), flowFiles.end());

        // Sample or pad to numFrames_
        std::vector<fs::path> selected;
        int n = static_cast<int>(flowFiles.size());

        if (n >= numFrames_) {
            for (int k = 0; k < numFrames_; ++k) {
                int i = static_cast<int>(std::round(k * (n - 1.0) / (numFrames_ - 1)));
                selected.push_back(flowFiles[i]);
            }
        } else {
            selected = flowFiles;
            while (static_cast<int>(selected.size()) < numFrames_)
                selected.push_back(flowFiles.back());
        }

        // Load each .yml, stack u and v → final tensor (2*N, H, W)
        std::vector<torch::Tensor> channels;
        for (const auto& fp : selected) {
            cv::FileStorage fs_in(fp.string(), cv::FileStorage::READ);
            cv::Mat u_mat, v_mat;
            fs_in["u"] >> u_mat;
            fs_in["v"] >> v_mat;
            fs_in.release();

            // Convert cv::Mat (CV_32F) → torch Tensor (H, W)
            auto uT = torch::from_blob(u_mat.data, {IMG_H, IMG_W},
                                       torch::kFloat32).clone();
            auto vT = torch::from_blob(v_mat.data, {IMG_H, IMG_W},
                                       torch::kFloat32).clone();

            // Normalize to [-1, 1] (flow was clipped to [-20, 20])
            channels.push_back(uT / 20.0f);
            channels.push_back(vT / 20.0f);
        }

        // Stack: (2*N, H, W)
        torch::Tensor data = torch::stack(channels, 0);

        return {data, torch::tensor(s.label, torch::kLong)};
    }

    torch::optional<size_t> size() const override {
        return samples_.size();
    }

private:
    std::string              flowRoot_;
    int                      numFrames_;
    std::vector<Sample>      samples_;
    std::map<std::string, int> classToIdx_;
};
