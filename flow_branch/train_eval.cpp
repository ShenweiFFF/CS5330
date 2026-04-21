/*
 * train_eval.cpp
 * --------------
 * Trains and evaluates the optical flow ResNet-18 branch on UCF-101.
 *
 * Build:
 *   g++ train_eval.cpp -o train_eval \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -I/path/to/libtorch/include \
 *       -L/path/to/libtorch/lib -ltorch -lc10 \
 *       -std=c++17 -O2
 *
 * Usage:
 *   ./train_eval --split 1              # train fold 1
 *   ./train_eval --split 1 --eval       # eval only (loads checkpoint)
 *   ./train_eval --ablation             # ablation across all 3 folds
 */

#include "flow_model.hpp"

#include <torch/torch.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ── Config ───────────────────────────────────────────────────────────────────
const std::string FLOW_ROOT   = "ucf101_flow";
// Shared split files live one level up at ../splits/ (project root)
const std::string SPLITS_DIR  = "../splits";
const int   BATCH_SIZE        = 16;
const int   EPOCHS            = 30;
const float LR                = 1e-3f;
const float WEIGHT_DECAY      = 1e-4f;
const int   NUM_WORKERS       = 4;
// ─────────────────────────────────────────────────────────────────────────────


/*
 * trainOneEpoch
 * -------------
 * Runs one training epoch. Returns average cross-entropy loss.
 */
float trainOneEpoch(FlowResNet18& model,
                    torch::data::StatelessDataLoader<
                        torch::data::datasets::MapDataset<
                            UCF101FlowDataset,
                            torch::data::transforms::Stack<>>,
                        torch::data::samplers::RandomSampler>& loader,
                    torch::optim::Adam& optimizer,
                    torch::Device device)
{
    model->train();
    float totalLoss = 0.0f;
    int   totalSamples = 0;

    for (auto& batch : loader) {
        auto data   = batch.data.to(device);
        auto labels = batch.target.to(device);

        optimizer.zero_grad();
        auto logits = model->forward(data);
        auto loss   = torch::nn::functional::cross_entropy(logits, labels);
        loss.backward();
        optimizer.step();

        totalLoss    += loss.item<float>() * data.size(0);
        totalSamples += data.size(0);
    }

    return totalLoss / totalSamples;
}


/*
 * evaluate
 * --------
 * Computes Top-1 and Top-5 accuracy on a DataLoader.
 * Returns {top1_pct, top5_pct}.
 */
std::pair<float, float> evaluate(FlowResNet18& model,
                                  torch::data::StatelessDataLoader<
                                      torch::data::datasets::MapDataset<
                                          UCF101FlowDataset,
                                          torch::data::transforms::Stack<>>,
                                      torch::data::samplers::SequentialSampler>& loader,
                                  torch::Device device)
{
    model->eval();
    torch::NoGradGuard no_grad;

    int top1Correct = 0, top5Correct = 0, total = 0;

    for (auto& batch : loader) {
        auto data   = batch.data.to(device);
        auto labels = batch.target.to(device);

        auto logits = model->forward(data);   // (B, num_classes)

        // Top-1
        auto pred1 = logits.argmax(1);
        top1Correct += pred1.eq(labels).sum().item<int>();

        // Top-5
        auto top5   = std::get<1>(logits.topk(5, 1));   // (B, 5)
        int  B      = data.size(0);
        for (int i = 0; i < B; ++i) {
            int gt = labels[i].item<int>();
            for (int k = 0; k < 5; ++k) {
                if (top5[i][k].item<int>() == gt) {
                    ++top5Correct;
                    break;
                }
            }
        }

        total += B;
    }

    float top1 = 100.0f * top1Correct / total;
    float top5 = 100.0f * top5Correct / total;
    return {top1, top5};
}


/*
 * runAblation
 * -----------
 * Loads saved checkpoints for all three folds and prints a results table.
 * Train all three folds first.
 */
void runAblation(torch::Device device) {
    std::cout << "\n" << std::string(55, '=') << "\n"
              << "Ablation: Optical Flow Branch Only (3-fold CV)\n"
              << std::string(55, '=') << "\n"
              << std::left  << std::setw(8)  << "Split"
              << std::right << std::setw(12) << "Top-1 (%)"
              << std::right << std::setw(12) << "Top-5 (%)" << "\n"
              << std::string(55, '-') << "\n";

    std::vector<float> top1Scores, top5Scores;

    for (int split : {1, 2, 3}) {
        std::string ckpt = "flow_branch_split" + std::to_string(split) + ".pt";
        if (!fs::exists(ckpt)) {
            std::cout << "  Split " << split << ": checkpoint not found, skipping.\n";
            continue;
        }

        // Build test loader
        auto testDs = UCF101FlowDataset(
            FLOW_ROOT,
            SPLITS_DIR + "/testlist0" + std::to_string(split) + ".txt",
            SPLITS_DIR + "/classInd.txt"
        );
        auto testLoader = torch::data::make_data_loader<
            torch::data::samplers::SequentialSampler>(
            std::move(testDs).map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(NUM_WORKERS));

        FlowResNet18 model(NUM_FRAMES, NUM_CLASSES);
        torch::load(model, ckpt);
        model->to(device);

        auto [top1, top5] = evaluate(model, *testLoader, device);
        top1Scores.push_back(top1);
        top5Scores.push_back(top5);

        std::cout << "  " << std::left  << std::setw(6)  << split
                          << std::right << std::setw(12) << std::fixed << std::setprecision(2) << top1
                          << std::right << std::setw(12) << top5 << "\n";
    }

    if (!top1Scores.empty()) {
        float avg1 = 0, avg5 = 0;
        for (auto v : top1Scores) avg1 += v;
        for (auto v : top5Scores) avg5 += v;
        avg1 /= top1Scores.size();
        avg5 /= top5Scores.size();

        std::cout << std::string(55, '-') << "\n"
                  << "  " << std::left  << std::setw(6)  << "Avg"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << avg1
                  << std::right << std::setw(12) << avg5 << "\n";
    }
    std::cout << std::string(55, '=') << "\n";
}


/*
 * train
 * -----
 * Full training loop for one fold.
 */
void train(int split, torch::Device device) {
    std::cout << "Device: " << device << "\n";
    std::cout << "Running fold " << split << "\n";

    std::string trainFile = SPLITS_DIR + "/trainlist0" + std::to_string(split) + ".txt";
    std::string testFile  = SPLITS_DIR + "/testlist0"  + std::to_string(split) + ".txt";
    std::string classFile = SPLITS_DIR + "/classInd.txt";
    std::string ckptPath  = "flow_branch_split" + std::to_string(split) + ".pt";

    // Build datasets and loaders
    auto trainDs = UCF101FlowDataset(FLOW_ROOT, trainFile, classFile);
    auto testDs  = UCF101FlowDataset(FLOW_ROOT, testFile,  classFile);

    auto trainLoader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
        std::move(trainDs).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(NUM_WORKERS));

    auto testLoader = torch::data::make_data_loader<
        torch::data::samplers::SequentialSampler>(
        std::move(testDs).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(NUM_WORKERS));

    FlowResNet18 model(NUM_FRAMES, NUM_CLASSES);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(LR).weight_decay(WEIGHT_DECAY));

    // Step LR: decay by 0.1 every 10 epochs
    auto scheduler = [&](int epoch) {
        if (epoch % 10 == 0 && epoch > 0) {
            for (auto& pg : optimizer.param_groups()) {
                static_cast<torch::optim::AdamOptions&>(pg.options()).lr(
                    static_cast<torch::optim::AdamOptions&>(pg.options()).lr() * 0.1);
            }
        }
    };

    float bestTop1 = 0.0f;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float loss = trainOneEpoch(model, *trainLoader, optimizer, device);
        auto [top1, top5] = evaluate(model, *testLoader, device);
        scheduler(epoch);

        std::cout << "Epoch [" << epoch << "/" << EPOCHS << "]  "
                  << "Loss: "  << std::fixed << std::setprecision(4) << loss  << "  "
                  << "Top-1: " << std::fixed << std::setprecision(2) << top1  << "%  "
                  << "Top-5: " << top5 << "%\n";

        if (top1 > bestTop1) {
            bestTop1 = top1;
            torch::save(model, ckptPath);
            std::cout << "  → Saved best model (Top-1: " << bestTop1 << "%)\n";
        }
    }

    std::cout << "\nTraining complete. Best Top-1: " << bestTop1 << "%\n";
}


// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Parse arguments
    int  split    = 1;
    bool evalOnly = false;
    bool ablation = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--split" && i + 1 < argc)
            split = std::stoi(argv[++i]);
        else if (arg == "--eval")
            evalOnly = true;
        else if (arg == "--ablation")
            ablation = true;
    }

    torch::Device device = torch::cuda::is_available()
                           ? torch::Device(torch::kCUDA)
                           : torch::Device(torch::kCPU);

    if (ablation) {
        runAblation(device);
        return 0;
    }

    if (evalOnly) {
        std::string ckpt = "flow_branch_split" + std::to_string(split) + ".pt";
        auto testDs = UCF101FlowDataset(
            FLOW_ROOT,
            SPLITS_DIR + "/testlist0" + std::to_string(split) + ".txt",
            SPLITS_DIR + "/classInd.txt"
        );
        auto testLoader = torch::data::make_data_loader<
            torch::data::samplers::SequentialSampler>(
            std::move(testDs).map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(NUM_WORKERS));

        FlowResNet18 model(NUM_FRAMES, NUM_CLASSES);
        torch::load(model, ckpt);
        model->to(device);

        auto [top1, top5] = evaluate(model, *testLoader, device);
        std::cout << "Top-1: " << top1 << "%   Top-5: " << top5 << "%\n";
        return 0;
    }

    train(split, device);
    return 0;
}
