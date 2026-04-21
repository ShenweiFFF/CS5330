/*
 * extract_flow.cpp
 * ----------------
 * Extracts dense optical flow (Farneback) from UCF-101 video clips
 * and saves u/v flow maps as .yml files for training.
 *
 * Build:
 *   g++ extract_flow.cpp -o extract_flow \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -std=c++17 -O2
 *
 * Expected UCF-101 directory structure:
 *   ucf101/
 *       ApplyEyeMakeup/
 *           v_ApplyEyeMakeup_g01_c01.avi
 *           ...
 *
 * Output structure:
 *   ucf101_flow/
 *       ApplyEyeMakeup/
 *           v_ApplyEyeMakeup_g01_c01/
 *               flow_000.yml   // contains "u" and "v" Mat (CV_32F)
 *               flow_001.yml
 *               ...
 */

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Config ───────────────────────────────────────────────────────────────────
const std::string UCF_ROOT  = "ucf101";       // path to raw UCF-101 videos (set to your data/UCF-101 directory)
const std::string FLOW_ROOT = "ucf101_flow";  // where to save flow maps (inside flow_branch/)
const int NUM_FRAMES        = 16;             // flow frames to extract per clip
const cv::Size RESIZE       = {224, 224};     // frame size before flow
const float CLIP_VAL        = 20.0f;          // clip flow to [-20, 20]
// ─────────────────────────────────────────────────────────────────────────────


/*
 * extractFlowFromVideo
 * --------------------
 * Opens a video and extracts NUM_FRAMES optical flow maps evenly spaced
 * across the clip. Each flow map is saved as a .yml file with two matrices:
 *   "u" — horizontal flow (CV_32F, H x W)
 *   "v" — vertical   flow (CV_32F, H x W)
 *
 * Returns true on success, false if the video cannot be opened.
 */
bool extractFlowFromVideo(const fs::path& videoPath, const fs::path& outDir,
                          int numFrames, cv::Size resize, float clipVal)
{
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened()) {
        std::cerr << "  [WARN] Cannot open: " << videoPath << "\n";
        return false;
    }

    // Collect all grayscale frames
    std::vector<cv::Mat> frames;
    cv::Mat frame, gray, resized;

    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, resized, resize);
        frames.push_back(resized.clone());
    }
    cap.release();

    if (static_cast<int>(frames.size()) < 2) {
        std::cerr << "  [WARN] Too few frames: " << videoPath << "\n";
        return false;
    }

    // Sample numFrames evenly spaced frame-pair indices
    int total = static_cast<int>(frames.size()) - 1;  // max index for prev frame
    for (int k = 0; k < numFrames; ++k) {
        int idx = static_cast<int>(std::round(k * total / (numFrames - 1.0)));
        idx = std::min(idx, total - 1);

        cv::Mat& prev = frames[idx];
        cv::Mat& curr = frames[idx + 1];

        // Farneback optical flow — result is (H, W, 2) stored as CV_32FC2
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            prev, curr, flow,
            0.5,   // pyr_scale
            3,     // levels
            15,    // winsize
            3,     // iterations
            5,     // poly_n
            1.2,   // poly_sigma
            0      // flags
        );

        // Split into u and v channels
        std::vector<cv::Mat> channels(2);
        cv::split(flow, channels);   // channels[0] = u, channels[1] = v

        // Clip to [-clipVal, clipVal] (symmetric). The old double-threshold
        // TRUNC sequence incorrectly saturated most pixels to +clipVal.
        channels[0] = cv::min(cv::max(channels[0], -clipVal), clipVal);
        channels[1] = cv::min(cv::max(channels[1], -clipVal), clipVal);

        // Save as .yml
        char filename[32];
        std::snprintf(filename, sizeof(filename), "flow_%03d.yml", k);
        fs::path savePath = outDir / filename;

        cv::FileStorage fs_out(savePath.string(), cv::FileStorage::WRITE);
        fs_out << "u" << channels[0];
        fs_out << "v" << channels[1];
        fs_out.release();
    }

    return true;
}


int main()
{
    fs::create_directories(FLOW_ROOT);

    // Collect all .avi files recursively
    std::vector<fs::path> videoPaths;
    for (const auto& entry : fs::recursive_directory_iterator(UCF_ROOT)) {
        if (entry.path().extension() == ".avi") {
            videoPaths.push_back(entry.path());
        }
    }

    std::cout << "Found " << videoPaths.size() << " videos in " << UCF_ROOT << "\n";

    int processed = 0, skipped = 0, failed = 0;

    for (const auto& videoPath : videoPaths) {
        // Mirror input directory structure in output
        // e.g. ucf101/ApplyEyeMakeup/v_xxx.avi
        //   -> ucf101_flow/ApplyEyeMakeup/v_xxx/
        fs::path rel    = fs::relative(videoPath, UCF_ROOT);
        fs::path outDir = fs::path(FLOW_ROOT) / rel.parent_path() / rel.stem();
        fs::create_directories(outDir);

        // Skip if already processed
        int existingCount = 0;
        for (const auto& f : fs::directory_iterator(outDir)) {
            if (f.path().extension() == ".yml") ++existingCount;
        }
        if (existingCount == NUM_FRAMES) {
            ++skipped;
            continue;
        }

        bool ok = extractFlowFromVideo(videoPath, outDir, NUM_FRAMES, RESIZE, CLIP_VAL);
        ok ? ++processed : ++failed;

        if ((processed + skipped + failed) % 100 == 0) {
            std::cout << "  Progress: " << (processed + skipped + failed)
                      << " / " << videoPaths.size() << "\r" << std::flush;
        }
    }

    std::cout << "\nDone."
              << "  Processed: " << processed
              << "  Skipped: "   << skipped
              << "  Failed: "    << failed << "\n";
    return 0;
}
