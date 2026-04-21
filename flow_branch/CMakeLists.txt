cmake_minimum_required(VERSION 3.18)
project(FlowBranch)

set(CMAKE_CXX_STANDARD 17)

# Find LibTorch — set TORCH_DIR to your libtorch installation
# e.g.  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# ── extract_flow ─────────────────────────────────────────────────────────────
add_executable(extract_flow extract_flow.cpp)
target_link_libraries(extract_flow ${OpenCV_LIBS})

# ── train_eval ───────────────────────────────────────────────────────────────
add_executable(train_eval train_eval.cpp)
target_link_libraries(train_eval
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_include_directories(train_eval PRIVATE ${OpenCV_INCLUDE_DIRS})
set_property(TARGET train_eval PROPERTY CXX_STANDARD 17)
