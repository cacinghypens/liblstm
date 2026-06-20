cmake_minimum_required(VERSION 3.18)
project(lstm_vae VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find TorchScript (LibTorch)
find_package(Torch REQUIRED)

# Source files
set(LSTM_VAE_SOURCES
    LSTMEncoder.cpp
    LSTMDecoder.cpp
    LSTMVAE.cpp
)

# Header files
set(LSTM_VAE_HEADERS
    LSTMEncoder.h
    LSTMDecoder.h
    LSTMVAE.h
)

# Create shared library
add_library(lstm_vae SHARED ${LSTM_VAE_SOURCES} ${LSTM_VAE_HEADERS})

target_include_directories(lstm_vae PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link against LibTorch
target_link_libraries(lstm_vae "${TORCH_LIBRARIES}")

# Compiler flags
if(MSVC)
    target_compile_options(lstm_vae PRIVATE /W4 /O2)
else()
    target_compile_options(lstm_vae PRIVATE -Wall -Wextra -O2)
endif()

# Example executable
add_executable(example examples/example.cpp)
target_link_libraries(example lstm_vae "${TORCH_LIBRARIES}")

# Installation rules
install(TARGETS lstm_vae
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${LSTM_VAE_HEADERS}
    DESTINATION include/lstm_vae
)

# Print configuration message
message(STATUS "LibTorch version: ${TORCH_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
