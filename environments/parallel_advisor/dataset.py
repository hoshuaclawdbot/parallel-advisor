"""
Dataset of C++ code samples for parallelization.

Each sample includes:
- Serial C++ code
- Hardware spec (cores, threads, GPU)
- Optional ground truth (best strategy, expected speedup)
- Test inputs for correctness checking
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class HardwareSpec:
    """Hardware specification for parallelization decisions."""
    cores: int = 8
    threads: int = 16
    gpu: bool = False
    gpu_name: Optional[str] = None
    memory_gb: int = 16


@dataclass 
class CodeSample:
    """A single code sample for the dataset."""
    id: str
    code: str
    description: str
    hardware: HardwareSpec
    
    # For correctness testing
    test_input: Optional[str] = None
    expected_output: Optional[str] = None
    
    # Ground truth (optional, for evaluation)
    best_strategy: Optional[str] = None  # "serial", "threads", "openmp", "cuda"
    best_threads: Optional[int] = None
    expected_speedup: Optional[float] = None
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard


# Basic dataset of parallelizable functions
BASIC_DATASET = [
    CodeSample(
        id="array_sum",
        description="Sum all elements in an array",
        code='''#include <iostream>
#include <vector>
#include <chrono>

long long sum_array(const std::vector<int>& arr) {
    long long sum = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    const size_t N = 100000000;
    std::vector<int> arr(N);
    for (size_t i = 0; i < N; i++) {
        arr[i] = i % 100;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    long long result = sum_array(arr);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sum: " << result << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=8,
        expected_speedup=4.0,
        tags=["reduction", "simple", "memory-bound"],
        difficulty="easy",
    ),
    
    CodeSample(
        id="matrix_multiply",
        description="Dense matrix multiplication C = A * B",
        code='''#include <iostream>
#include <vector>
#include <chrono>

void matmul(const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            std::vector<std::vector<double>>& C,
            int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    const int N = 1024;
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> B(N, std::vector<double>(N, 2.0));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "C[0][0]: " << C[0][0] << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=8,
        expected_speedup=6.0,
        tags=["compute-bound", "nested-loops", "classic"],
        difficulty="medium",
    ),
    
    CodeSample(
        id="vector_add",
        description="Element-wise vector addition",
        code='''#include <iostream>
#include <vector>
#include <chrono>

void vector_add(const std::vector<float>& a,
                const std::vector<float>& b,
                std::vector<float>& c,
                size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 50000000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N);
    
    auto start = std::chrono::high_resolution_clock::now();
    vector_add(a, b, c, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "c[0]: " << c[0] << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=4,
        expected_speedup=2.5,
        tags=["embarrassingly-parallel", "memory-bound", "simple"],
        difficulty="easy",
    ),
    
    CodeSample(
        id="small_loop",
        description="Very small loop - parallelization overhead may hurt",
        code='''#include <iostream>
#include <vector>
#include <chrono>

int small_sum(const std::vector<int>& arr) {
    int sum = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 1000000; iter++) {
        volatile int result = small_sum(arr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="serial",
        best_threads=None,
        expected_speedup=1.0,
        tags=["too-small", "overhead-dominated"],
        difficulty="easy",
    ),
    
    CodeSample(
        id="dot_product",
        description="Dot product of two vectors",
        code='''#include <iostream>
#include <vector>
#include <chrono>

double dot_product(const std::vector<double>& a,
                   const std::vector<double>& b,
                   size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    const size_t N = 100000000;
    std::vector<double> a(N, 0.5);
    std::vector<double> b(N, 2.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    double result = dot_product(a, b, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Dot product: " << result << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=8,
        expected_speedup=4.0,
        tags=["reduction", "memory-bound"],
        difficulty="easy",
    ),
    
    CodeSample(
        id="mandelbrot",
        description="Mandelbrot set computation",
        code='''#include <iostream>
#include <vector>
#include <chrono>
#include <complex>

int mandelbrot(std::complex<double> c, int max_iter) {
    std::complex<double> z = 0;
    for (int i = 0; i < max_iter; i++) {
        if (std::abs(z) > 2.0) return i;
        z = z * z + c;
    }
    return max_iter;
}

void compute_mandelbrot(std::vector<int>& image, int width, int height, int max_iter) {
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double real = x_min + (x_max - x_min) * x / width;
            double imag = y_min + (y_max - y_min) * y / height;
            image[y * width + x] = mandelbrot(std::complex<double>(real, imag), max_iter);
        }
    }
}

int main() {
    const int WIDTH = 2048;
    const int HEIGHT = 2048;
    const int MAX_ITER = 1000;
    
    std::vector<int> image(WIDTH * HEIGHT);
    
    auto start = std::chrono::high_resolution_clock::now();
    compute_mandelbrot(image, WIDTH, HEIGHT, MAX_ITER);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=16,
        expected_speedup=10.0,
        tags=["embarrassingly-parallel", "compute-bound", "irregular-work"],
        difficulty="medium",
    ),
    
    CodeSample(
        id="histogram",
        description="Compute histogram of values",
        code='''#include <iostream>
#include <vector>
#include <chrono>
#include <random>

void compute_histogram(const std::vector<int>& data, std::vector<int>& hist, int num_bins) {
    for (size_t i = 0; i < data.size(); i++) {
        int bin = data[i] % num_bins;
        hist[bin]++;
    }
}

int main() {
    const size_t N = 100000000;
    const int NUM_BINS = 256;
    
    std::vector<int> data(N);
    std::mt19937 gen(42);
    for (size_t i = 0; i < N; i++) {
        data[i] = gen() % 1000;
    }
    
    std::vector<int> hist(NUM_BINS, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    compute_histogram(data, hist, NUM_BINS);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "hist[0]: " << hist[0] << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",
        best_threads=8,
        expected_speedup=3.0,
        tags=["histogram", "atomic", "reduction-variant"],
        difficulty="medium",
    ),
    
    CodeSample(
        id="prefix_sum",
        description="Parallel prefix sum (scan)",
        code='''#include <iostream>
#include <vector>
#include <chrono>

void prefix_sum(const std::vector<int>& input, std::vector<int>& output, size_t n) {
    output[0] = input[0];
    for (size_t i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main() {
    const size_t N = 100000000;
    std::vector<int> input(N);
    std::vector<int> output(N);
    
    for (size_t i = 0; i < N; i++) {
        input[i] = i % 10;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    prefix_sum(input, output, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "output[N-1]: " << output[N-1] << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    return 0;
}''',
        hardware=HardwareSpec(cores=8, threads=16),
        best_strategy="openmp",  # With work-efficient parallel scan
        best_threads=8,
        expected_speedup=2.0,
        tags=["scan", "prefix-sum", "data-dependent"],
        difficulty="hard",
    ),
]


def load_dataset(
    name: str = "basic",
    hardware_override: Optional[HardwareSpec] = None,
) -> list[CodeSample]:
    """
    Load a dataset of code samples.
    
    Args:
        name: Dataset name ("basic", "polybench", "custom")
        hardware_override: Override hardware spec for all samples
    
    Returns:
        List of CodeSample objects
    """
    if name == "basic":
        dataset = BASIC_DATASET
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Override hardware if specified
    if hardware_override:
        for sample in dataset:
            sample.hardware = hardware_override
    
    return dataset


def save_dataset(samples: list[CodeSample], path: str):
    """Save dataset to JSON."""
    data = [
        {
            "id": s.id,
            "code": s.code,
            "description": s.description,
            "hardware": {
                "cores": s.hardware.cores,
                "threads": s.hardware.threads,
                "gpu": s.hardware.gpu,
            },
            "best_strategy": s.best_strategy,
            "best_threads": s.best_threads,
            "tags": s.tags,
            "difficulty": s.difficulty,
        }
        for s in samples
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
