
#include "cudaLib.cuh"
#include "cpuLib.h"
#include <cmath>
#include <curand_kernel.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	// Calculate the array index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// if the array index is out of bounds of array size, do nothing.
	// Otherwise, do saxpy!
	if (i < size)
		y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {
	std::cout << "Hello GPU Saxpy!\n";
	float *host_x, *host_y, *device_x, *device_y, *result_y;

	// allocate a separate result_y array to store the results.
	// So that the verifyVector function can verify the GPU results.
	host_x = new float[vectorSize * sizeof(float)];
	host_y = new float[vectorSize * sizeof(float)];
	result_y = new float[vectorSize * sizeof(float)];

	if (!host_x || !host_y || !result_y) {
		std::cout << "Unable to allocate memory ... Exiting!\n";
		return -1;
	}

	// fill the array host_x and host_y with some random data
	for (int i = 0; i < vectorSize; i++) {
		host_x[i] = (float)(rand() % 100);
	}
	for (int i = 0; i < vectorSize; i++) {
		host_y[i] = (float)(rand() % 100);
	}
	float scale = 2.0;

	#ifndef DEBUG_PRINT_DISABLE
		std::cout.setf(std::ios::fixed);
		std::cout.precision(4);
		std::cout << "\n Adding vectors : \n";
		std::cout << " scale = " << scale << "\n";
		std::cout << " host_x = { ";
		for (int i = 0; i < 5; ++i) {
			std::cout << host_x[i] << ", ";
		}
		std::cout << " ... }\n";
		std::cout << " host_y = { ";
		for (int i = 0; i < 5; ++i) {
			std::cout << host_y[i] << ", ";
		}
		std::cout << " ... }\n";
	#endif

	gpuErrchk(cudaMalloc((void **)&device_x, vectorSize * sizeof(float)));
	gpuErrchk(cudaMalloc((void **)&device_y, vectorSize * sizeof(float)));

	gpuErrchk(cudaMemcpy(device_x, host_x, vectorSize * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_y, host_y, vectorSize * sizeof(float), cudaMemcpyHostToDevice));

	// Call the saxpy kernel
	saxpy_gpu<<<std::ceil(float(vectorSize) / 256.0), 256>>>(device_x, device_y, scale, vectorSize);
	// Copy the results from device to the host for verification
	gpuErrchk(cudaMemcpy(result_y, device_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

	#ifndef DEBUG_PRINT_DISABLE
		std::cout.setf(std::ios::fixed);
		std::cout.precision(4);
		std::cout << " result_y = { ";
		for (int i = 0; i < 5; ++i) {
			std::cout << result_y[i] << ", ";
		}
		std::cout << " ... }\n";
	#endif

	// verify the vector
	int errorCount = verifyVector(host_x, host_y, result_y, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// free all the allocated memory
	delete host_x;
	delete host_y;
	delete result_y;
	gpuErrchk(cudaFree(device_x));
	gpuErrchk(cudaFree(device_y));

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// Calculate the array index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float device_x, device_y;
	curandState_t device_rnd_states;
	// Init the random number generator with current clock tick
	// as the seed.
	curand_init(clock64(), tid, 0, &device_rnd_states);

	if (tid < pSumSize) {
		for (int i = 0; i < sampleSize; i++) {
			// get a random float between 0 - 1 for x and y
			device_x = curand_uniform(&device_rnd_states);
			device_y = curand_uniform(&device_rnd_states);

			// calculate the distance of this (x, y) from
			// origin (0, 0). If it is less than 1, it means it
			// lies inside the circle, increase the pSums array by 1.
			if (int(device_x * device_x + device_y * device_y) == 0)
				pSums[tid] = pSums[tid] + 1;
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Since we want to add reduceSize number of elements at the
	// Same time in this thread. We set the start and end as follows.
	int indexStart = tid * reduceSize;
	int indexEnd = (tid + 1) * reduceSize;

	if (indexEnd <= pSumSize && tid < (pSumSize / reduceSize)) {
		// Sum all the pSums into totals[tid]
		for (int i = indexStart; i < indexEnd; i++) {
			totals[tid] += pSums[i];
		}
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
	uint64_t totalSum = 0;
	uint64_t *pSums, *totals, *totals_Host;
	uint64_t pSumSize = generateThreadCount;

	gpuErrchk(cudaMalloc((uint64_t**)&pSums, pSumSize * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc((uint64_t**)&totals, reduceThreadCount * sizeof(uint64_t)));
	totals_Host = new uint64_t[reduceThreadCount];

	// Call the kernel which computes number of random points
	// inside a unit circle.
	generatePoints<<<std::ceil((float)generateThreadCount/256.0), 256>>>(pSums, pSumSize, sampleSize);

	// Perform a check on the reduceThreadCount to make sure
	// it is bounded. Update the number of blocks accordingly.
	int updatedReduceThreadCount = reduceThreadCount;
	int numReduceBlock = 1;
	if (reduceThreadCount > 256) {
		updatedReduceThreadCount = 256;
		numReduceBlock = std::ceil((float)reduceThreadCount / (float)updatedReduceThreadCount);
	}
	// Call the kernel to reduce the partial sums to a lower number.
	reduceCounts<<<numReduceBlock, updatedReduceThreadCount>>>(pSums, totals, generateThreadCount, reduceSize);
	// Copy the reduced partial sums to host.
	gpuErrchk(cudaMemcpy(totals_Host, totals, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	for (int i = 0; i < reduceThreadCount; i++)
		totalSum += totals_Host[i];

	approxPi = double(totalSum) / double(generateThreadCount * sampleSize);
	approxPi = approxPi * 4.0f;

	return approxPi;
}
