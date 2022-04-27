#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_image.h>

#include "HostFilter.h"
#include "DeviceFilter.cuh"

using namespace std;

void CompareImages(
	const unsigned char* imageA,
	unsigned char* imageB,
	int size,
	int* difference
)
{
	int y = 0;
	*difference = 0;
	for (int i = 0; i < size; i++)
	{
		if (imageA[i] != imageB[i])
		{
			imageB[i] = 255;
			if (y < i / 3 / 3840)
			{
				y = i / 3 / 3840;
			}
			cout <<"("<<i % 3840<<", " << i / 3 / 3840 << ")  ";
			
			(*difference)++;
		}
	}

	cout << endl;
	cout << "Last y : " << y << endl;

	
	if ((*difference) == 0)
	{
		cout << "Images are equal." << endl;;
	}
	else
	{
		cout << "Images are different." << endl;
	}
}

void StartHostRGB(
	RGB* inputDataRgb,
	RGB* hostResult,
	size_t w,
	size_t h
)
{
	auto start_cpu = chrono::steady_clock::now();
	HostRgbFilter(inputDataRgb, hostResult, w, h);
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	cout << "Host time (RGB): " << chrono::duration<double, milli>(cpu_time).count() << endl << endl;
}

void StartDeviceRGB(
	RGB* inputDataRgb,
	RGB* deviceResult,
	size_t w,
	size_t h,
	const size_t widthInBytes
)
{
	size_t input_pitch = 0;
	//RGB* zoomed_input = ExtendBorders(inputDataRgb, w, h);
	unsigned char* pitched_input_data = nullptr;
	cudaMallocPitch((void**)(&pitched_input_data), &input_pitch, widthInBytes, h);
	cudaMemcpy2D(
		pitched_input_data,
		input_pitch,
		(unsigned char**)(inputDataRgb),
		widthInBytes,
		widthInBytes,
		h,
		cudaMemcpyHostToDevice
	);

	size_t output_pitch = 0;
	unsigned char* pitched_output_data = nullptr;
	cudaMallocPitch((void**)(&pitched_output_data), &output_pitch, widthInBytes, h);

	float time = 0;
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	int grid_x = (input_pitch + THREADS_X - 1) / THREADS_X;
	int grid_y = (h + THREADS_Y - 1) / THREADS_Y;

	dim3 dimGrid(grid_x, grid_y, 1);
	dim3 dimBlock(THREADS_X, THREADS_Y, 1);

	cout << "Grid size: " << dimGrid.x << " x " << dimGrid.y << endl;
	cout << "Block size: " << dimBlock.x << " x " << dimBlock.y << endl << endl;

	cudaEventRecord(startEvent, 0);
	DeviceRgbFilter << <dimGrid, dimBlock >> > (
		pitched_input_data, pitched_output_data,
		widthInBytes, h,
		input_pitch, output_pitch
		);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&time, startEvent, stopEvent);

	cout << "Device time (RGB): " << time << endl << endl;

	cudaMemcpy2D(
		(unsigned char*)(deviceResult),
		widthInBytes,
		pitched_output_data,
		output_pitch,
		widthInBytes,
		h,
		cudaMemcpyDeviceToHost
	);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cudaFree(pitched_input_data);
	cudaFree(pitched_output_data);
}

int main()
{
	size_t w = 0;
	size_t h = 0;
	int channels = 0;
	string fileName = "images/amogus";
	int difference = 0;

	cout << "-----------------RGB----------------" << endl;
	RGB* inputDataRgb = nullptr;
	__loadPPM(
		(fileName + ".ppm").c_str(),
		(unsigned char**)(&inputDataRgb),
		(unsigned int*)(&w),
		(unsigned int*)(&h),
		(unsigned int*)(&channels)
	);

	const size_t size = w * h;

	const size_t widthInBytes = w * sizeof(RGB);
	RGB* hostResultRGB = new RGB[size];
	RGB* deviceResultRGB = new RGB[size];

	cout << fileName << ".ppm" << " (" << w << " x " << h << ") Channels: " << channels << endl << endl;

	StartHostRGB(
		inputDataRgb,
		hostResultRGB,
		w,
		h
	);

	StartDeviceRGB(
		inputDataRgb,
		deviceResultRGB,
		w,
		h,
		widthInBytes
	);

	CompareImages((unsigned char*)(hostResultRGB), (unsigned char*)(deviceResultRGB), size, &difference);
	cout << "Difference between images: " << difference << endl;

	__savePPM((fileName + "_HostRGB.ppm").c_str(), (unsigned char*)(hostResultRGB), w, h, channels);
	__savePPM((fileName + "_DeviceRGB.ppm").c_str(), (unsigned char*)(deviceResultRGB), w, h, channels);

	delete[] inputDataRgb;
	delete[] hostResultRGB;
	delete[] deviceResultRGB;
}