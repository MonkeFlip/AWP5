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
	*difference = 0;
	int counter = 0;
	for (int i = 0; i < size; i++)
	{
		if (imageA[i] != imageB[i])
		{
			//cout << i << ": " << imageA[i] << " " << imageB[i] << endl;
			//cout << " != ";
			//return;
			imageB[i] = 255;
			if (counter++ < 10)
			{
				cout << i << " ";
			}
			(*difference)++;
		}
	}

	cout << endl;
	if ((*difference) == 0)
	{
		cout << "Images are equal." << endl;;
	}
	else
	{
		cout << "Images are different." << endl;
	}
}

void StartHostGray(
	unsigned char* inputDataRgb,
	unsigned char* hostResult,
	size_t w,
	size_t h
)
{
	auto start_cpu = chrono::steady_clock::now();
	HostGrayDenoising(inputDataRgb, hostResult, w, h);
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	cout << "Host time (Gray): " << chrono::duration<double, milli>(cpu_time).count() << endl << endl;
}

void StartDeviceGray(
	unsigned char* inputDataGray,
	unsigned char* deviceResultGray,
	size_t w,
	size_t h,
	const size_t zoomedHeight,
	const size_t zoomedWidth
)
{
	size_t input_pitch = 0;
	unsigned char* zoomed_input = ExtendBorders(inputDataGray, w, h);
	unsigned char* pitched_input_data = nullptr;
	cudaMallocPitch((void**)(&pitched_input_data), &input_pitch, zoomedWidth, zoomedHeight);
	cudaMemcpy2D(
		pitched_input_data,
		input_pitch,
		zoomed_input,
		zoomedWidth,
		zoomedWidth,
		zoomedHeight,
		cudaMemcpyHostToDevice
	);

	size_t output_pitch = 0;
	unsigned char* pitched_output_data = nullptr;
	cudaMallocPitch((void**)(&pitched_output_data), &output_pitch, w, h + 1);

	float time = 0;
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	int grid_x = (input_pitch + THREADS_X - 1) / THREADS_X;
	int grid_y = (h + THREADS_Y - 1) / THREADS_Y + 1;
	dim3 dimGrid(grid_x, grid_y, 1);
	dim3 dimBlock(THREADS_X, THREADS_Y, 1);

	cout << "Grid size: " << dimGrid.x << " x " << dimGrid.y << endl;
	cout << "Block size: " << dimBlock.x << " x " << dimBlock.y << endl << endl;

	cudaEventRecord(startEvent, 0);
	DeviceGrayFilter << <dimGrid, dimBlock >> > (
		pitched_input_data, pitched_output_data,
		w, h,
		zoomedWidth, zoomedHeight,
		input_pitch, output_pitch
		);

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&time, startEvent, stopEvent);

	cout << "Device time (Gray): " << time << endl << endl;

	cudaMemcpy2D(
		deviceResultGray,
		w,
		pitched_output_data,
		output_pitch,
		w,
		h,
		cudaMemcpyDeviceToHost
	);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cudaFree(pitched_input_data);
	cudaFree(pitched_output_data);
}

void StartHostRGB(
	RGB* inputDataRgb,
	RGB* hostResult,
	size_t w,
	size_t h
)
{
	auto start_cpu = chrono::steady_clock::now();
	HostRgbDenoising(inputDataRgb, hostResult, w, h);
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	cout << "Host time (RGB): " << chrono::duration<double, milli>(cpu_time).count() << endl << endl;
}

void StartDeviceRGB(
	RGB* inputDataRgb,
	RGB* deviceResult,
	size_t w,
	size_t h,
	const size_t zoomedWidthInBytes,
	const size_t zoomedHeight,
	const size_t widthInBytes
)
{
	size_t input_pitch = 0;
	RGB* zoomed_input = ExtendBorders(inputDataRgb, w, h);
	unsigned char* pitched_input_data = nullptr;
	cudaMallocPitch((void**)(&pitched_input_data), &input_pitch, zoomedWidthInBytes, zoomedHeight);
	cudaMemcpy2D(
		pitched_input_data,
		input_pitch,
		(unsigned char**)(zoomed_input),
		zoomedWidthInBytes,
		zoomedWidthInBytes,
		zoomedHeight,
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
	int grid_y = (zoomedHeight + THREADS_Y - 1) / THREADS_Y;

	dim3 dimGrid(grid_x, grid_y, 1);
	dim3 dimBlock(THREADS_X, THREADS_Y, 1);

	cout << "Grid size: " << dimGrid.x << " x " << dimGrid.y << endl;
	cout << "Block size: " << dimBlock.x << " x " << dimBlock.y << endl << endl;

	cudaEventRecord(startEvent, 0);
	DeviceRgbFilter << <dimGrid, dimBlock >> > (
		pitched_input_data, pitched_output_data,
		widthInBytes, h,
		zoomedWidthInBytes, zoomedHeight,
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
	unsigned char* inputDataGray = nullptr;
	__loadPPM(
		(fileName + ".pgm").c_str(),
		&inputDataGray,
		(unsigned int*)(&w),
		(unsigned int*)(&h),
		(unsigned int*)(&channels)
	);

	cout << "----------------Gray----------------" << endl;
	cout << fileName << ".pgm" << " (" << w << " x " << h << ") Channels: " << channels << endl << endl;

	const size_t zoomedWidth = w + 2;
	const size_t zoomedHeight = h + 2;
	const size_t size = w * h;

	unsigned char* hostResultGray = new unsigned char[size];
	unsigned char* deviceResultGray = new unsigned char[size];

	StartHostGray(
		inputDataGray,
		hostResultGray,
		w,
		h
	);

	StartDeviceGray(
		inputDataGray,
		deviceResultGray,
		w,
		h,
		zoomedHeight,
		zoomedWidth
	);

	int difference = 0;
	CompareImages(hostResultGray, deviceResultGray, size, &difference);
	cout << "Difference between images: " << difference << endl;

	__savePPM((fileName + "_HostGray.pgm").c_str(), hostResultGray, w, h, channels);
	__savePPM((fileName + "_DeviceGray.pgm").c_str(), deviceResultGray, w, h, channels);



	/*cout << "-----------------RGB----------------" << endl;
	RGB* inputDataRgb = nullptr;
	__loadPPM(
		(fileName + ".ppm").c_str(),
		(unsigned char**)(&inputDataRgb),
		(unsigned int*)(&w),
		(unsigned int*)(&h),
		(unsigned int*)(&channels)
	);

	const size_t widthInBytes = w * sizeof(RGB);
	const size_t zoomedWidthInBytes = zoomedWidth * sizeof(RGB);
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
		zoomedWidthInBytes,
		zoomedHeight,
		widthInBytes
	);

	CompareImages((unsigned char*)(hostResultRGB), (unsigned char*)(deviceResultRGB), size, &difference);
	cout << "Difference between images: " << difference << endl;

	__savePPM((fileName + "_HostRGB.ppm").c_str(), (unsigned char*)(hostResultRGB), w, h, channels);
	__savePPM((fileName + "_DeviceRGB.ppm").c_str(), (unsigned char*)(deviceResultRGB), w, h, channels);*/

	delete[] inputDataGray;
	delete[] hostResultGray;
	delete[] deviceResultGray;

	/*delete[] inputDataRgb;
	delete[] hostResultRGB;
	delete[] deviceResultRGB;*/
}