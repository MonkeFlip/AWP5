#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceFilter.cuh"

#include <stdio.h>

__global__ void DeviceGrayFilter(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
)
{
	const int x = blockIdx.x * THREADS_X + threadIdx.x;
	const int y = blockIdx.y * THREADS_Y + threadIdx.y;

	const int int_widht = inPitch / sizeof(int);
	const int width_border = (w + sizeof(int) - 1) / sizeof(int);

	uchar4* thread_input = (uchar4*)(inputData);
	uchar4* thread_output = (uchar4*)(outputData);

	__shared__ uchar4 shared_memory[THREADS_Y + 2][THREADS_X + 2];
	__shared__ uchar4 left_extension[1][THREADS_Y + 2];
	__shared__ uchar4 right_extension[1][THREADS_Y + 2];
	__shared__ uchar4 top_extension[1][THREADS_X + 2];
	__shared__ uchar4 bottom_extension[1][THREADS_X + 2];
	__shared__ uchar4 corners[4];

	if (y <= h)
	{
		uchar4* thread_input = (uchar4*)(inputData);
		uchar4* thread_output = (uchar4*)(outputData);

		shared_memory[threadIdx.y][threadIdx.x] = thread_input[y * int_widht + x];
		if (threadIdx.y == 0 && (blockIdx.y > 0))
		{
			top_extension[0][threadIdx.x] = thread_input[(y - 1) * int_widht + x];
		}

		if (threadIdx.y == THREADS_Y - 1)
		{
			bottom_extension[0][threadIdx.x] = thread_input[(y + 1) * int_widht + x];
		}

		if (y == h)
		{
			bottom_extension[0][threadIdx.x] = thread_input[(y)*int_widht + x];
		}

		if (threadIdx.x == 0 && blockIdx.x > 0)
		{
			left_extension[0][threadIdx.y] = thread_input[y * int_widht + x - 1];
		}

		if (threadIdx.x == THREADS_X - 1)
		{
			right_extension[0][threadIdx.y] = thread_input[y * int_widht + x + 1];
		}

		if (x != 0 && y != 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			corners[0] = thread_input[(y - 1) * int_widht + x - 1];
		}

		if (y != 0 && threadIdx.y == 0 && threadIdx.x == THREADS_X - 1)
		{
			corners[1] = thread_input[(y - 1) * int_widht + x + 1];
		}

		if (x != 0 && threadIdx.y == THREADS_Y - 1 && threadIdx.x == 0)
		{
			corners[2] = thread_input[(y + 1) * int_widht + x - 1];
		}


		if (threadIdx.y == THREADS_Y - 1 && threadIdx.x == THREADS_X - 1)
		{
			corners[3] = thread_input[(y + 1) * int_widht + x + 1];
		}

		if (y == h && threadIdx.x == THREADS_X - 1)
		{
			corners[3] = thread_input[(y)*int_widht + x + 1];
		}


		if (threadIdx.y < 2)
		{
			if (y + THREADS_Y <= h)
			{
				shared_memory[THREADS_Y + threadIdx.y][threadIdx.x] = thread_input[(THREADS_Y + y) * int_widht + x];
			}
			if (threadIdx.x < THREADS_Y + 2)
			{
				int temp_x = blockIdx.x * THREADS_X + threadIdx.y;
				int temp_y = blockIdx.y * THREADS_Y + threadIdx.x;

				if (temp_x < int_widht && temp_y < zoomedHeight)
				{
					shared_memory[threadIdx.x][THREADS_X + threadIdx.y] = thread_input[temp_y * int_widht + THREADS_X + temp_x];
				}
			}
		}
	}

	__syncthreads();

	if (x <= int_widht && y <= h && y >= 1)
	{
		uchar4 generated_int = { 0 };
		unsigned char result;
		unsigned char value;
		uchar4 int_1;
		uchar4 mem;
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			int_1.w = top_extension[0][0].x;
		}
		else if (threadIdx.x == 0)
		{
			int_1.w = shared_memory[threadIdx.y - 1][threadIdx.x].x;
		}
		else if (threadIdx.y == 0)
		{
			int_1.w = top_extension[0][threadIdx.x].x;
		}
		else
		{
			int_1.w = shared_memory[threadIdx.y - 1][threadIdx.x].x;
		}
		result = 0;
		value = int_1.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_1.w = result;
		uchar4 int_2;
		if (threadIdx.y == 0)
		{
			mem = top_extension[0][threadIdx.x];
			int_2.x = mem.y;
			int_2.y = mem.z;
			int_2.z = mem.w;
			int_2.w = threadIdx.x == THREADS_X - 1 ? corners[1].x : top_extension[0][threadIdx.x + 1].x;
		}
		else
		{
			mem = shared_memory[threadIdx.y - 1][threadIdx.x];
			int_2.x = mem.y;
			int_2.y = mem.z;
			int_2.z = mem.w;
			int_2.w = shared_memory[threadIdx.y - 1][threadIdx.x + 1].x;
		}
		result = 0;
		value = int_2.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.x = result;
		result = 0;
		value = int_2.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.y = result;
		result = 0;
		value = int_2.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.z = result;
		result = 0;
		value = int_2.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.w = result;
		uchar4 int_3;
		if (threadIdx.x == 0)
		{
			int_3.w = shared_memory[threadIdx.y][threadIdx.x].x;
		}
		else
		{
			int_3.w = shared_memory[threadIdx.y][threadIdx.x].x;
		}
		result = 0;
		value = int_3.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_3.w = result;
		uchar4 int_4;
		mem = shared_memory[threadIdx.y][threadIdx.x];
		int_4.x = mem.y;
		int_4.y = mem.z;
		int_4.z = mem.w;
		int_4.w = threadIdx.x == THREADS_X - 1 ? right_extension[0][threadIdx.y].x : shared_memory[threadIdx.y][threadIdx.x + 1].x;
		result = 0;
		value = int_4.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.x = result;
		result = 0;
		value = int_4.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.y = result;
		result = 0;
		value = int_4.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.z = result;
		result = 0;
		value = int_4.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.w = result;
		uchar4 int_5;
		int_5.w = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x].x : shared_memory[threadIdx.y + 1][threadIdx.x].x;

		result = 0;
		value = int_5.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_5.w = result;
		uchar4 int_6;
		mem = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x] : shared_memory[threadIdx.y + 1][threadIdx.x];
		int_6.x = mem.y;
		int_6.y = mem.z;
		int_6.z = mem.w;
		if (threadIdx.x == THREADS_X - 1)
		{
			int_6.w = threadIdx.y == THREADS_Y - 1 || y == h ? corners[3].x : shared_memory[threadIdx.y + 1][threadIdx.x + 1].x;
		}
		else
		{
			int_6.w = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x + 1].x : shared_memory[threadIdx.y + 1][threadIdx.x + 1].x;
		}

		result = 0;
		value = int_6.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.x = result;
		result = 0;
		value = int_6.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.y = result;
		result = 0;
		value = int_6.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.z = result;
		result = 0;
		value = int_6.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.w = result;
		uchar4 int_7;
		if (threadIdx.y == 0)
		{
			mem = threadIdx.x >= THREADS_X - 1 ? corners[1] : top_extension[0][threadIdx.x + 1];
			int_7.x = mem.y;
		}
		else
		{
			mem = shared_memory[threadIdx.y - 1][threadIdx.x + 1];
			int_7.x = mem.y;
		}
		result = 0;
		value = int_7.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_7.x = result;
		uchar4 int_8;
		mem = shared_memory[threadIdx.y][threadIdx.x + 1];
		int_8.x = mem.y;
		result = 0;
		value = int_8.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_8.x = result;
		uchar4 int_9;
		if (threadIdx.y == THREADS_Y - 1 || y == h)
		{
			mem = threadIdx.x == THREADS_X - 1 ? corners[3] : bottom_extension[0][threadIdx.x + 1];
		}
		else
		{
			mem = shared_memory[threadIdx.y + 1][threadIdx.x + 1];
		}

		int_9.x = mem.y;
		result = 0;
		value = int_9.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_9.x = result;

		generated_int.x = (
			int_1.w + int_2.x * (-2) + int_2.y +
			int_3.w * (-2) + int_4.x * 4 + int_4.y * (-2) +
			int_5.w + int_6.x * (-2) + int_6.y
			);

		generated_int.y = (
			int_2.x + int_2.y * (-2) + int_2.z +
			int_4.x * (-2) + int_4.y * 4 + int_4.z * (-2) +
			int_6.x + int_6.y * (-2) + int_6.z
			);

		generated_int.z = (
			int_2.y + int_2.z * (-2) + int_2.w +
			int_4.y * (-2) + int_4.z * 4 + int_4.w * (-2) +
			int_6.y + int_6.z * (-2) + int_6.w
			);

		generated_int.w = (
			int_2.z + int_2.w * (-2) + int_7.x +
			int_4.z * (-2) + int_4.w * 4 + int_8.x * (-2) +
			int_6.z + int_6.w * (-2) + int_9.x
			);

		const int output_int_width = outPitch / sizeof(int);


		result = 0;
		value = generated_int.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.x = result;
		result = 0;
		value = generated_int.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.y = result;
		result = 0;
		value = generated_int.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.z = result;
		result = 0;
		value = generated_int.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.w = result;
		thread_output[(y - 1) * output_int_width + x] = generated_int;
	}
}

__global__ void DeviceRgbFilter(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
)
{
	const int x = blockIdx.x * THREADS_X + threadIdx.x;
	const int y = blockIdx.y * THREADS_Y + threadIdx.y;

	const int int_widht = inPitch / sizeof(int);
	const int width_border = (w + sizeof(int) - 1) / sizeof(int);

	uchar4* thread_input = (uchar4*)(inputData);
	uchar4* thread_output = (uchar4*)(outputData);

	__shared__ uchar4 shared_memory[THREADS_Y + 2][THREADS_X + 2];
	__shared__ uchar4 left_extension[1][THREADS_Y + 2];
	__shared__ uchar4 right_extension[1][THREADS_Y + 2];
	__shared__ uchar4 top_extension[1][THREADS_X + 2];
	__shared__ uchar4 bottom_extension[1][THREADS_X + 2];
	__shared__ uchar4 corners[4];

	if (y <= h)
	{
		uchar4* thread_input = (uchar4*)(inputData);
		uchar4* thread_output = (uchar4*)(outputData);

		shared_memory[threadIdx.y][threadIdx.x] = thread_input[y * int_widht + x];
		if (threadIdx.y == 0 && (blockIdx.y > 0))
		{
			top_extension[0][threadIdx.x] = thread_input[(y - 1) * int_widht + x];
		}

		if (threadIdx.y == THREADS_Y - 1)
		{
			bottom_extension[0][threadIdx.x] = thread_input[(y + 1) * int_widht + x];
		}

		if (y == h)
		{
			bottom_extension[0][threadIdx.x] = thread_input[(y)*int_widht + x];
		}

		if (threadIdx.x == 0 && blockIdx.x > 0)
		{
			left_extension[0][threadIdx.y] = thread_input[y * int_widht + x - 1];
		}

		if (threadIdx.x == THREADS_X - 1)
		{
			right_extension[0][threadIdx.y] = thread_input[y * int_widht + x + 1];
		}

		if (x != 0 && y != 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			corners[0] = thread_input[(y - 1) * int_widht + x - 1];
		}

		if (y != 0 && threadIdx.y == 0 && threadIdx.x == THREADS_X - 1)
		{
			corners[1] = thread_input[(y - 1) * int_widht + x + 1];
		}

		if (x != 0 && threadIdx.y == THREADS_Y - 1 && threadIdx.x == 0)
		{
			corners[2] = thread_input[(y + 1) * int_widht + x - 1];
		}


		if (threadIdx.y == THREADS_Y - 1 && threadIdx.x == THREADS_X - 1)
		{
			corners[3] = thread_input[(y + 1) * int_widht + x + 1];
		}

		if (y == h && threadIdx.x == THREADS_X - 1)
		{
			corners[3] = thread_input[(y)*int_widht + x + 1];
		}


		if (threadIdx.y < 2)
		{
			if (y + THREADS_Y <= h)
			{
				shared_memory[THREADS_Y + threadIdx.y][threadIdx.x] = thread_input[(THREADS_Y + y) * int_widht + x];
			}
			if (threadIdx.x < THREADS_Y + 2)
			{
				int temp_x = blockIdx.x * THREADS_X + threadIdx.y;
				int temp_y = blockIdx.y * THREADS_Y + threadIdx.x;

				if (temp_x < int_widht && temp_y < zoomedHeight)
				{
					shared_memory[threadIdx.x][THREADS_X + threadIdx.y] = thread_input[temp_y * int_widht + THREADS_X + temp_x];
				}
			}
		}
	}

	__syncthreads();

	if (x <= int_widht && y <= h && y >= 1)
	{
		uchar4 generated_int = { 0 };
		unsigned char result;
		unsigned char value;
		uchar4 int_1;
		uchar4 mem;
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			mem = corners[0];
			int_1.x = mem.y;
			int_1.y = mem.z;
			int_1.z = mem.w;
			int_1.w = top_extension[0][0].x;
		}
		else if (threadIdx.x == 0)
		{
			mem = left_extension[0][threadIdx.y];
			int_1.x = mem.y;
			int_1.y = mem.z;
			int_1.z = mem.w;
			int_1.w = shared_memory[threadIdx.y - 1][threadIdx.x].x;
		}
		else if (threadIdx.y == 0)
		{
			mem = top_extension[0][threadIdx.x - 1];
			int_1.x = mem.y;
			int_1.y = mem.z;
			int_1.z = mem.w;
			int_1.w = top_extension[0][threadIdx.x].x;
		}
		else
		{
			mem = shared_memory[threadIdx.y - 1][threadIdx.x - 1];
			int_1.x = mem.y;
			int_1.y = mem.z;
			int_1.z = mem.w;
			int_1.w = shared_memory[threadIdx.y - 1][threadIdx.x].x;
		}
		result = 0;
		value = int_1.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_1.x = result;
		result = 0;
		value = int_1.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_1.y = result;
		result = 0;
		value = int_1.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_1.z = result;
		result = 0;
		value = int_1.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_1.w = result;
		uchar4 int_2;
		if (threadIdx.y == 0)
		{
			mem = top_extension[0][threadIdx.x];
			int_2.x = mem.y;
			int_2.y = mem.z;
			int_2.z = mem.w;
			int_2.w = threadIdx.x == THREADS_X - 1 ? corners[1].x : top_extension[0][threadIdx.x + 1].x;
		}
		else
		{
			mem = shared_memory[threadIdx.y - 1][threadIdx.x];
			int_2.x = mem.y;
			int_2.y = mem.z;
			int_2.z = mem.w;
			int_2.w = shared_memory[threadIdx.y - 1][threadIdx.x + 1].x;
		}
		result = 0;
		value = int_2.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.x = result;
		result = 0;
		value = int_2.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.y = result;
		result = 0;
		value = int_2.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.z = result;
		result = 0;
		value = int_2.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_2.w = result;
		uchar4 int_3;
		if (threadIdx.x == 0)
		{
			mem = left_extension[0][threadIdx.y];
			int_3.x = mem.y;
			int_3.y = mem.z;
			int_3.z = mem.w;
			int_3.w = shared_memory[threadIdx.y][threadIdx.x].x;
		}
		else
		{
			mem = shared_memory[threadIdx.y][threadIdx.x - 1];
			int_3.x = mem.y;
			int_3.y = mem.z;
			int_3.z = mem.w;
			int_3.w = shared_memory[threadIdx.y][threadIdx.x].x;
		}
		result = 0;
		value = int_3.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_3.x = result;
		result = 0;
		value = int_3.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_3.y = result;
		result = 0;
		value = int_3.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_3.z = result;
		result = 0;
		value = int_3.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_3.w = result;
		uchar4 int_4;
		mem = shared_memory[threadIdx.y][threadIdx.x];
		int_4.x = mem.y;
		int_4.y = mem.z;
		int_4.z = mem.w;
		int_4.w = threadIdx.x == THREADS_X - 1 ? right_extension[0][threadIdx.y].x : shared_memory[threadIdx.y][threadIdx.x + 1].x;
		result = 0;
		value = int_4.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.x = result;
		result = 0;
		value = int_4.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.y = result;
		result = 0;
		value = int_4.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.z = result;
		result = 0;
		value = int_4.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_4.w = result;
		uchar4 int_5;
		mem = shared_memory[threadIdx.y][threadIdx.x];
		int_5.x = mem.y;
		int_5.y = mem.z;
		int_5.z = mem.w;
		int_5.w = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x].x : shared_memory[threadIdx.y + 1][threadIdx.x].x;
		result = 0;
		value = int_5.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_5.x = result;
		result = 0;
		value = int_5.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_5.y = result;
		result = 0;
		value = int_5.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_5.z = result;
		result = 0;
		value = int_5.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_5.w = result;
		uchar4 int_6;
		mem = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x] : shared_memory[threadIdx.y + 1][threadIdx.x];
		int_6.x = mem.y;
		int_6.y = mem.z;
		int_6.z = mem.w;
		if (threadIdx.x == THREADS_X - 1)
		{
			int_6.w = threadIdx.y == THREADS_Y - 1 || y == h ? corners[3].x : shared_memory[threadIdx.y + 1][threadIdx.x + 1].x;
		}
		else
		{
			int_6.w = threadIdx.y == THREADS_Y - 1 || y == h ? bottom_extension[0][threadIdx.x + 1].x : shared_memory[threadIdx.y + 1][threadIdx.x + 1].x;
		}

		result = 0;
		value = int_6.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.x = result;
		result = 0;
		value = int_6.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.y = result;
		result = 0;
		value = int_6.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.z = result;
		result = 0;
		value = int_6.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_6.w = result;
		uchar4 int_7;
		if (threadIdx.y == 0)
		{
			mem = threadIdx.x >= THREADS_X - 1 ? corners[1] : top_extension[0][threadIdx.x + 1];
			int_7.x = mem.y;
			int_7.y = mem.z;
			int_7.z = mem.w;
			int_7.w = top_extension[0][threadIdx.x + 2].x;
		}
		else
		{
			mem = shared_memory[threadIdx.y - 1][threadIdx.x + 1];
			int_7.x = mem.y;
			int_7.y = mem.z;
			int_7.z = mem.w;
			int_7.w = shared_memory[threadIdx.y - 1][threadIdx.x + 2].x;
		}
		result = 0;
		value = int_7.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_7.x = result;
		result = 0;
		value = int_7.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_7.y = result;
		result = 0;
		value = int_7.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_7.z = result;
		result = 0;
		value = int_7.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_7.w = result;
		uchar4 int_8;
		mem = shared_memory[threadIdx.y][threadIdx.x + 1];
		int_8.x = mem.y;
		int_8.y = mem.z;
		int_8.z = mem.z;
		int_8.w = shared_memory[threadIdx.y][threadIdx.x + 2].x;
		result = 0;
		value = int_8.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_8.x = result;
		result = 0;
		value = int_8.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_8.y = result;
		result = 0;
		value = int_8.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_8.z = result;
		result = 0;
		value = int_8.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_8.w = result;
		uchar4 int_9;
		if (threadIdx.y == THREADS_Y - 1 || y == h)
		{
			mem = threadIdx.x == THREADS_X - 1 ? corners[3] : bottom_extension[0][threadIdx.x + 1];
		}
		else
		{
			mem = shared_memory[threadIdx.y + 1][threadIdx.x + 1];
		}

		int_9.x = mem.y;
		result = 0;
		value = int_9.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		int_9.x = result;

		generated_int.x = (
			int_1.y + int_2.x * (-2) + int_2.w +
			int_3.y * (-2) + int_4.x * 4 + int_4.w * (-2) +
			int_5.y + int_6.x * (-2) + int_6.w
			);

		generated_int.y = (
			int_1.z + int_2.y * (-2) + int_7.x +
			int_3.z * (-2) + int_4.y * 4 + int_8.x * (-2) +
			int_5.z + int_6.y * (-2) + int_9.x
			);

		generated_int.z = (
			int_1.w + int_2.z * (-2) + int_7.y +
			int_3.w * (-2) + int_4.z * 4 + int_8.y * (-2) +
			int_5.w + int_6.z * (-2) + int_9.y
			);

		generated_int.w = (
			int_2.x + int_2.w * (-2) + int_7.z +
			int_4.x * (-2) + int_4.w * 4 + int_8.z * (-2) +
			int_6.x + int_6.w * (-2) + int_9.z
			);

		const int output_int_width = outPitch / sizeof(int);


		result = 0;
		value = generated_int.x;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.x = result;
		result = 0;
		value = generated_int.y;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.y = result;
		result = 0;
		value = generated_int.z;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.z = result;
		result = 0;
		value = generated_int.w;
		for (int g = 0; g < 8; g++)
		{
			result <<= 1;
			result |= (value & 1);
			value >>= 1;
		}
		generated_int.w = result;
		thread_output[(y - 1) * output_int_width + x] = generated_int;
	}
}