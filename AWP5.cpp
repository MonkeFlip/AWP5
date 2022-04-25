#include <iostream>
#include <iomanip>

#include "HostFilter.h"

using namespace std;

unsigned char* ExtendBorders(
	unsigned char* inputData,
	int w,
	int h
)
{
	const int new_width = w + 2;
	const int new_height = h + 2;

	unsigned char* output_data = new unsigned char[new_width * new_height];

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			output_data[(y + 1) * new_width + x + 1] = inputData[y * w + x];
		}
	}

	output_data[0] = inputData[0];
	output_data[new_width - 1] = inputData[w - 1];
	output_data[new_width * (new_height - 1)] = inputData[w * (h - 1)];
	output_data[new_width * new_height - 1] = inputData[w * h - 1];

	for (int x = 0; x < w; x++)
	{
		output_data[x + 1] = inputData[x];
		output_data[(new_height - 1) * new_width + x + 1] = inputData[w * (h - 1) + x];
	}

	for (int y = 0; y < h; y++)
	{
		output_data[(y + 1) * new_width] = inputData[y * w];
		output_data[(y + 1) * new_width + new_width - 1] = inputData[y * w + w - 1];
	}

	return output_data;
}

RGB* ExtendBorders(
	RGB* inputData,
	int w,
	int h
)
{
	const int new_width = w + 2;
	const int new_height = h + 2;

	RGB* output_data = new RGB[new_width * new_height];

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			output_data[(y + 1) * new_width + x + 1] = inputData[y * w + x];
		}
	}

	output_data[0] = inputData[0];
	output_data[new_width - 1] = inputData[w - 1];
	output_data[new_width * (new_height - 1)] = inputData[w * (h - 1)];
	output_data[new_width * new_height - 1] = inputData[w * h - 1];

	for (int x = 0; x < w; x++)
	{
		output_data[x + 1] = inputData[x];
		output_data[(new_height - 1) * new_width + x + 1] = inputData[w * (h - 1) + x];
	}

	for (int y = 0; y < h; y++)
	{
		output_data[(y + 1) * new_width] = inputData[y * w];
		output_data[(y + 1) * new_width + new_width - 1] = inputData[y * w + w - 1];
	}

	return output_data;
}

void GrayFilter(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight
)
{
	for (int y = 1; y <= h; y++)
	{
		for (int x = 1; x <= w; x++)
		{
			unsigned char a1 = inputData[(y - 1) * zoomedWidth + x - 1];
			unsigned char f1 = 0;
			for (int g = 0; g < 8; g++)
			{
				f1 <<= 1;
				f1 |= (a1 & 1);
				a1 >>= 1;
			}
			unsigned char a2 = inputData[(y - 1) * zoomedWidth + x];
			unsigned char f2 = 0;
			for (int g = 0; g < 8; g++)
			{
				f2 <<= 1;
				f2 |= (a2 & 1);
				a2 >>= 1;
			}
			unsigned char a3 = inputData[(y - 1) * zoomedWidth + x + 1];
			unsigned char f3 = 0;
			for (int g = 0; g < 8; g++)
			{
				f3 <<= 1;
				f3 |= (a3 & 1);
				a3 >>= 1;
			}
			unsigned char a4 = inputData[y * zoomedWidth + x - 1];
			unsigned char f4 = 0;
			for (int g = 0; g < 8; g++)
			{
				f4 <<= 1;
				f4 |= (a4 & 1);
				a4 >>= 1;
			}
			unsigned char a5 = inputData[y * zoomedWidth + x];
			unsigned char f5 = 0;
			for (int g = 0; g < 8; g++)
			{
				f5 <<= 1;
				f5 |= (a5 & 1);
				a5 >>= 1;
			}
			unsigned char a6 = inputData[y * zoomedWidth + x + 1];
			unsigned char f6 = 0;
			for (int g = 0; g < 8; g++)
			{
				f6 <<= 1;
				f6 |= (a6 & 1);
				a6 >>= 1;
			}
			unsigned char a7 = inputData[(y + 1) * zoomedWidth + x - 1];
			unsigned char f7 = 0;
			for (int g = 0; g < 8; g++)
			{
				f7 <<= 1;
				f7 |= (a7 & 1);
				a7 >>= 1;
			}
			unsigned char a8 = inputData[(y + 1) * zoomedWidth + x];
			unsigned char f8 = 0;
			for (int g = 0; g < 8; g++)
			{
				f8 <<= 1;
				f8 |= (a8 & 1);
				a8 >>= 1;
			}
			unsigned char a9 = inputData[(y + 1) * zoomedWidth + x + 1];
			unsigned char f9 = 0;
			for (int g = 0; g < 8; g++)
			{
				f9 <<= 1;
				f9 |= (a9 & 1);
				a9 >>= 1;
			}

			unsigned char result =
				f1 + f2 * (-2) + f3 +
				f4 * (-2) + f5 * 4 + f6 * (-2) +
				f7 + f8 * (-2) + f9;

			unsigned char res = 0;
			for (int g = 0; g < 8; g++)
			{
				res <<= 1;
				res |= (result & 1);
				result >>= 1;
			}
			outputData[(y - 1) * w + (x - 1)] = res;

		}
	}
}


void HostGrayFilter(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h
)
{
	unsigned char* zoomedData = ExtendBorders(inputData, w, h);

	const int zoomedWidth = w + 2;
	const int zoomedHeight = h + 2;

	GrayFilter(zoomedData, outputData, w, h, zoomedWidth, zoomedHeight);
}

void RgbFilter(
	RGB* inputData,
	RGB* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight
)
{
	for (int y = 1; y < h; y++)
	{
		for (int x = 1; x < w; x++)
		{
			RGB result = { 0 };

			// top row
			RGB t1 = inputData[(y - 1) * zoomedWidth + x - 1];
			RGB f1 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f1.g <<= 1;
				f1.g |= (t1.g & 1);
				t1.g >>= 1;

				f1.b <<= 1;
				f1.b |= (t1.b & 1);
				t1.b >>= 1;

				f1.r <<= 1;
				f1.r |= (t1.r & 1);
				t1.r >>= 1;
			}
			RGB t2 = inputData[(y - 1) * zoomedWidth + x];
			RGB f2 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f2.g <<= 1;
				f2.g |= (t2.g & 1);
				t2.g >>= 1;

				f2.b <<= 1;
				f2.b |= (t2.b & 1);
				t2.b >>= 1;

				f2.r <<= 1;
				f2.r |= (t2.r & 1);
				t2.r >>= 1;
			}
			RGB t3 = inputData[(y - 1) * zoomedWidth + x + 1];
			RGB f3 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f3.g <<= 1;
				f3.g |= (t3.g & 1);
				t3.g >>= 1;

				f3.b <<= 1;
				f3.b |= (t3.b & 1);
				t3.b >>= 1;

				f3.r <<= 1;
				f3.r |= (t3.r & 1);
				t3.r >>= 1;
			}

			// middle row
			RGB m1 = inputData[(y)*zoomedWidth + x - 1];
			RGB f4 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f4.g <<= 1;
				f4.g |= (m1.g & 1);
				m1.g >>= 1;

				f4.b <<= 1;
				f4.b |= (m1.b & 1);
				m1.b >>= 1;

				f4.r <<= 1;
				f4.r |= (m1.r & 1);
				m1.r >>= 1;
			}
			RGB m2 = inputData[(y)*zoomedWidth + x];
			RGB f5 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f5.g <<= 1;
				f5.g |= (m2.g & 1);
				m2.g >>= 1;

				f5.b <<= 1;
				f5.b |= (m2.b & 1);
				m2.b >>= 1;

				f5.r <<= 1;
				f5.r |= (m2.r & 1);
				m2.r >>= 1;
			}
			RGB m3 = inputData[(y)*zoomedWidth + x + 1];
			RGB f6 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f6.g <<= 1;
				f6.g |= (m3.g & 1);
				m3.g >>= 1;

				f6.b <<= 1;
				f6.b |= (m3.b & 1);
				m3.b >>= 1;

				f6.r <<= 1;
				f6.r |= (m3.r & 1);
				m3.r >>= 1;
			}

			// bottom row
			RGB b1 = inputData[(y + 1) * zoomedWidth + x - 1];
			RGB f7 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f7.g <<= 1;
				f7.g |= (b1.g & 1);
				b1.g >>= 1;

				f7.b <<= 1;
				f7.b |= (b1.b & 1);
				b1.b >>= 1;

				f7.r <<= 1;
				f7.r |= (b1.r & 1);
				b1.r >>= 1;
			}
			RGB b2 = inputData[(y + 1) * zoomedWidth + x];
			RGB f8 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f8.g <<= 1;
				f8.g |= (b2.g & 1);
				b2.g >>= 1;

				f8.b <<= 1;
				f8.b |= (b2.b & 1);
				b2.b >>= 1;

				f8.r <<= 1;
				f8.r |= (b2.r & 1);
				b2.r >>= 1;
			}
			RGB b3 = inputData[(y + 1) * zoomedWidth + x + 1];
			RGB f9 = { 0 };
			for (int g = 0; g < 8; g++)
			{
				f9.g <<= 1;
				f9.g |= (b3.g & 1);
				b3.g >>= 1;

				f9.b <<= 1;
				f9.b |= (b3.b & 1);
				b3.b >>= 1;

				f9.r <<= 1;
				f9.r |= (b3.r & 1);
				b3.r >>= 1;
			}

			result.r =
				(
					f1.r + f2.r * (-2) + f3.r +
					f4.r * (-2) + f5.r * 4 + f6.r * (-2) +
					f7.r + f8.r * (-2) + f9.r
					);

			result.g =
				(
					f1.g + f2.g * (-2) + f3.g +
					f4.g * (-2) + f5.g * 4 + f6.g * (-2) +
					f7.g + f8.g * (-2) + f9.g
					);

			result.b =
				(
					f1.b + f2.b * (-2) + f3.b +
					f4.b * (-2) + f5.b * 4 + f6.b * (-2) +
					f7.b + f8.b * (-2) + f9.b
					);
			RGB res = { 0 };
			for (int g = 0; g < 8; g++)
			{
				res.g <<= 1;
				res.g |= (result.g & 1);
				result.g >>= 1;

				res.b <<= 1;
				res.b |= (result.b & 1);
				result.b >>= 1;

				res.r <<= 1;
				res.r |= (result.r & 1);
				result.r >>= 1;
			}
			outputData[(y - 1) * w + x - 1] = res;

		}
	}
}

void HostRgbFilter(
	RGB* inputData,
	RGB* outputData,
	const int w,
	const int h
)
{
	RGB* zoomedData = ExtendBorders(inputData, w, h);

	const int zoomedWidth = w + 2;
	const int zoomedHeight = h + 2;

	RgbFilter(zoomedData, outputData, w, h, zoomedWidth, zoomedHeight);
}