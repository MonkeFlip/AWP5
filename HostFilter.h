typedef struct RGB
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

unsigned char* ExtendBorders(unsigned char* inputData, int w, int h);

RGB* ExtendBorders(RGB* inputData, int w, int h);

void HostGrayFilter(unsigned char* inputData, unsigned char* outputData, const int w, const int h);

void HostRgbFilter(RGB* inputData, RGB* outputData, const int w, const int h);