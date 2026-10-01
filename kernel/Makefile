CC ?= gcc
CFLAGS ?= -O3 -shared -fPIC -Wall
LDFLAGS ?= -lm -lpthread

TARGET = libspectral_dma_kernel.so

ifeq ($(OS),Windows_NT)
    TARGET = spectral_dma_kernel.dll
    LDFLAGS = -lm
endif

all: $(TARGET)

$(TARGET): spectral_dma_kernel.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.dll *.so *.dylib
