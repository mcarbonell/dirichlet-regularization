CC ?= gcc
CFLAGS ?= -O3 -std=c11 -shared -fPIC -Wall -Wextra
LDFLAGS ?= -pthread -lm

TARGET = libspectral_dma_kernel.so

ifeq ($(OS),Windows_NT)
    TARGET = spectral_dma_kernel.dll
    CFLAGS = -O3 -std=c11 -shared -Wall -Wextra
    LDFLAGS = -lm
endif

all: $(TARGET)

$(TARGET): spectral_dma_kernel.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.dll *.so *.dylib
