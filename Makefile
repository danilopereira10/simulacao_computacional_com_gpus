###############################################################################################
######################### Compilação com suporte a múltiplos arquivos #########################
###############################################################################################

.PHONY: build clean

LAB = simulator

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
GFLAGS  = -std=c99 -Wall -Werror

build: $(OBJECTS)
	gcc $(GFLAGS) *.o -o $(LAB) -lm -O3

%.o: %.c
	gcc $(GFLAGS) -g -c $< -o $@ -O3

clean:
	rm -f *.o

