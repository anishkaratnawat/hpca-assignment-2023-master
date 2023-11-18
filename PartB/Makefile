sim: data/64.in data/128.in dilated_conv_sim

server: data/2048.in data/1024.in data/128.in data/64.in dilated_conv_server

dilated_conv_server: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv_server -I ./header

dilated_conv_sim: main.cu header/gpu_thread.h
	nvcc main.cu -o dilated_conv -I ./header --cudart shared -gencode arch=compute_70,code=compute_70

data/generate: data/generate.cpp
	g++ ./data/generate.cpp -o ./data/generate

data/64.in: data/generate
	./data/generate 64

data/128.in: data/generate
	./data/generate 128

data/1024.in: data/generate
	./data/generate 1024
data/11.in: data/generate
	./data/generate 11
data/2048.in: data/generate
	./data/generate 2048
run_sim: data/128.in dilated_conv_sim
	./dilated_conv_sim -i data/128.in -k data/64.in
	
run_server: data/2048.in data/128.in data/1024.in data/64.in data/11.in dilated_conv_server
	./dilated_conv_server -i data/2048.in -k data/64.in
	./dilated_conv_server -i data/128.in -k data/64.in
	./dilated_conv_server -i data/1024.in -k data/64.in
	./dilated_conv_server -i data/64.in -k data/11.in

clean:
	rm -f dilated_conv_sim dilated_conv_server
