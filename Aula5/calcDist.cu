#include <math.h>
#include <stdio.h>

//para tomada de tempo
#include "clock_timer.h"

#define PRINT_VETOR

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if (err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); } }

//funcao para execucao sequencial
float maxElementSeq(float *a, int n) {
   int i;
   float aux = 0;
   for(i = 0; i < n; i++)
       aux = max(a[i], aux);
   return sup;
}

//kernel para execucao paralela na GPU (os tres vetores possuem o mesmo tamanho n)
__global__ void calcDist(const int point, const float *points, float *dist, const int tam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dist[i] = 0;
    if (i < tam) {
        for (int j = 0; j < 3; j++)
            dist[i] += (points[3 * i + j] - points[3 * point + j]) * (points[3 * i + j] - points[3 * point + j]);
        dist[i] = sqrt(dist[i]);
    }
}

//kernel para execucao paralela em CUDA (!!assume-se que o tamanho do vetor sera sempre potencia de 2!!)
//e que foram disparadas uma thread para cada elemento, em um unico bloco
__global__ void maxElementGPU(float *a) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int n = blockDim.x;
   int offset;
   float aux;

   for(offset=1; offset<n; offset*=2) {
      if(threadIdx.x >= offset) {
         aux = a[i-offset];
      }
      __syncthreads();
      if(threadIdx.x >= offset) {
         a[i] = max(aux, a[i]);
      }
      __syncthreads();
   }
}

//funcao para imprimir os elementos de um vetor
void printVetor(float *vetor, int n) {
   for(int i = 0; i < n; i++)
      printf("%.1f ", vetor[i]);
   printf("\n");
}

int main(int argc, const char *argv[]) {

    srand(time(NULL));

	// vetores de pontos, suas distancias e o ponto escolhido
	float *h_point, *d_point, *h_dist, *d_dist, *s_dist;
	int chosenPoint = 0;
	// para tamanho do vetor
	int n, n_bytes;
	// Distancias
	float correctMaxDist, maxDist;
    //para medidas de tempo
	float inicio, fim;
    float tempo_seq, tempo_ida, tempo_volta;
    cudaEvent_t start, stop;

    //verifica os parametros de entrada
    if(argc != 2) {
      printf("Digite: %s <no. de pontos>\n", argv[0]);
      exit(EXIT_FAILURE);
    }

    //armazena os parametros de entrada
    n = atol(argv[1]);
    //calcula o tamanho em bytes dos vetores
    n_bytes = n * sizeof(float);

    //aloca memoria na CPU (host)
    h_point = (float*) malloc (3 * n_bytes);
    if (h_point == NULL) {
      printf("Erro malloc vetor de pontos\n");
      exit(EXIT_FAILURE);
    }
    h_dist = (float*) malloc (n_bytes);
    if (h_dist == NULL) {
      printf("Erro malloc vetor de pontos\n");
      exit(EXIT_FAILURE);
    }
    s_dist = (float*) malloc (n_bytes);
    if (s_dist == NULL) {
      printf("Erro malloc vetor de pontos\n");
      exit(EXIT_FAILURE);
    }

    // Cria n pontos aleatórios
    for(int i = 0; i < n; i++)
        for(int j = 0; j < 3; j++)
            h_point[3*i+j] = rand() % (n / 100);

    //!!! ------------------------ executa sequencial ---------------------------------- !!!//
    GET_TIME(inicio);
    for(int i = 0; i < n; i++) {
        s_dist[i] = 0;
        for(int j = 0; j < 3; j++)
            s_dist[i] += (h_point[3*chosenPoint+j] - h_point[3*i+j])*(h_point[3*chosenPoint+j] - h_point[3*i+j]);
        s_dist[i] = sqrt(s_dist[i]);
    }

    correctMaxDist = maxElementsSeq(s_dist, n);
    GET_TIME(fim);


    tempo_seq = fim-inicio; // em segundos

    //!!! ------------------------ executa em paralelo em CUDA -------------------------- !!!//
    //dispara o kernel
    GET_TIME(inicio);
    int n_threads = 256; //numero de threads por bloco
	int n_blocos = n / n_threads; //numero de blocos

    //aloca espaco para os vetores na GPU
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_point, 3 * n_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_dist, n_bytes));

    //copia os vetores de entrada da CPU para a GPU (host para device)
    CUDA_SAFE_CALL(cudaMemcpy(d_point, h_point, 3*n_bytes, cudaMemcpyHostToDevice));

    GET_TIME(fim);
    printf("Kernel com %d bloco(s) de %d threads:\n", n_blocos, n_threads);
    tempo_ida = fim-inicio; // em segundos

    //mede o tempo de execucao do kernel apenas
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    //inicia a contagem de tempo
    CUDA_SAFE_CALL(cudaEventRecord(start));
    //dispara o kernel

    calcDist <<<n_blocos, n_threads>>> (chosenPoint, d_point, d_dist, n);
    CUDA_SAFE_CALL(cudaGetLastError());
    maxElementGPU <<<n_blocos, n_threads>>> (d_dist, n);
    CUDA_SAFE_CALL(cudaGetLastError());

    //copia resultado da GPU para a CPU (device para host)
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    //finaliza a contagem de tempo
    float deltaEvent = 0;
    //(tempo  decorrido entre o registro do event start e do evento stop, em milisegundos)
    CUDA_SAFE_CALL(cudaEventElapsedTime(&deltaEvent, start, stop));

    GET_TIME(inicio);

    //copia resultado da GPU para a CPU (device para host)
    CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, n_bytes, cudaMemcpyDeviceToHost))
    GET_TIME(fim);
    tempo_volta = fim-inicio; // em segundos

    maxDist = h_dist[n-1];
	printf("\n");
	printf("%.10f %.10f\n", maxDist, correctMaxDist);
    if (fabs(maxDist - correctMaxDist) > 1e-5) {
        fprintf(stderr, "resultado maximo incorreto!\n");
        exit(EXIT_FAILURE);
    }
    //libera a memoria na GPU
    CUDA_SAFE_CALL(cudaFree(d_point))
    CUDA_SAFE_CALL(cudaFree(d_dist))


    printf("PASSOU NO TESTE\n");

    //libera a memoria na CPU
    free(h_point);
    free(h_dist);
    free(s_dist);

    //------------------------------- imprime dos tempos de execucao ----------------------//
    printf("Tempo sequencial = %f seg \n", tempo_seq.count());
    printf("Tempo paralelo   = %f seg \n\n", deltaEvent/1000);

    printf("Tempo ida (kernel)   = %f seg \n", tempo_ida);
    printf("Tempo volta (kernel) = %f seg \n", tempo_volta);
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_SAFE_CALL(cudaDeviceReset());

}
