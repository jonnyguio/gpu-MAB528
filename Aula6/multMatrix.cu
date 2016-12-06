/* Prof.: Silvana Rossetto */

#include <cstdio>
#include <cstring>

//para tomada de tempo
#include "clock_timer.h"

#define DEBUG 0
#define MAKESEQ 0

#define TAM_BLOCO 32

//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); } }


void printMatrix(FILE *out, float *a, int n, const char* txt) {
    int i, j;
    if (!DEBUG && out == stdout) return;
    fprintf(out, "%s\n", txt);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            fprintf(out, "%f ", a[i * n + j]);
        fprintf(out, "\n");
    }
    fprintf(out, "\n");
}

bool fillMatrix(FILE *arq, float *matrix, int n) {
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            fscanf(arq, "%f", &matrix[i * n + j]);
        }
    }
    return true;
}

//funcao para execucao sequencial
void multMatrixSeq(float *a, float *b, float *c, int n) {
    int i, j, k;
    float soma;
    for(i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            soma = 0;
            for (j = 0; j < n; j++) {
                soma += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = soma;
        }
    }
}

//multiplicacao de matriz com memoria compartilhada
__global__ void multMatrixGPUShared(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //coordenadas locais da thread
    int i_bloco = threadIdx.x;
    int j_bloco = threadIdx.y;

    //mem´oria compartilhada para a submatriz de A
    __shared__ float asub[TAM_BLOCO][TAM_BLOCO];

    //mem´oria compartilhada para a submatriz de B
    __shared__ float bsub[TAM_BLOCO][TAM_BLOCO];

    //calcula o elemento C(i,j)
    float valor = 0;
    for(int passo = 0; passo < n; passo += TAM_BLOCO) {

        //cada thread carrega um elemento de A e B
        asub[i_bloco][j_bloco] = a[( i )* n + passo + j_bloco];
        bsub[i_bloco][j_bloco] = b[(passo + i_bloco) * n + j];

        //sincroniza para terminar a c´opia
        __syncthreads();

        //cada thread computa um elemento
        for (int k = 0; k < TAM_BLOCO; k++) {
            valor += asub[i_bloco][k] * bsub[k][j_bloco];
        }

        //sincroniza para terminar a computa¸c~ao
        __syncthreads();
    }

    //escreve o valor calculado na matriz de saida
    c[i * n + j] = valor;
}

//multiplicacao de matriz sem memoria compartilhada
__global__ void multMatrixGPU(float *a, float *b, float *c, int n) {
    //indices da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //calcula o elemento C(i,j)
    float valor = 0;
    for (int k = 0; k < n; k++) {
        valor += a[i * n + k] * b[k * n + j];
    }
    //escreve o valor calculado na matriz de saida
    c[i * n + j] = valor;
}

char* concat(const char *s1, const char *s2)
{
    int size = 0;
    size += strlen(s1);
    size += strlen(s2);
    size++;
    char *result = (char *) malloc(size);//+1 for the zero-terminator

    if (result == NULL) {
        printf("Erro malloc on concat\n");
        exit(EXIT_FAILURE);
    }

    strcpy(result, s1);
    strcat(result, s2);

    return result;
}

//funcao principal
int main(int argc, char const *argv[]) {
    // Matrizes
    float *h_matrixA, *h_matrixB, *h_matrixC, *d_matrixA, *d_matrixB, *d_matrixC;
    // Arquivo para Matrizes
    FILE *fMatA, *fMatB, *fMatC_seq, *fMatC_gpu;

    int n, n_bytes;
    //para medidas de tempo
    double inicio, fim;
    double tempo_seq = 0, tempo_gpu = 0, tempo_gpu_shared = 0;
    float delta_eventos, delta_eventos2;
    cudaEvent_t start, stop;

    //verifica os parametros de entrada
    if(argc < 5) {
        printf("Digite: %s <tamanho das matrizes> <matriz A> <matriz B> <matriz C>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    n = atol(argv[1]);
    const char *strFMatA = argv[2], *strFMatB = argv[3], *strFMatC_seq = concat(argv[4], "_seq.txt"),  *strFMatC_gpu = concat(argv[4], "_gpu.txt");

    fMatA = fopen(strFMatA, "r");  fMatB = fopen(strFMatB, "r");  fMatC_seq = fopen(strFMatC_seq, "w"); fMatC_gpu = fopen(strFMatC_gpu, "w");

    //calcula o tamanho em bytes do vetor
    n_bytes = n * n * sizeof(float);

    //aloca memoria na CPU (host)
    h_matrixA = (float*) malloc (n_bytes);
    h_matrixB = (float*) malloc (n_bytes);
    h_matrixC = (float*) malloc (n_bytes);
    if ((h_matrixA == NULL) || (h_matrixB == NULL) || ((h_matrixC) == NULL)) {
        printf("Erro malloc matriz\n");
        exit(EXIT_FAILURE);
    }

    fillMatrix(fMatA, h_matrixA, n);
    fillMatrix(fMatB, h_matrixB, n);
    printMatrix(stdout, h_matrixA, n, "a");
    printMatrix(stdout, h_matrixB, n, "b");

    //!!! ------------------------ executa sequencial ---------------------------------- !!!//
    if (MAKESEQ) {
        GET_TIME(inicio);
        multMatrixSeq(h_matrixA, h_matrixB, h_matrixC, n);
        fprintf(stdout, "Imprimindo sequencial no arquivo:\n");
        printMatrix(fMatC_seq, h_matrixC, n, "c_seq");
        GET_TIME(fim);
    }

    tempo_seq = MAKESEQ ? fim-inicio : 0; // em segundos

    //!!! ------------------------ executa em paralelo em CUDA -------------------------- !!!//

    // !! Começando sem memória compartilhada !! //
    GET_TIME(inicio);

    //aloca espaco para as matrizes na GPU
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixA, n_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixB, n_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixC, n_bytes));

    //copia o vetor de entrada da CPU para a GPU (host para device)
    CUDA_SAFE_CALL(cudaMemcpy(d_matrixA, h_matrixA, n_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_matrixB, h_matrixB, n_bytes, cudaMemcpyHostToDevice));

    //dispara o kernel
    dim3 threadsBloco(TAM_BLOCO, TAM_BLOCO);
    dim3 blocosGrade(n/threadsBloco.x, n/threadsBloco.y);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));

    multMatrixGPU <<<blocosGrade, threadsBloco>>> (d_matrixA, d_matrixB, d_matrixC, n);

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaGetLastError());

    CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

    //copia resultado da GPU para a CPU (device para host)
    CUDA_SAFE_CALL(cudaMemcpy(h_matrixC, d_matrixC, n_bytes, cudaMemcpyDeviceToHost))
    printMatrix(fMatC_gpu, h_matrixC, n, "c_gpu");
    GET_TIME(fim);

    tempo_gpu = fim - inicio;

    //libera a memoria na GPU
    CUDA_SAFE_CALL(cudaFree(d_matrixA));
    CUDA_SAFE_CALL(cudaFree(d_matrixB));
    CUDA_SAFE_CALL(cudaFree(d_matrixC));

    // !! Começando com memória compartilhada !! //
    GET_TIME(inicio);

    //aloca espaco para as matrizes na GPU
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixA, n_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixB, n_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_matrixC, n_bytes));

    //copia o vetor de entrada da CPU para a GPU (host para device)
    CUDA_SAFE_CALL(cudaMemcpy(d_matrixA, h_matrixA, n_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_matrixB, h_matrixB, n_bytes, cudaMemcpyHostToDevice));

    //dispara o kernel

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));

    multMatrixGPUShared <<<blocosGrade, threadsBloco>>> (d_matrixA, d_matrixB, d_matrixC, n);

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaGetLastError());

    CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos2, start, stop));

    //copia resultado da GPU para a CPU (device para host)
    CUDA_SAFE_CALL(cudaMemcpy(h_matrixC, d_matrixC, n_bytes, cudaMemcpyDeviceToHost))
    printMatrix(fMatC_gpu, h_matrixC, n, "c_gpu");
    GET_TIME(fim);

    tempo_gpu_shared = fim - inicio;

    //libera a memoria na GPU
    CUDA_SAFE_CALL(cudaFree(d_matrixA));
    CUDA_SAFE_CALL(cudaFree(d_matrixB));
    CUDA_SAFE_CALL(cudaFree(d_matrixC));

    //libera a memoria na CPU*/
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixC);

    //------------------------------- imprime dos tempos de execucao ----------------------//
    if (MAKESEQ)
        fprintf(stdout, "Tempo sequencial = %.10f seg \n", tempo_seq);
    fprintf(stdout, "Tempo paralelo = %f seg \n", tempo_gpu + delta_eventos/1000);
    //printf("Tempo paralelo = %.10f seg \n", tempo_gpu);
    fprintf(stdout, "Tempo paralelo com memória compartilhada = %.10f seg \n", tempo_gpu_shared + delta_eventos2/1000);

    if (MAKESEQ) {
        fprintf(stdout, "Speedup tempo paralelo: %.10f\n", tempo_seq / (tempo_gpu + delta_eventos/1000));
        fprintf(stdout, "Speedup tempo paralelo com memória compartilhada: %.10f\n", tempo_seq / (tempo_gpu_shared + delta_eventos2/1000));
    }

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_SAFE_CALL(cudaDeviceReset());

    return 0;
}
       
