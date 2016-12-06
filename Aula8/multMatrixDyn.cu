/* Prof.: Silvana Rossetto */
/* Codigo: Soma os elementos de um vetor em CUDA usando o mesmo algoritmo de soma de prefixos */

/* Para compilar: nvcc -o somaelementos somaelementos.cu */

#include <cstdio>
#include <cstring>

//para tomada de tempo
#include "clock_timer.h"

#define DEBUG 0
#define MAKESEQ 0

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

bool checkMatrix(const float *m1, const float *m2, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (m1[i * n + j] != m2[i * n + j]) {
                fprintf(stdout, "Error on [%d][%d] -> %f != %f\n", i, j, m1[i *n + j], m2[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

//funcao para execucao sequencial
void multMatrixSeq(const float *a, const float *b, float *c, int n) {
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
__global__ void multMatrixGPUSharedDyn(float *a, float *b, float *c, int n) {
    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //coordenadas locais da thread
    int i_bloco = threadIdx.x;
    int j_bloco = threadIdx.y;

    extern __shared__ float mat_sub[];
    //mem´oria compartilhada para a submatriz de A
    float* Asub = (float*) &mat_sub;
    //mem´oria compartilhada para a submatriz de B
    float* Bsub= (float*) &Asub[blockDim.x * blockDim.y];

    float valor = 0;

    for(int passo=0; passo<n; passo+=blockDim.y) {
        Asub[i_bloco*blockDim.y+j_bloco] = a[i*n+passo+j_bloco];
        Bsub[i_bloco*blockDim.y+j_bloco] = b[(passo+i_bloco)*n+j];
        __syncthreads();
        for (int k = 0; k < blockDim.y; k++) {
            valor += Asub[i_bloco*blockDim.y+k] * Bsub[k*blockDim.y+j_bloco];
        }
        __syncthreads();
    }
    c[i*n+j] = valor;
}

char* concat(const char *s1, const char *s2) {
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
    float *h_matrixA, *h_matrixB, *h_matrixCseq, *h_matrixCgpu, *d_matrixA, *d_matrixB, *d_matrixC;
    // Arquivo para Matrizes
    FILE *fMatA, *fMatB, *fMatC_seq, *fMatC_gpu;

    int n, n_bytes, sharedBytes, sizeBlock;
    //para medidas de tempo
    double inicio, fim;
    double tempo_seq = 0, tempo_gpu_dyn_shared = 0;
    float delta_eventos;
    cudaEvent_t start, stop;

    //verifica os parametros de entrada
    if(argc < 6) {
        printf("Digite: %s <tamanho das matrizes> <matriz A> <matriz B> <matriz C> <tamanho do bloco>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    n = atol(argv[1]);
    const char *strFMatA = argv[2], *strFMatB = argv[3], *strFMatC_seq = concat(argv[4], "_seq.txt"),  *strFMatC_gpu = concat(argv[4], "_gpu.txt");
    sizeBlock = atol(argv[5]);

    fMatA = fopen(strFMatA, "r");  fMatB = fopen(strFMatB, "r");  fMatC_seq = fopen(strFMatC_seq, "w"); fMatC_gpu = fopen(strFMatC_gpu, "w");

    //calcula o tamanho em bytes do vetor
    n_bytes = n * n * sizeof(float);

    //aloca memoria na CPU (host)
    h_matrixA = (float*) malloc (n_bytes);
    h_matrixB = (float*) malloc (n_bytes);
    h_matrixCseq = (float*) malloc (n_bytes);
    h_matrixCgpu = (float*) malloc (n_bytes);
    if ((h_matrixA == NULL) || (h_matrixB == NULL) || ((h_matrixCseq) == NULL) || ((h_matrixCgpu) == NULL)) {
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
        multMatrixSeq(h_matrixA, h_matrixB, h_matrixCseq, n);
        printMatrix(stdout, h_matrixCseq, n, "c_seq");
        fprintf(stdout, "%f\n", h_matrixCseq[0]);
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
    dim3 threadsBloco(sizeBlock, sizeBlock);
    dim3 blocosGrade(n/threadsBloco.x, n/threadsBloco.y);

    //memória compartilhada: espaço para duas matrizes float de tamanho sizeBlock x sizeBlock
    //A[sizeBlock][sizeBlock] e B[sizeBlock][sizeBlock]
    sharedBytes = sizeBlock * sizeBlock * sizeof(float) * 2;

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));

    multMatrixGPUSharedDyn <<<blocosGrade, threadsBloco, sharedBytes>>> (d_matrixA, d_matrixB, d_matrixC, n);

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaGetLastError());

    CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

    //copia resultado da GPU para a CPU (device para host)
    CUDA_SAFE_CALL(cudaMemcpy(h_matrixCgpu, d_matrixC, n_bytes, cudaMemcpyDeviceToHost))
    printMatrix(fMatC_gpu, h_matrixCgpu, n, "c_gpu");
    GET_TIME(fim);

    tempo_gpu_dyn_shared = fim - inicio;

    //libera a memoria na GPU
    CUDA_SAFE_CALL(cudaFree(d_matrixA));
    CUDA_SAFE_CALL(cudaFree(d_matrixB));
    CUDA_SAFE_CALL(cudaFree(d_matrixC));

    //libera a memoria na CPU
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixCseq);
    free(h_matrixCgpu);

    //------------------------------- imprime dos tempos de execucao ----------------------//
    if (MAKESEQ)
        fprintf(stdout, "Tempo sequencial = %.10f seg \n", tempo_seq);
    fprintf(stdout, "Tempo paralelo com memória compartilhada dinâmica = %.10f seg \n", tempo_gpu_dyn_shared + delta_eventos/1000);

    if (MAKESEQ) {
        fprintf(stdout, "Speedup tempo paralelo com memória compartilhada dinâmica: %.10f\n", tempo_seq / (tempo_gpu_dyn_shared + delta_eventos/1000));
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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
