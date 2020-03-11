#include <smmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


/*  compile with: gcc -march=native -mtune=native -mno-avx -masm=intel -fno-PIE */


/*  Function:       hsum_ps_sse3 ----------------------------------------------
    Description:    Performs a fast horizontal add of all four floats in an xmm
                    register. Used in Vector matrix multiplication algorithm.
    Inputs:         __m128 xmm register
    Outputs:        sum of the four floats
-------------------------------------------------------------------------------*/
float hsum_ps_sse3(register __m128 v) {

    register float sum;

    asm volatile
    (
        "movshdup   xmm1, xmm0          # broadcast elements 3,1 to 2,0 \n\t"
        "addps      xmm0, xmm1          # add                           \n\t"
        "movhlps    xmm1, xmm0          # high half -> low half         \n\t"
        "addss      xmm0, xmm1          # add                           \n\t"

        :[sum] "=x" (sum)
        :
        :
    );
    return sum;
}



/*  Function:       random_init_m128_ps----------------------------------------
    Description:    Initializes xmm register with four random floats in the
                    range of [0...1]
    Inputs:         n/a
    Outputs:        usage example: __m128 xmm = { w, x, y, z } where w,x,y,z
                    are random floats in the range or [0...1]
-------------------------------------------------------------------------------*/
__m128 random_init_m128_ps(void) {

    #define IRM 1.0f/RAND_MAX
    static const float max[] = {IRM,IRM,IRM,IRM};
    register __m128 result;

    asm volatile
    (
        "push       rax                 # rax will be overwritten by rand()     \n\t"
        "push       rbx                 # rbx will be overwritten by rand()     \n\t"
        "push       rcx                 # rcx will be overwritten by rand()     \n\t"
        "push       rdx                 # rdx will be overwritten by rand()     \n\t"
        "push       rsi                 # rsi will be overwritten by rand()     \n\t"
        "push       rdi                 # rdi will be overwritten by rand()     \n\t"
        "call       rand                #                                       \n\t"
        "movd       xmm0, eax           # insert random number in xmm0 float 0  \n\t"
        "call       rand                #                                       \n\t"
        "pinsrd     xmm0, eax, 1        # insert random number in xmm0 float 1  \n\t"
        "call       rand                #                                       \n\t"
        "pinsrd     xmm0, eax, 2        # insert random number in xmm0 float 2  \n\t"
        "call       rand                #                                       \n\t"
        "pinsrd     xmm0, eax, 3        # insert random number in xmm0 float 3  \n\t"
        "cvtdq2ps   xmm0, xmm0          # convert random ints in xmm0 to floats \n\t"
        "mulps      xmm0, %[max]        # scale: mult by 1 / RAND_MAX           \n\t"
        "pop       rdi                  # rdi will be restored                  \n\t"
        "pop       rsi                  # rsi will be restored                  \n\t"
        "pop       rdx                  # rdx will be restored                  \n\t"
        "pop       rcx                  # rcx will be restored                  \n\t"
        "pop       rbx                  # rbx will be restored                  \n\t"
        "pop       rax                  # rax will be restored                  \n\t"
        :[result] "=x" (result)
        :[max] "m" (max)
        :"eax","edx","memory"
    );
    return result;
}



/*  Function:       mul_acc_m128_ps  ------------------------------------------
    Description:    Performs a multiplication of two xmm registers with 4
                    packed floats each and afterwards accumulates the resulting
                    four floats.
    Inputs:         __m128 xmm registers x and y
    Outputs:        sum of the four floats
-------------------------------------------------------------------------------*/
float mul_acc_m128_ps(register __m128 x, register __m128 y) {

    register float mulacc;
    asm volatile
    (
        "mulps      %[x], %[y]          # xmm multiply four floats          \n\t"
        "call       hsum_ps_sse3        # do fast horizontal add            \n\t"

        :[mulacc] "=x" (mulacc)
        :[x] "x" (x), [y] "x" (y)
        :
    );
    return mulacc;
}



typedef struct matrix {
    float * m;
    unsigned long r,c;
} matrix,vector;


/*  Function:       create_matrix ---------------------------------------------
    Description:    Initializes Matrix struct with number of rows and columns
                    given and tries to allocate memory for matrix elements.
                    Matrix elements are initialized to zero (calloc).
    Inputs:         matrix struct, number of rows r, number of columns c
    Outputs:        n/a
-------------------------------------------------------------------------------*/
void create_matrix(matrix * A, unsigned long r, unsigned long c) {

    A->r = r;
    A->c = c;
    A->m = calloc(r*c,sizeof(float));
    assert (A->m != NULL);           // fail if memory could not be allocated
}


/*  Function:       randomize_matrix ------------------------------------------
    Description:    Initializes Matrix with random numbers (floats)
                    Uses random_init_m128_ps to init four floats at a time
    Inputs:         matrix A
    Outputs:        n/a
-------------------------------------------------------------------------------*/
void randomize_matrix(matrix * A) {

    unsigned long   num_steps           =   (A->r * A->c) >> 2;
    __m128          *p_matrix_elements  =   (__m128 *)A->m;

    while (num_steps--) {
        *p_matrix_elements++ = random_init_m128_ps();
    }
}



/*  Function:       mul_vec_mat -----------------------------------------------
    Description:    Performs vector matrix multiplication Y = X*A using SIMD
                    mul_acc_m128_ps is used to multiply accumulate vectors of
                    four floats at once. mul_vec_mat iterates over the input
                    vector and the matrix in steps of four floats. Four floats
                    in the input vector are multiplied at once with four
                    floats in the matrix and are accumulated afterwards. The
                    result is stored in the output vector Y. Vector length as
                    well as matrix row and column size must be a factor of
                    four (not checked yet - fixme!).

    Inputs:         vector Y, vector X, matrix A
    Outputs:        n/a
-------------------------------------------------------------------------------*/
void mul_vec_mat(vector *Y, vector *X, matrix *A) {

    unsigned long    i                      =   0;                  // output vector index
    unsigned long    num_steps_total        =   (A->r * A->c) >> 2; // steps of four needed to iterate through matrix
    unsigned long    num_steps_vector       =   X->r >> 2;          // steps of four needed to iterate through input vector
    unsigned long    step_index_vector      =   num_steps_vector;   // current step index
    __m128          *p_matrix_elements      =   (__m128 *)A->m;
    __m128          *p_vector_x_elements    =   (__m128 *)X->m;

     while (num_steps_total--) {
        Y->m[i] += mul_acc_m128_ps(*p_vector_x_elements++,*p_matrix_elements++);
        if (--step_index_vector == 0) { // end of input vector and matrix row
            step_index_vector   =  num_steps_vector; // reset step index for input vector to start
            p_vector_x_elements = (__m128 *)X->m; // reset pointer to input vector element to start
            i++; // increment output vector index
        }
    }
}



int main(void)
{
    matrix  mat_A;
    matrix  *A  =   &mat_A;
    vector  vec_X;
    vector  *X  =   &vec_X;
    vector  vec_Y;
    vector  *Y  =   &vec_Y;

    srand (time(0));

    create_matrix(A,11500,11500);
    randomize_matrix(A);

    create_matrix(X,11500,1);
    randomize_matrix(X);

    create_matrix(Y,11500,1);

    mul_vec_mat(Y,X,A);

    for (int i=0; i<vec_X.r; i++)
        printf("%f\n",(float)vec_X.m[i]);

    for (int j=0; j<A->r;j++) {
        for (int k=0; k<A->c; k++) {
            printf("%f ",A->m[k+j*A->c]);
        }
        printf("\n");
    }
    for (int i=0; i<Y->r;i++)
        printf("%f\n",Y->m[i]);

    return 0;
}
