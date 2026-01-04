#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Sigmoid activation */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* XOR Model using OR, NAND, AND neurons */
typedef struct{
    float or_w1;
    float or_w2;
    float or_b;

    float and_w1;
    float and_w2;
    float and_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
}Xor;

/* Forward pass */
float forward(Xor m, float x1, float x2){
    float a = sigmoid(x1 * m.or_w1 + x2 * m.or_w2 + m.or_b);
    float b = sigmoid(x1 * m.nand_w1 + x2 * m.nand_w2 + m.nand_b);
    return sigmoid(a * m.and_w1 + b * m.and_w2 + m.and_b);
    
}

/* Training data */
typedef float sample[3];

sample train_xor[]={
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample *train = train_xor;

size_t train_count = 4;

/* Mean Squared Error */
float cost(Xor m) {
    float result = 0.0f;
    for (int  i=0; i<train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2); 
        float d = y - train[i][2];
        result += d*d;
    }  
    result /= train_count;
    return result ;
}

/* Random float [0,1] */
float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;}

/* Random model initialization */
Xor rand_xor(void){
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();
    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();
    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    return m;
}

void print_xor(Xor m){
    printf("OR: w1=%f, w2=%f, b=%f\n", m.or_w1, m.or_w2, m.or_b);
    printf("AND: w1=%f, w2=%f, b=%f\n", m.and_w1, m.and_w2, m.and_b);
    printf("NAND: w1=%f, w2=%f, b=%f\n", m.nand_w1, m.nand_w2, m.nand_b);
}

/* Finite difference gradient */
Xor finite_diff(Xor m,float eps){
    Xor g;
    float c = cost(m);
    float saved;
    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c)/eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    m.or_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = saved;
    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float learning_rate){
    m.or_w1 -= g.or_w1 * learning_rate;
    m.or_w2 -= g.or_w2 * learning_rate;
    m.or_b -= g.or_b * learning_rate;

    m.and_w1 -= g.and_w1 * learning_rate;
    m.and_w2 -= g.and_w2 * learning_rate;
    m.and_b -= g.and_b * learning_rate;

    m.nand_w1 -= g.nand_w1 * learning_rate;
    m.nand_w2 -= g.nand_w2 * learning_rate;
    m.nand_b -= g.nand_b * learning_rate;

    return m;
}

int main(){
    srand(time(NULL));
    Xor m = rand_xor();
    float eps = 1e-3f;
    float learning_rate = 1e-1f;
    for (size_t i = 0 ; i < 1000*1000  ; ++i) {
        Xor g = finite_diff(m, eps);
        m = learn(m, g, learning_rate);
        printf("Trian No.%zu COST=%f\n", i+1, cost(m));
    }
    printf("-----------------\n");    
    for (int i=0; i<train_count; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2); 
        printf("XOR(%f, %f) = %f\n", x1, x2, y);
    }

    return 0;
}
