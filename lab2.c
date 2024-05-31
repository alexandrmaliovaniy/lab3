#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Define the grid size and the radius
#define N 100  // Total number of neurons (cube root of N should be an integer)
#define R 1.5  // Connectivity radius

int main() {
    int size = cbrt(N);
    if (size * size * size != N) {
        printf("N must be a perfect cube!\n");
        return 1;
    }

    // Allocate memory for the connectivity matrix
    int **A = malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = calloc(N, sizeof(int));
    }

    // Calculate coordinates for each neuron
    int coordinates[N][3];
    for (int i = 0; i < N; i++) {
        coordinates[i][0] = i % size; // x-coordinate
        coordinates[i][1] = (i / size) % size; // y-coordinate
        coordinates[i][2] = i / (size * size); // z-coordinate
    }

    // Calculate the connectivity matrix
    double start_time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                int dx = coordinates[i][0] - coordinates[j][0];
                int dy = coordinates[i][1] - coordinates[j][1];
                int dz = coordinates[i][2] - coordinates[j][2];
                // Ensure periodic boundary conditions
                dx = dx - size * ((dx > size/2) - (dx < -size/2));
                dy = dy - size * ((dy > size/2) - (dy < -size/2));
                dz = dz - size * ((dz > size/2) - (dz < -size/2));
                float distance = sqrt(dx*dx + dy*dy + dz*dz);
                if (distance <= R) {
                    A[i][j] = 1;
                }
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("Connectivity calculation took %f seconds.\n", end_time - start_time);

    // Free the allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);

    return 0;
}
