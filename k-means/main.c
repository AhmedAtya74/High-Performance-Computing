#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_ITERATIONS 100

#define THRESHOLD 1e-3

#define N 8


float *centroids_global;
float *cluster_points_global;
float delta_global = THRESHOLD + 1;
double delta_global_forEach_iteration;


void readDataset(const char *filename, float **data_points)
{
    FILE *file = fopen(filename, "r");

    printf("Number of data points in the dataset: %d\n", N);

    *data_points = (float *)malloc((N * 2) * sizeof(float));

    for (int i = 0; i < N * 2; i++)
    {
        int tmp;
        fscanf(file, "%d", &tmp);
        *(*data_points + i) = tmp;
    }

    fclose(file);
}


double CalculateEuclideanDistance(float *A, float *B)
{
    return sqrt(pow(((double)(*(B + 0)) - (double)(*(A + 0))), 2) + pow(((double)(*(B + 1)) - (double)(*(A + 1))), 2));
}


void Clustering(int tID, int K, int num_threads, float *data_points, float **cluster_points, int *iterations_num)
{
    // printf("Inside Clustering %d\n", tID);
    int chunk = ceil(N * 1.0 / num_threads);
    int st = tID * chunk;
    int ed = st + chunk;

    if (ed > N)
    {
        ed = N;
        chunk = ed - st;
    }
    // printf("start = %d \n", st);
    // printf("end = %d \n", ed);
    // printf("Chunk = %d \n", chunk);
    double min_distance, current_distance;
    int *current_cluster_id = (int *)malloc(sizeof(int) * chunk);

    int iteration_count = 0;

    while ((delta_global > THRESHOLD) && (iteration_count < MAX_ITERATIONS))
    {
        float *current_centroid = (float *)calloc(K * 2, sizeof(float));
        int *cluster_count = (int *)calloc(K, sizeof(int));
        // iterate over thread chunk
        for (int i = st; i < ed; i++)
        {

            min_distance = INT_MAX;
            // iterate over number of threads
            // calculate distance between each point and centriods
            for (int j = 0; j < K; j++)
            {
                current_distance = CalculateEuclideanDistance((data_points + (i * 2)), (centroids_global + (iteration_count * K + j) * 2));
                if (current_distance < min_distance)
                {
                    min_distance = current_distance;
                    current_cluster_id[i - st] = j; // set cluster id
                }
            }

            cluster_count[current_cluster_id[i - st]]++;

            current_centroid[current_cluster_id[i - st] * 2] += data_points[(i * 2)];
            current_centroid[current_cluster_id[i - st] * 2 + 1] += data_points[(i * 2) + 1];
        }

        #pragma omp critical
        {
            for (int i = 0; i < K; i++)
            {
                if (cluster_count[i] == 0){ continue;}

                // Update the centroids
                centroids_global[((iteration_count + 1) * K + i) * 2] = current_centroid[(i * 2)] / (float)cluster_count[i];
                centroids_global[((iteration_count + 1) * K + i) * 2 + 1] = current_centroid[(i * 2) + 1] / (float)cluster_count[i];
            }
        }

        // Find delta value after each iteration in all the threads
        double current_delta = 0.0;
        for (int i = 0; i < K; i++)
        {
            current_delta += CalculateEuclideanDistance((centroids_global + (iteration_count * K + i) * 2), (centroids_global + ((iteration_count - 1) * K + i) * 2));
        }

        // Store the largest delta value among all delta values in all the threads
        #pragma omp barrier
        {
            if (current_delta > delta_global_forEach_iteration)
                delta_global_forEach_iteration = current_delta;
        }

        #pragma omp barrier
        {
            iteration_count++;
        }
        // Set the global delta value and increment the number of iterations
        #pragma omp master
        {
            delta_global = delta_global_forEach_iteration;
            delta_global_forEach_iteration = 0.0;
            (*iterations_num)++;
        }
    }

    // Update the cluster_points
    for (int i = st; i < ed; i++)
    {
        cluster_points_global[i * 3] = data_points[i * 2];
        cluster_points_global[i * 3 + 1] = data_points[i * 2 + 1];
        cluster_points_global[i * 3 + 2] = (float)current_cluster_id[i - st];
    }
}


void kmeansClustering(int K, int num_threads, float *data_points, float **cluster_points, int *iterations_num)
{

    *cluster_points = (float *)malloc(sizeof(float) * N * 3);
    cluster_points_global = *cluster_points;

    // calloc intitalizes the values to zero
    centroids_global = (float *)calloc(MAX_ITERATIONS * K * 2, sizeof(float));

    // Assigning the first K data points to be the centroids of the K clusters
    for (int i = 0; i < K; i++)
    {
        centroids_global[(i * 2)] = data_points[(i * 2)];
        centroids_global[(i * 2) + 1] = data_points[(i * 2) + 1];
    }

    omp_set_num_threads(num_threads);

    int tID;
    #pragma omp parallel private(tID)
    {
        tID = omp_get_thread_num();
        Clustering(tID, K, num_threads, data_points, cluster_points, iterations_num);
        printf("Thread Number %d created\n", tID);
    }

    printf("Number of Iterations = %d \n", *iterations_num);
    printf("Process Completed\n");
}

int main(int argc, char const *argv[])
{
    /*
        gcc -o out.o -fopenmp main.c
        out.o datapoints.txt 3 2
    */

	if (argc < 4)
	{
		printf("Less Number of Command Line Arguments\n\n");
		return 0;
	}
	else if (argc > 4)
	{
		printf("Too Many Commands Line Arguments\n\n");
		return 0;
	}

	const char *dataset = argv[1];
	const int K = atoi(argv[2]);
	const int num_threads = atoi(argv[3]);

	float *data_points;
	float *cluster_points;
	int iterations_num = 0;

	readDataset(dataset, &data_points);

	kmeansClustering(K, num_threads, data_points, &cluster_points, &iterations_num);

    for(int j = 0; j < K; j++)
    {
        printf("cluster %d:\n", j);
        for(int i = 0; i < N; i++)
        {
            if(*(cluster_points + (i * 3) + 2) == j){
            printf("(%f, %f)\n", *(cluster_points + (i * 3)), *(cluster_points + (i * 3) + 1));
            }
        }
    }

	return 0;
}
