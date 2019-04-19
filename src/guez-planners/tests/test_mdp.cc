#include "DiscreteMDPCounts.h"
#include "MDPModel.h"
#include "real.h"

int main(int argc, char** argv) {

	real dirichlet_mass = 2.0;
	auto reward_prior = DiscreteMDPCounts::BETA;
	MDPModel* belief = new DiscreteMDPCounts(500, 6, dirichlet_mass, reward_prior);
	for (int i=0;i<20;i++){
		belief->AddTransition(0,0,2.7,4);
		belief->AddTransition(4,1,2.5,0);
	}
	belief->generate();

/*
	int i,j;
	int r=5;
	int c=2;
	real **arr = (real **)malloc(r * sizeof(real *)); 
    for (i=0; i<r; i++) 
         arr[i] = (real *)malloc(c * sizeof(real));

real count = 0;
for (i = 0; i <  r; i++) 
      for (j = 0; j < c; j++) 
         arr[i][j] = ++count;  // OR *(*(arr+i)+j) = ++count 
  
    for (i = 0; i <  r; i++){
	printf("\nnew row "); 
      for (j = 0; j < c; j++) 
         printf("%f ", arr[i][j]);
	}

//Freeing up
for (i=0; i<r; ++i) {
  free(arr[i]);
}
free(arr);

*/

/*
int r = 3, c = 4; 
    int *arr = (int *)malloc(r * c * sizeof(int)); 
  
    int i, j, count = 0; 
    for (i = 0; i <  r; i++) 
      for (j = 0; j < c; j++) 
         arr[i*c + j] = ++count; 
  
    for (i = 0; i <  r; i++) 
      for (j = 0; j < c; j++) 
         printf("%d ", *(arr + i*c + j)); 
*/

return 0;
}
