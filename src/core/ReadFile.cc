#include "ReadFile.h"
#include <cstdlib>
#include <cstdio>
#include "debug.h"
#include <cerrno>

int FileToIntVector(std::vector<int>& data, char* fname, int tmpT)
{
    FILE* file = fopen(fname, "r");
    if (!file) {
        fprintf (stderr, "Error: Could not open file %s\n", fname);
        exit(-1);
    }

	int T = 0;
    int success = fscanf(file, "%d", &T);
	if (success <= 0) {
            Serror("Could not scan file %s - T =%d - retval: %d - errno %d\n", fname, T, success, errno);
	}

	if (tmpT > 0 && tmpT < T) {
		T = tmpT;
	}

    printf("horizon: %d\n", T);
    data.resize(T);
    int n_observations = 0;
    for (int t=0; t<T; ++t) {
		int success = fscanf(file, "%d", &data[t]);
		if (success <=0) {
			Serror("Could not scan file\n");
		}
		if (data[t] > n_observations) {
            n_observations = data[t];
        }
        data[t] -= 1;
    }
    fclose(file);
	return n_observations;
}

int ReadClassData(Matrix& data, std::vector<int>& labels, char* fname) 
{
    FILE* file = fopen(fname, "r");
    if (!file) {
        fprintf (stderr, "Error: Could not open file %s\n", fname);
        exit(-1);
    }

	int T = 0;
    int columns;
    int success = fscanf(file, "%d %d", &T, &columns);
	if (success <= 0) {
            Serror("Could not scan file %s - T =%d - retval: %d - errno %d\n", fname, T, success, errno);
	}
    
    printf("horizon: %d, columns: %d\n", T, columns);
    data.Resize(T, columns - 1);
    labels.resize(T);
    int n_observations = 0;
    for (int t=0; t<T; ++t) {
        for (int i=0; i<columns - 1; ++i) {
            int success = fscanf(file, "%lf", &data(t,i));
            if (success <=0) {
                Serror("Could not scan file, line %d, column %d\n", t, i);
            }
        }
        int success = fscanf(file, "%d", &labels[t]);
        if (success <=0) {
            Serror("Could not scan file, line %d\n", t);
        }
    }
    fclose(file);
	return n_observations;
}
