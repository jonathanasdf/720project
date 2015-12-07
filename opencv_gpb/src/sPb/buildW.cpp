#include "buildW.h"

using namespace std;

namespace cv
{
void buildW(const SMatrix* W, double** &T, int & wz, double* &D)
{
    //output assignment
    wz = 0;
    for(size_t i=0; i<W->n; i++)
        wz += W->nz[i];

    T = new double*[wz];
    D = new double[W->n];
    int ct = 0;
    for(size_t row = 0; row < W->n; row++) {
        //initialize diag matrix.
        //int diag_ind = 0;
        D[row] = 0;
        for(size_t i=0; i<W->nz[row]; i++) {
            //initialize sparse matrix.
            //if(row == W->col[row][i])
            //diag_ind = i;
            T[ct+i] = new double[3];
            //fill sparse matrix.
            T[ct+i][0] = static_cast<double>(row);
            T[ct+i][1] = static_cast<double>(W->col[row][i]);
            T[ct+i][2] = static_cast<double>(W->values[row][i]);
            //calculate sparse diagnoal matrix
            D[row] += static_cast<double>(W->values[row][i]);
        }
        //T[ct+diag_ind][2] += D[row];
        D[row] = sqrt(D[row]);
        ct +=  W->nz[row];
    }
}
}
