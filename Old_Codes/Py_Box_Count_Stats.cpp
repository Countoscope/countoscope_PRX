#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h> //for pseudorandom numbers
#include <time.h>
#include <fstream>
#include <math.h>
#include <complex>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <fftw3.h>
#include <boost/math/distributions/chi_squared.hpp>

using Eigen::ArrayXd;
using Eigen::VectorXi;

typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;

namespace py = pybind11;

template <typename VectorType>
Vector autocorrFFT(const VectorType& x) {
    int N = x.size();
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2 * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2 * N);
    fftw_plan p;

    for (int i = 0; i < N; i++) {
        in[i][0] = x(i);
        in[i][1] = 0;
    }
    for (int i = N; i < 2 * N; i++) {
        in[i][0] = 0;
        in[i][1] = 0;
    }

    p = fftw_plan_dft_1d(2 * N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i = 0; i < 2 * N; i++) {
        double real = out[i][0];
        double imag = out[i][1];
        out[i][0] = real * real + imag * imag;
        out[i][1] = 0;
    }

    p = fftw_plan_dft_1d(2 * N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    Vector res(N);
    for (int i = 0; i < N; i++) {
        res(i) = in[i][0] / (2.0*N);
        res(i) /= N - i;
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return res;
}

template <typename MatrixType>
Vector msd_fft(MatrixType& r) {
    int N = r.rows();
    int dim = r.cols();

    ArrayXd D = r.array().square().rowwise().sum();
    D.conservativeResize(D.size() + 1);
    D(D.size() - 1) = 0;

    Vector S2 = Vector::Zero(N);
    for (int i = 0; i < dim; ++i) {
        S2 += autocorrFFT(r.col(i));
    }
    double Q = 2.0 * D.sum();
    Vector S1 = Vector::Zero(N);
    int Ind = 0;;
    for (int m = 0; m < N; ++m) {
        Ind = m-1;
        if(m == 0){Ind = N;}
        Q = Q - D[Ind] - D[N-m];
        S1[m] = Q / (N - m);
    }
    return S1 - 2 * S2;
}


template <typename VectorType>
Vector Box_Bin_Exp(VectorType& x,
                 VectorType& y,
                 double Lx,
                 double Ly,
                 double Lbox,
                 double sep) {
    int Np = x.size();
    double SepSize = (Lbox + sep);
    int Nx = std::floor(Lx / SepSize);
    int Ny = std::floor(Ly / SepSize);
    Vector Counts = Vector::Zero(Nx * Ny);

    for (int i = 0; i < Np; ++i) {
        // periodic corrections
        while(x[i] > Lx){x[i] = x[i] - Lx;}
        while(x[i] < 0.0){x[i] = x[i] + Lx;}
        while(y[i] > Ly){y[i] = y[i] - Ly;}
        while(y[i] < 0.0){y[i] = y[i] + Ly;}

        // find correct box and increment counts
        int II = std::floor(x[i] / SepSize);
        int JJ = std::floor(y[i] / SepSize);
        double Xmod = std::fmod(x[i], SepSize);
        double Ymod = std::fmod(y[i], SepSize);

        if((II+1.0)*SepSize > Lx){continue;}
        if((JJ+1.0)*SepSize > Ly){continue;}

        // ((Xmod > 0.5*sep) && (Ymod > 0.5*sep) && (Xmod < (SepSize-0.5*sep)) && (Ymod < (SepSize-0.5*sep)))
        if ( std::max(std::abs(Xmod-0.5*SepSize), std::abs(Ymod-0.5*SepSize)) < Lbox/2.0 ) {
            Counts[II * Ny + JJ] += 1.0;
        }
    }

    return Counts;
}

void outputMatrixToFile(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        file << std::setprecision(10) << matrix << std::endl;
        file.close();
        std::cout << "Matrix data has been written to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open the file " << filename << std::endl;
    }
}


void computeColumnStats(const Matrix& matrix, ArrayXd& mean, ArrayXd& SEM) {
    int numRows = matrix.rows();
    int numCols = matrix.cols();
    ArrayXd sumSqDiff = ArrayXd::Zero(numCols);
    ArrayXd row_i;
    ArrayXd delta, delta2;
    ArrayXd temp;

    for (int i = 0; i < numRows; ++i) {
        row_i = matrix.row(i);
        delta = (row_i - mean);
        mean += (1.0/(i+1.0))*delta;
        delta2 = (row_i - mean);
        sumSqDiff += delta.cwiseProduct(delta2);
    }
    SEM = (2.0/std::sqrt(numRows*(numRows-1.0)))*sumSqDiff.sqrt();
}





void ConvertDataFile(const char* filename) {
    FILE* fileinput = fopen(filename, "r");
    if (!fileinput) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int np;
    int Ntimes = 0;
    Vector x, y, z;
    double aux1, aux2, aux3, aux4;

    // Parse 'filename' to remove the extension (chars after a period) and add the string "_modified.txt" to the result.
    std::string inputFilename(filename);
    std::size_t pos = inputFilename.find_last_of('.');
    std::string baseFilename = inputFilename.substr(0, pos);
    std::string outfile = baseFilename + "_modified.txt";

    FILE* fileoutput = fopen(outfile.c_str(), "w");
    if (!fileoutput) {
        std::cerr << "Error opening output file: " << outfile << std::endl;
        fclose(fileinput);
        return;
    }

    int ferr;
    
    while (fscanf(fileinput, "%d", &np) != EOF) {
        Ntimes++;
        std::cout << Ntimes << "\n";

        x.resize(np);
        y.resize(np);
        z.resize(np);

        // Read the data directly into the vectors and write to the output file
        for (int i = 0; i < np; i++) {
            ferr = fscanf(fileinput, "%lf %lf %lf %lf %lf %lf %lf", &x(i), &y(i), &z(i), &aux1, &aux2, &aux3, &aux4);
            fprintf(fileoutput, "%.6f %.6f %d\n", x(i), y(i), Ntimes);
        }
    }

    fclose(fileinput);
    fclose(fileoutput);
}

std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>> processDataFile(const char* filename, int Nframes) {
    std::vector<std::vector<double>> Xs(Nframes);
    std::vector<std::vector<double>> Ys(Nframes);

    FILE* fileinput = fopen(filename, "r");
    if (!fileinput) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return std::make_tuple(Xs,Ys);
    }

    int ind, ind_p;
    double x, y;

    int frame = 0;
    int start = 0;
    while (fscanf(fileinput, "%lf %lf %d", &x, &y, &ind) != EOF) {
        if((frame == 0) & (ind != 0)){
             start = ind;
             frame = 1;
             ind_p = ind-1;
        }
        if(ind_p != ind){std::cout << ind << "\n";}

        Xs[ind-start].push_back(x);
        Ys[ind-start].push_back(y);
        ind_p = ind;
    }
    fclose(fileinput);

    return std::make_tuple(Xs,Ys);
}

std::vector<Matrix> processDataFile_and_Count(const char* filename, int Nframes, double Lx, double Ly, std::vector<double>& Lbox, std::vector<double>& sep) {


    int remo = 1;
    std::vector<std::vector<double>> x, y;
    Vector Count_nt;
    std::tie(x,y) = processDataFile(filename, Nframes);
    std::cout << "Done with data read \n";

    std::vector<std::vector<Vector>> Counts(Lbox.size());


    for (int lbIdx = 0; lbIdx < Lbox.size(); ++lbIdx) {
        std::cout << "Counting boxes L = " << Lbox[lbIdx] << "\n";
        //int nbins = std::round(Lp / Lbox[lbIdx]) - remo;
        //sep = static_cast<double>(remo) * Lbox[lbIdx] / static_cast<double>(nbins);
        for (int nt = 0; nt < x.size(); ++nt){
            Count_nt = Box_Bin_Exp(x[nt], y[nt], Lx, Ly, Lbox[lbIdx], sep[lbIdx]);
            Counts[lbIdx].push_back(Count_nt);
        }
    }


    // convert data to std::vector of Matrices
    std::vector<Matrix> CountMs(Lbox.size());
    for (int lbIdx = 0; lbIdx < Lbox.size(); ++lbIdx) {
        int numCounts = Counts[lbIdx].size(); // number of time steps
        int countSize = Counts[lbIdx][0].size(); // number of boxes

        CountMs[lbIdx].resize(countSize, numCounts);

        for (int i = 0; i < numCounts; ++i) {
            CountMs[lbIdx].col(i) = Counts[lbIdx][i];
        }
    }
    std::cout << "Done with counting \n";
    return CountMs;
}

void computeMeanAndSecondMoment(const Matrix& matrix, double& mean, double& variance, double& variance_sem_lb, double& variance_sem_ub) {
    int numRows = matrix.rows();
    int numCols = matrix.cols();
    int n = numRows * numCols;

    mean = 0.0;
    double m2 = 0.0;

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            double value = matrix(i, j);
            double delta = value - mean;
            mean += delta / (i * numCols + j + 1.0);
            m2 += delta * (value - mean);
        }
    }
    variance = m2 / n;
    
    double alpha = 0.01;
    double df = 1.0*matrix.size() - 1.0;
    double chi_lb = boost::math::quantile(boost::math::chi_squared_distribution<double>(df), 0.5*alpha);
    double chi_ub = boost::math::quantile(complement(boost::math::chi_squared_distribution<double>(df), 0.5*alpha));
    
    //double test1 = boost::math::quantile(boost::math::chi_squared_distribution<double>(100.0), 0.5*alpha);
    //double test2 = boost::math::quantile(complement(boost::math::chi_squared_distribution<double>(100.0), 0.5*alpha));
    //std::cout << "chi test: " << test1 << "  " << test2 << "\n";
    
    variance_sem_lb = (df/chi_lb)*variance;
    variance_sem_ub = (df/chi_ub)*variance;
    
//////////////////////////////////////////////////////////    
//     ArrayXd row_var = ArrayXd::Zero(numRows);
//     for (int i = 0; i < numRows; ++i) {
//         double row_mean = matrix.row(i).mean();
//         for (int j = 0; j < numCols; ++j) {
//             double diff = matrix(i, j) - row_mean;
//             row_var(i) += diff * diff;
//         }
//         row_var(i) /= (numCols - 1.0);
//     }
//     variance = row_var.mean();
//     double var_variance = ((row_var-variance).square().sum()) / (numRows - 1.0);
//     variance_sem = 2.0*std::sqrt(var_variance / (1.0*numRows));
////////////////////////////////////////////////////////////    
    
}

void Calc_and_Output_Stats(const char* infile, std::string outfile, int Nframes, double Lx, double Ly, std::vector<double>& Lbs, std::vector<double>& sep) {
    std::vector<Matrix> CountMs = processDataFile_and_Count(infile, Nframes, Lx, Ly, Lbs, sep);

    Matrix N_Stats(Lbs.size(),5);

    for (int lbIdx = 0; lbIdx < Lbs.size(); ++lbIdx) {
        std::cout << "Processing Box size: " << Lbs[lbIdx] << "\n";

        N_Stats(lbIdx,0) = Lbs[lbIdx];
        computeMeanAndSecondMoment(CountMs[lbIdx], N_Stats(lbIdx,1), N_Stats(lbIdx,2), N_Stats(lbIdx,3), N_Stats(lbIdx,4));

        Vector MSDrow;
        Matrix MSDs = Matrix::Zero(CountMs[lbIdx].rows(),CountMs[lbIdx].cols());
        for (int i = 0; i < MSDs.rows(); ++i) {
            std::cout << 100.0*((1.0*i)/(1.0*MSDs.rows())) << " percent done with MSD calc" << "\n";
            MSDrow = CountMs[lbIdx].row(i);
            MSDs.row(i) = msd_fft(MSDrow);
            //std::cout << "MSD: " << MSDs.row(i) << "\n";
        }
        //outputMatrixToFile(MSDs, "./Count_Data_Cpp/MSDs_BoxL_" + std::to_string(Lbox) + "_phi_0.34.txt");
        ArrayXd MSDmean = ArrayXd::Zero(MSDs.cols());
        ArrayXd MSDsem = ArrayXd::Zero(MSDs.cols());
        computeColumnStats(MSDs, MSDmean, MSDsem);
        outputMatrixToFile(MSDmean, outfile + "_MSDmean_BoxL_" + std::to_string(Lbs[lbIdx]) + ".txt"); //No_Hydro_
        outputMatrixToFile(MSDsem, outfile + "_MSDerror_BoxL_" + std::to_string(Lbs[lbIdx]) + ".txt");
    }
    outputMatrixToFile(N_Stats.array(), outfile+"_N_stats.txt");
}


PYBIND11_MODULE(Py_Box_Count_Stats, m) {
    m.def("autocorrFFT", &autocorrFFT<RefVector& >, "calc autocorr using FFTW");
    m.def("msd_fft", &msd_fft<RefMatrix& >, "calc MSD using FFTW");
    m.def("Box_Bin_Exp", &Box_Bin_Exp<RefVector& >, "Does the actual counting on arrays of x and y coords");
    m.def("ConvertDataFile", &ConvertDataFile, "converts sim data files and appends *modified* to the output filename");
    m.def("processDataFile", &processDataFile, "Process data file");
    m.def("processDataFile_and_Count", &processDataFile_and_Count, "Process data file and count");
    m.def("computeMeanAndSecondMoment", &computeMeanAndSecondMoment, "Compute mean and second moment");
    m.def("Calc_and_Output_Stats", &Calc_and_Output_Stats, "Calculate and Output Stats");
}


