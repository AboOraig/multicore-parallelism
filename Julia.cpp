#include "CImg.h"
#include <iostream>
#include <complex>
#include <omp.h>

using namespace cimg_library;

int main(int argc, char* argv[])
{
    cimg::openmp_mode(2);

    int width  = 800;
    int height = 600;

    double xmin = -1.5;
    double xmax =  1.5;
    double ymin = -1.5;
    double ymax =  1.5;

    int max_iter = 5000;
    int nb_threads = 4;

    // Constante complexe de Julia (modifiable)
    std::complex<double> c(-0.7, 0.27015);
    //(-0.7, 0.27015)
    //(-0.8, 0.156)
    //(0.285, 0.01)
    //(-0.4, 0.6)

    omp_set_num_threads(nb_threads);

    CImg<unsigned char> img(width, height, 1, 3, 0);

    double dx = (xmax - xmin) / width;
    double dy = (ymax - ymin) / height;

    double t0 = omp_get_wtime();

//#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double zr = xmin + x * dx;
            double zi = ymin + y * dy;

            std::complex<double> z(zr, zi);
            int iter = 0;

            while (std::abs(z) < 2.0 && iter < max_iter)
            {
                z = z * z + c;
                iter++;
            }

			unsigned char color = 255 - (255 * iter / max_iter);

			img(x,y,0) = color;
			img(x,y,1) = color;
			img(x,y,2) = color;
        }
    }

    double t1 = omp_get_wtime();

    std::cout << "Temps d'execution : " << t1 - t0 << " s" << std::endl;

    img.display("Ensemble de Julia");

    return 0;
}
