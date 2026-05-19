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

    double xmin = -2.0;
    double xmax =  1.0;
    double ymin = -1.5;
    double ymax =  1.5;

    int max_iter = 5000;
    int nb_threads = 12;

    omp_set_num_threads(nb_threads);

    CImg<unsigned char> img(width, height, 1, 3, 0);

    double dx = (xmax - xmin) / width;
    double dy = (ymax - ymin) / height;

    double t0 = omp_get_wtime();

#pragma omp parallel
{
    #pragma omp master
    std::cout << "Threads actifs : "
              << omp_get_num_threads() << std::endl;
}

#pragma omp parallel for schedule(dynamic)
    //#pragma omp parallel
	//{
		//#pragma omp master
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double cr = xmin + x * dx;
				double ci = ymin + y * dy;

				std::complex<double> c(cr, ci);
				std::complex<double> z(0.0, 0.0);

				int iter = 0;

				while (std::abs(z) < 2.0 && iter < max_iter)
				{
					z = z * z + c;
					iter++;
				}

				unsigned char color =
					static_cast<unsigned char>(255.0 * iter / max_iter);

				img(x, y, 0) = color;        // Rouge
				img(x, y, 1) = 0;            // Vert
				img(x, y, 2) = 255 - color;  // Bleu
			}
		}
		//std::cout << "Nb threads : " << omp_get_num_threads() << std::endl;
	//}

    double t1 = omp_get_wtime();

    std::cout << "Temps d'execution : " << t1 - t0 << " s" << std::endl;

    img.display("Ensemble de Mandelbrot");

    return 0;
}

