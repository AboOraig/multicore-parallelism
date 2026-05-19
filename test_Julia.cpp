#include "CImg.h"
#include <iostream>
#include <complex>
#include <omp.h>
#include <cstdlib>  

using namespace cimg_library;

static omp_sched_t sched_from_int(int s) {
  switch (s) {
    case 0: return omp_sched_static;
    case 1: return omp_sched_dynamic;
    case 2: return omp_sched_auto;
    default: return omp_sched_dynamic;
  }
}

int main(int argc, char* argv[]) {
  // CImg OpenMP mode (0/1/2)
  cimg::openmp_mode(2);

  /* ================= Paramètres ================= */
  int width  = (argc > 1) ? std::atoi(argv[1]) : 800;
  int height = (argc > 2) ? std::atoi(argv[2]) : 600;

  double xmin = (argc > 3) ? std::atof(argv[3]) : -2.0;
  double xmax = (argc > 4) ? std::atof(argv[4]) :  1.0;
  double ymin = (argc > 5) ? std::atof(argv[5]) : -1.5;
  double ymax = (argc > 6) ? std::atof(argv[6]) :  1.5;

  int max_iter   = (argc > 7) ? std::atoi(argv[7]) : 5000;
  int nb_threads = (argc > 8) ? std::atoi(argv[8]) : 4;

  // Scheduler & chunk optionnels
  int sched_i = (argc > 9)  ? std::atoi(argv[9])  : 1;
  int chunk   = (argc > 10) ? std::atoi(argv[10]) : 1;

  omp_set_num_threads(nb_threads);
  omp_set_schedule(sched_from_int(sched_i), chunk);

  std::cout << "Config:\n";
  std::cout << "  size=" << width << "x" << height << "\n";
  std::cout << "  window=[" << xmin << "," << xmax << "]x[" << ymin << "," << ymax << "]\n";
  std::cout << "  iter=" << max_iter
            << "  threads(requested)=" << nb_threads
            << "  sched=" << sched_i << " chunk=" << chunk << "\n";

  CImg<unsigned char> img(width, height, 1, 3, 0);

  const double dx = (xmax - xmin) / (double)width;
  const double dy = (ymax - ymin) / (double)height;

  double t0 = omp_get_wtime();

#pragma omp parallel
  {
#pragma omp single
	std::cout << "  threads(actual)=" << omp_get_num_threads() << "\n";

const std::complex<double> c(-0.7, 0.27015);

#pragma omp for collapse(2) schedule(runtime)
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const double zr = xmin + x * dx;
        const double zi = ymin + y * dy;

        std::complex<double> z(zr, zi);

        int iter = 0;
        while (std::abs(z) < 2.0 && iter < max_iter) {
          z = z * z + c;
          ++iter;
        }

        unsigned char color = static_cast<unsigned char>(255 - (255.0 * iter / max_iter));
        img(x, y, 0) = color;
        img(x, y, 1) = 0;
        img(x, y, 2) = 255 - color;
      }
    }
  }

  double t1 = omp_get_wtime();
  std::cout << "Temps d'execution (calcul): " << (t1 - t0) << " s\n";

  img.display("Mandelbrot (OpenMP)");
  return 0;
}

/*
g++ julia_omp.cpp -O3 -fopenmp -lX11 -o julia_omp

# sched: 0=static, 1=dynamic, 2=auto
./julia_omp 1920 1080 -2.0 1.0 -1.2 1.2 1000 8 0 1     # static, chunk 1
./julia_omp 1920 1080 -2.0 1.0 -1.2 1.2 1000 8 1 16    # dynamic, chunk 16
./julia_omp 1920 1080 -2.0 1.0 -1.2 1.2 1000 8 2 8     # guided, chunk 8
*/
