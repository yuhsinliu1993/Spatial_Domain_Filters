// Complile:
//		For Linux:
//			g++ hw.cpp -o hw `pkg-config opencv --cflags --libs`
// 		For Mac OSX:
//			g++ hw.cpp -o hw -lopencv_core -lopencv_highgui

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <time.h>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

union RGB_READER {
	float f;
	unsigned char c[4];
};

void read_pcd_file(string, Mat &);
double Gaussian(const double, const double);
Mat Mean_Filter(const Mat &, const int);
Mat Mean_Filter_with_Integral_Image(const Mat &, const int);
Mat Binoimal_Filter(const Mat &, const int);
Mat Binoimal_Filter_Separable(const Mat &, const int);
Mat Bilateral_Filter(const Mat &, const int, const double, const double);
Mat Fast_Bilateral_Filter(const Mat &, const int, const double, const double);
void _fast_bilateral_impl(const Mat &, const Mat &, const Mat &, Mat &, const int, const double, const double);
Vec2d trilinear_interpolation(const Mat, const double, const double, const double);
int clamp(const int, const int, const int);
void type2str(int);


int main(int argc, char const *argv[]) {
	Mat src;

	if (argc != 4){
		printf("Usage: ./p1 <pcd_filename> <filter_size> <file_id>\n");
		return 1;
	}

	read_pcd_file(argv[1], src);

	cout << "Height: " << src.rows << endl;
	cout << "Width: " << src.cols << endl;
	cout << "Depth: " << src.channels() << endl;
	cout << "================ START FILTERING ================" << endl;

	int filter_size = atoi(argv[2]);
	double START, END;
	char name[50];

	imshow("Origin", src);
	sprintf(name, "%s_Origin.jpg", argv[3]);
	imwrite(name, src);

	// Mean Filter
	START = clock();
	Mat mean_f = Mean_Filter(src, filter_size);
	END = clock();
	cout << "EXECUTION TIME OF MEAN_FILTER: " << (END - START) / CLOCKS_PER_SEC << endl;
	imshow("MEAN_FILTER", mean_f);
	sprintf(name, "%s_%s_mean.jpg", argv[3], argv[2]);
	imwrite(name, mean_f);

	// Mean Filter with Integral Image
	START = clock();
	Mat mean_f_i = Mean_Filter_with_Integral_Image(src, filter_size);
	END = clock();
	cout << "EXECUTION TIME OF MEAN_FILTER_WITH_INTEGRAL_IMAGE: " << (END - START) / CLOCKS_PER_SEC << endl;
	// imshow("MEAN_FILTER_WITH_INTEGRAL_IMAGE", mean_f_i);
	sprintf(name, "%s_%s_mean_with_integral.jpg", argv[3], argv[2]);
	imwrite(name, mean_f_i);
	cout << endl;

	// Binoimal Filter
	START = clock();
	Mat binoimal_f = Binoimal_Filter(src, filter_size);
	END = clock();
	cout << "EXECUTION TIME OF BINOMIAL_FILTER: " << (END - START) / CLOCKS_PER_SEC << endl;
	// imshow("BINOMIAL_FILTER", binoimal_f);
	sprintf(name, "%s_%s_binomial.jpg", argv[3], argv[2]);
	imwrite(name, binoimal_f);

	// Separable Binoimal Filter
	START = clock();
	Mat binoimal_f_s = Binoimal_Filter_Separable(src, filter_size);
	END = clock();
	cout << "EXECUTION TIME OF BINOMIAL_FILTER_SEPARABLE: " << (END - START) / CLOCKS_PER_SEC << endl;
	// imshow("BINOMIAL_FILTER_SEPARABLE", binoimal_f_s);
	sprintf(name, "%s_%s_binomial_separable.jpg", argv[3], argv[2]);
	imwrite(name, binoimal_f_s);
	cout << endl;

	// Bilateral Filter
	START = clock();
	Mat bilateral_f = Bilateral_Filter(src, filter_size, 10, 100);
	END = clock();
	cout << "EXECUTION TIME OF BILATERAL_FILTER: " << (END - START) / CLOCKS_PER_SEC << endl;
	// imshow("BILATERAL_FILTER", bilateral_f);
	sprintf(name, "%s_%s_bilateral.jpg", argv[3], argv[2]);
	imwrite(name, bilateral_f);

	// Fast Bilateral Filter
	// START = clock();
	// Mat fast_bilateral_f = Fast_Bilateral_Filter(src, filter_size, 10, 100);
	// END = clock();
	// cout << "EXECUTION TIME OF FAST_BILATERAL_FILTER: " << (END - START) / CLOCKS_PER_SEC << endl;
	// // imshow("FAST_BILATERAL_FILTER", fast_bilateral_f);
	// sprintf(name, "%s_%s_bilateral_grid.jpg", argv[3], argv[2]);
	// imwrite(name, fast_bilateral_f);

	// waitKey(0);

	return 0;
}


void read_pcd_file(string file_name, Mat &src){
	bool read_pixel = false;
	int count = 0, width = 0, height = 0;
	string line;
	string x, y, z, rgb;

	ifstream infile(file_name);

	while(getline(infile, line)){
		infile >> x >> y;

		if (read_pixel){
			// read pixels
			infile >> z >> rgb;

			if (count < width * height){
				RGB_READER tmp;
				tmp.f = (float)atof(rgb.c_str());
				src.at<Vec3b>(count)[0] = tmp.c[0]; // B
				src.at<Vec3b>(count)[1] = tmp.c[1]; // G
				src.at<Vec3b>(count)[2] = tmp.c[2]; // R
			}
			count++;
		} else {
			// read picture informations
			if (x == "WIDTH") width = atoi(y.c_str());
			if (x == "HEIGHT") height = atoi(y.c_str());
			if (x == "DATA" && y == "ascii") read_pixel = true;
			if (width > 0 && height > 0){
				// Using 3-channel unsigned char (CV_8UC3) => Vec3b
				src.create(height, width, CV_8UC3);
			}
		}
	}
	infile.close();
}


double Gaussian(const double sigma, const double x){
	return exp(-(x * x) / (2 * CV_PI * sigma * sigma));
}


Mat Mean_Filter(const Mat &src, const int filter_size){
	int diameter = filter_size / 2;

	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC3);

	for (int center_i = diameter; center_i < src.rows - diameter; center_i++){
		for (int center_j = diameter; center_j < src.cols - diameter; center_j++){
			double sum[3] = {0, 0 ,0};

			for (int x = center_i - diameter; x <= center_i + diameter; x++)
				for (int y = center_j - diameter; y <= center_j + diameter; y++)
					for (int d = 0; d < 3; d++)
						sum[d] += src.at<Vec3b>(x, y)[d];

			for (int d = 0; d < 3; d++)
				dst.at<Vec3b>(center_i, center_j)[d] = sum[d] / (filter_size * filter_size);
		}
	}

	return dst;
}


Mat Mean_Filter_with_Integral_Image(const Mat &src, const int filter_size){
	int diameter = filter_size / 2, ***Integral;
	double avg[3] = {0.0, 0.0, 0.0}	;

	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC3);

	// Initialization
	Integral = new int **[src.rows];
	for (int i = 0; i < src.rows; i++){
		Integral[i] = new int *[src.cols];
		for (int j = 0; j < src.cols; j++)
			Integral[i][j] = new int[3];
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			for (int k = 0; k < 3; k++)
				Integral[i][j][k] = 0;

	// Calculate Integral Image
	for (int i = 0; i < src.rows; i++){
		int sum[3] = {0, 0, 0};

		for (int j = 0; j < src.cols; j++){
			for (int d = 0; d < 3; d++)
				sum[d] += src.at<Vec3b>(i, j)[d];

			if (i > 0){
				for (int d = 0; d < 3; d++)
					Integral[i][j][d] = Integral[i-1][j][d] + sum[d];
			} else {
				for (int d = 0; d < 3; d++)
					Integral[i][j][d] = sum[d];
			}
		}
	}

	for (int center_i = diameter; center_i < src.rows - diameter; center_i++){
		for (int center_j = diameter; center_j < src.cols - diameter; center_j++){
			for (int d = 0; d < 3; d++){
				avg[d] = Integral[center_i+diameter][center_j+diameter][d] +
						 Integral[center_i-diameter][center_j-diameter][d] -
						 Integral[center_i-diameter][center_j+diameter][d] -
						 Integral[center_i+diameter][center_j-diameter][d];

				dst.at<Vec3b>(center_i, center_j)[d] = avg[d] / (filter_size * filter_size);
			}
		}
	}

	return dst;
}


Mat Binoimal_Filter(const Mat &src, const int filter_size){
	int diameter = filter_size / 2;
	int kernel[filter_size];
	int tmp[filter_size];

	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC3);

	for(int i = 0; i<filter_size; i++){
		if (i < 2){
			kernel[i] = 1;
		} else {
			for (int n = 0; n<=i; n++)
				if (n == 0 || n == i) tmp[n] = 1;
				else tmp[n] = kernel[n-1] + kernel[n];
			for (int n = 0; n<=i; n++)
				kernel[n] = tmp[n];
		}
	}

	for (int center_i = diameter; center_i < src.rows - diameter; center_i++){
		for (int center_j = diameter; center_j < src.cols - diameter; center_j++){
			int sum[3] = {0, 0, 0}, m = 0, n = 0;

			for (int x = center_i - diameter; x < center_i + diameter + 1; x++){
				for (int y = center_j - diameter; y < center_j + diameter + 1; y++){
					for(int d = 0; d < 3; d++)
						sum[d] += src.at<Vec3b>(x, y)[d] * kernel[m] * kernel[n];
					m++;
				}
				n++; m=0;
			}

			for(int d = 0; d < 3; d++){
				dst.at<Vec3b>(center_i, center_j)[d] = sum[d] >> 2*(filter_size-1);
			}
		}
	}

	return dst;
}


Mat Binoimal_Filter_Separable(const Mat &src, const int filter_size){
	int diameter = filter_size / 2;
	int kernel[filter_size];
	int tmp[filter_size];

	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC3);

	// Initialization
	int ***Temp;
	Temp = new int **[src.rows];
	for (int i = 0; i < src.rows; i++){
		Temp[i] = new int *[src.cols];
		for (int j = 0; j < src.cols; j++)
			Temp[i][j] = new int[3];
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			for (int k = 0; k < 3; k++)
				Temp[i][j][k] = 0;

	// calculate the kernel
	for(int i = 0; i<filter_size; i++){
		if (i < 2){
			kernel[i] = 1;
		} else {
			for (int n = 0; n<=i; n++)
				if (n == 0 || n == i) tmp[n] = 1;
				else tmp[n] = kernel[n-1] + kernel[n];
			for (int n = 0; n<=i; n++)
				kernel[n] = tmp[n];
		}
	}

	// along x-direction
	for (int x = 0; x < src.rows; x++){
		for (int y = 0; y < src.cols - filter_size + 1; y++){
			for (int k = 0; k < filter_size; k++){
				Temp[x][y][0] += src.at<Vec3b>(x, (y+k))[0] * kernel[k];
				Temp[x][y][1] += src.at<Vec3b>(x, (y+k))[1] * kernel[k];
				Temp[x][y][2] += src.at<Vec3b>(x, (y+k))[2] * kernel[k];
			}
		}
	}

	// along y-direction
	for (int y = 0; y < src.cols - filter_size + 1; y++){
		for (int x = 0; x < src.rows -filter_size + 1; x++){
				int sum[3] = {0, 0, 0};

				for (int k = 0; k < filter_size; k++){
					sum[0] += Temp[(x+k)][y][0] * kernel[k];
					sum[1] += Temp[(x+k)][y][1] * kernel[k];
					sum[2] += Temp[(x+k)][y][2] * kernel[k];
				}

				for (int d = 0; d < 3; d++)
					dst.at<Vec3b>((x+diameter), (y+diameter))[d] = sum[d] >> 2*(filter_size-1);
		}
	}

	return dst;
}


Mat Bilateral_Filter(const Mat &src, const int filter_size, const double sigma_S, const double sigma_R){
	double **Spatial, *Intensity, weight = 0.0, distance;
	int diameter = filter_size / 2;

	Mat dst;
	dst.create(src.rows, src.cols, CV_8UC3);

	// Pre-calculate kernel matrix for spatial closeness
	Spatial = new double *[filter_size];
	for (int i = 0; i < filter_size; i++)
		Spatial[i] = new double [filter_size];

	for (int x = 0; x < 0 + filter_size; x++){
		for (int y = 0; y < 0 + filter_size; y++){
			distance = sqrt(pow((x - diameter), 2) + pow((y - diameter), 2));
			Spatial[x][y] = Gaussian(sigma_S, distance);
		}
	}

	// Pre-calculate kernel matrix for intensity difference
	Intensity = new double[256];
	for (int i = 0; i < 256; i++)
		Intensity[i] = Gaussian(sigma_R, i);

	// Core
	for (int center_i = diameter; center_i < src.rows - diameter; center_i++){
		for (int center_j = diameter; center_j < src.cols - diameter; center_j++){
			double total_weight[3] = {0.0, 0.0, 0.0}, sum[3] = {0.0, 0.0, 0.0};

			// Convolution at the center (center_i, center_j)
			for (int x = center_i - diameter; x < center_i + diameter + 1; x++){
				for (int y = center_j - diameter; y < center_j + diameter + 1; y++){
					for (int d = 0; d < 3; d++){
						int diff = abs(src.at<Vec3b>(x, y)[d] - src.at<Vec3b>(center_i, center_j)[d]);
						weight = Spatial[center_i-x+diameter][center_j-y+diameter] * Intensity[diff];
						total_weight[d] += weight;
						sum[d] += weight * src.at<Vec3b>(x, y)[d];
					}
				}
			}

			for (int d = 0; d < 3; d++)
				dst.at<Vec3b>(center_i, center_j)[d] = sum[d] / total_weight[d];
		}
	}

	return dst;
}

//
// Mat Fast_Bilateral_Filter(const Mat &src, const int filter_size, const double sigma_S, const double sigma_R){
// 	Mat channels[3], _double_src;
//
// 	src.convertTo(_double_src, CV_64FC3);
// 	split(_double_src, channels);  // Split src into B, G, R channels with type CV_64FC1 each
//
// 	Mat dst_tmp, dst;
// 	dst_tmp.create(src.rows, src.cols, CV_64FC3);
//
// 	_fast_bilateral_impl(channels[0], channels[1], channels[2], dst_tmp, filter_size, sigma_S, sigma_R);
//
// 	dst.create(src.rows, src.cols, CV_64FC3);
// 	dst_tmp.convertTo(dst, src.type());
//
// 	return dst;
// }
//
//
// void _fast_bilateral_impl(const Mat &B, const Mat &G ,const Mat &R, Mat &dst, const int filter_size, const double sigma_S, const double sigma_R){
// 	const int diameter = filter_size / 2;
// 	int kernel[filter_size], tmp[filter_size], total;
//
// 	for(int i = 0; i < filter_size; i++){
// 		if (i < 2){
// 			kernel[i] = 1;
// 		} else {
// 			for (int n = 0; n<=i; n++)
// 				if (n == 0 || n == i) tmp[n] = 1;
// 				else tmp[n] = kernel[n-1] + kernel[n];
// 			for (int n = 0; n<=i; n++)
// 				kernel[n] = tmp[n];
// 		}
// 	}
//
// 	for(int i = 0; i < filter_size; i++)
// 		total += kernel[i];
//
// 	Mat tmp_B = Mat(B.size(), CV_64FC1);
// 	Mat tmp_G = Mat(G.size(), CV_64FC1);
// 	Mat tmp_R = Mat(R.size(), CV_64FC1);
//
// 	double b_min, b_max;
// 	double g_min, g_max;
// 	double r_min, r_max;
//
// 	minMaxLoc(B, &b_min, &b_max);
// 	minMaxLoc(G, &g_min, &g_max);
// 	minMaxLoc(R, &r_min, &r_max);
//
// 	const int padding_xy = 2, padding_z = 2, height = B.rows, width = B.cols;
// 	const int downsample_height = (int)floor((height - 1) / sigma_S) + 1 + 2 * padding_xy;
// 	const int downsample_width = (int)floor((width - 1) / sigma_S) + 1 + 2 * padding_xy;
// 	const int downsample_depth_b = (int)floor((b_max - b_min) / sigma_R) + 1 + 2 * padding_z;
// 	const int downsample_depth_g = (int)floor((g_max - g_min) / sigma_R) + 1 + 2 * padding_z;
// 	const int downsample_depth_r = (int)floor((r_max - r_min) / sigma_R) + 1 + 2 * padding_z;
//
// 	cout << downsample_height << " " << downsample_width << " " << downsample_depth_b << " " << downsample_depth_g << " " << downsample_depth_r << endl;
//
// 	int data_size_b[] = {downsample_height, downsample_width, downsample_depth_b};
// 	int data_size_g[] = {downsample_height, downsample_width, downsample_depth_g};
// 	int data_size_r[] = {downsample_height, downsample_width, downsample_depth_r};
//
// 	Mat grid_b(3, data_size_b, CV_64FC2);  // (i, j, range) => 2 channels (wi, w)
// 	Mat grid_g(3, data_size_g, CV_64FC2);
// 	Mat grid_r(3, data_size_r, CV_64FC2);
//
// 	grid_b.setTo(0);
// 	grid_g.setTo(0);
// 	grid_r.setTo(0);
//
// 	// Step 1: Downsampling - Compute the grid location
// 	for (int i = 0; i < height; i++){
// 		for (int j = 0; j < width ; j++){
// 			// Compute grid coordinate
// 			const int di = (int)round((double)i / sigma_S);
// 			const int dj = (int)round((double)j / sigma_S);
//             const int dz_b = (int)round((B.at<double>(i,j) - b_min) / sigma_R);
// 			const int dz_g = (int)round((G.at<double>(i,j) - g_min) / sigma_R);
// 			const int dz_r = (int)round((R.at<double>(i,j) - r_min) / sigma_R);
//
// 			// Retrieve the grid value
//             Vec2d v_b = grid_b.at<Vec2d>(di, dj, dz_b);
// 			Vec2d v_g = grid_g.at<Vec2d>(di, dj, dz_g);
// 			Vec2d v_r = grid_r.at<Vec2d>(di, dj, dz_r);
//
// 			// Updating the downsampled S x R space    => (w * i, w)
//             v_b[0] += B.at<double>(i,j);	v_b[1] += 1.0;
// 			v_g[0] += G.at<double>(i,j);	v_g[1] += 1.0;
// 			v_r[0] += R.at<double>(i,j);	v_r[1] += 1.0;
//
// 			grid_b.at<Vec2d>(di, dj, dz_b) = v_b;
// 			grid_g.at<Vec2d>(di, dj, dz_g) = v_g;
// 			grid_r.at<Vec2d>(di, dj, dz_r) = v_r;
// 		}
// 	}
//
// 	// Step 2: Convolution
// 	Mat buffer_b(3, data_size_b, CV_64FC2);
// 	Mat buffer_g(3, data_size_g, CV_64FC2);
// 	Mat buffer_r(3, data_size_r, CV_64FC2);
// 	buffer_b.setTo(0);
// 	buffer_g.setTo(0);
// 	buffer_r.setTo(0);
//
// 	int offset[3];
//     offset[0] = &(grid_b.at<Vec2d>(1,0,0)) - &(grid_b.at<Vec2d>(0,0,0));
//     offset[1] = &(grid_b.at<Vec2d>(0,1,0)) - &(grid_b.at<Vec2d>(0,0,0));
//     offset[2] = &(grid_b.at<Vec2d>(0,0,1)) - &(grid_b.at<Vec2d>(0,0,0));
//
// 	for (int d = 0; d < 3; d++){  	// For x, y, depth direction
// 		for (int ittr = 0; ittr < 2; ittr++){
// 			// SWAP:  Old_Value <=> New_Value
// 			swap(grid_b, buffer_b);
// 			swap(grid_g, buffer_g);
// 			swap(grid_r, buffer_r);
//
// 			for (int i = 0; i < downsample_height - 0; i++){
// 				for (int j = 0; j < downsample_width - 0; j++){
// 					Vec2d *grid_ptr_b = &(grid_b.at<Vec2d>(i,j,diameter));
// 					Vec2d *grid_ptr_g = &(grid_g.at<Vec2d>(i,j,diameter));
// 					Vec2d *grid_ptr_r = &(grid_r.at<Vec2d>(i,j,diameter));
//
// 	                Vec2d *buffer_ptr_b = &(buffer_b.at<Vec2d>(i,j,diameter));
// 					Vec2d *buffer_ptr_g = &(buffer_g.at<Vec2d>(i,j,diameter));
// 					Vec2d *buffer_ptr_r = &(buffer_r.at<Vec2d>(i,j,diameter));
//
// 					for (int z = diameter; z < downsample_depth_b - diameter ; z++, buffer_ptr_b++, grid_ptr_b++){
// 						for (int h = -diameter; h < -diameter + filter_size; h++)
// 							*grid_ptr_b += *(buffer_ptr_b + h * offset[d]) * kernel[h + diameter];
// 						*grid_ptr_b /= total;
// 					}
//
// 					for (int z = diameter; z < downsample_depth_g - diameter; z++, buffer_ptr_g++, grid_ptr_g++){
// 						for (int h = -diameter; h < -diameter + filter_size; h++)
// 							*grid_ptr_g += *(buffer_ptr_g + h * offset[d]) * kernel[h + diameter];
// 						*grid_ptr_g /= total;
// 					}
//
// 					for (int z = diameter; z < downsample_depth_r - diameter ; z++, buffer_ptr_r++, grid_ptr_r++){
// 						for (int h = -diameter; h < -diameter + filter_size; h++)
// 							*grid_ptr_r += *(buffer_ptr_r + h * offset[d]) * kernel[h + diameter];
// 						*grid_ptr_r /= total;
// 					}
// 				}
// 			}
//
// 		}
// 		// buffer_b.setTo(0);
// 		// buffer_g.setTo(0);
// 		// buffer_r.setTo(0);
// 	}
//
// 	// Step 3: Upsampling
// 	for ( MatIterator_<Vec2d> d = grid_b.begin<Vec2d>(); d != grid_b.end<Vec2d>(); d++){
// 		(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
// 	}
//
// 	for ( MatIterator_<Vec2d> d = grid_g.begin<Vec2d>(); d != grid_g.end<Vec2d>(); d++){
// 		(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
// 	}
//
// 	for ( MatIterator_<Vec2d> d = grid_r.begin<Vec2d>(); d != grid_r.end<Vec2d>(); d++){
// 		(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
// 	}
//
// 	for (int i = 0; i < height; i++){
// 		for (int j = 0; j < width; j++){
// 			const double pi = (double)i / sigma_S;
// 			const double pj = (double)j / sigma_S;
// 			const double pz_b = (B.at<double>(i,j) - b_min) / sigma_R;
// 			const double pz_g = (G.at<double>(i,j) - g_min) / sigma_R;
// 			const double pz_r = (R.at<double>(i,j) - r_min) / sigma_R;
//
// 			tmp_B.at<double>(i,j) = trilinear_interpolation(grid_b, pi, pj, pz_b)[0];
// 			tmp_G.at<double>(i,j) = trilinear_interpolation(grid_g, pi, pj, pz_g)[0];
// 			tmp_R.at<double>(i,j) = trilinear_interpolation(grid_r, pi, pj, pz_r)[0];
// 		}
// 	}
//
// 	// Merge B, G, R channels into a color image
// 	vector<Mat> BGR;
//
// 	BGR.push_back(tmp_B);
// 	BGR.push_back(tmp_G);
// 	BGR.push_back(tmp_R);
//
// 	merge(BGR, dst);
// }
//
//
// Vec2d trilinear_interpolation(const Mat mat, const double y, const double x, const double z){
//     const int height = mat.size[0];
//     const int width  = mat.size[1];
//     const int depth  = mat.size[2];
//
//     const int y_idx  = clamp(0, height-1, (int)y);
//     const int yy_idx = clamp(0, height-1, y_idx + 1);
//     const int x_idx  = clamp(0, width-1, (int)x);
//     const int xx_idx = clamp(0, width-1, x_idx + 1);
//     const int z_idx  = clamp(0, depth-1, (int)z);
//     const int zz_idx = clamp(0, depth-1, z_idx + 1);
//
// 	const double y_alpha = y - y_idx;
//     const double x_alpha = x - x_idx;
//     const double z_alpha = z - z_idx;
//
//     return  (1.0-y_alpha) * (1.0-x_alpha) * (1.0-z_alpha) * mat.at<Vec2d>(y_idx, x_idx, z_idx) 	 +
//         	(1.0-y_alpha) * x_alpha       * (1.0-z_alpha) * mat.at<Vec2d>(y_idx, xx_idx, z_idx)  +
//         	y_alpha       * (1.0-x_alpha) * (1.0-z_alpha) * mat.at<Vec2d>(yy_idx, x_idx, z_idx)  +
//         	y_alpha       * x_alpha       * (1.0-z_alpha) * mat.at<Vec2d>(yy_idx, xx_idx, z_idx) +
//         	(1.0-y_alpha) * (1.0-x_alpha) * z_alpha       * mat.at<Vec2d>(y_idx, x_idx, zz_idx)  +
//         	(1.0-y_alpha) * x_alpha       * z_alpha       * mat.at<Vec2d>(y_idx, xx_idx, zz_idx) +
//         	y_alpha       * (1.0-x_alpha) * z_alpha       * mat.at<Vec2d>(yy_idx, x_idx, zz_idx) +
//         	y_alpha       * x_alpha       * z_alpha       * mat.at<Vec2d>(yy_idx, xx_idx, zz_idx);
// }
//
//
// int clamp(const int min, const int max, const int x){
//     return ( x < min ) ? min : ( x < max ) ? x : max;
// }
//
//
// void type2str(int type){
// 	string r;
//
// 	uchar depth = type & CV_MAT_DEPTH_MASK;
// 	uchar chans = 1 + (type >> CV_CN_SHIFT);
//
// 	switch ( depth ) {
// 		case CV_8U:  r = "8U"; break;
// 		case CV_8S:  r = "8S"; break;
// 		case CV_16U: r = "16U"; break;
// 		case CV_16S: r = "16S"; break;
// 		case CV_32S: r = "32S"; break;
// 		case CV_32F: r = "32F"; break;
// 		case CV_64F: r = "64F"; break;
// 		default:     r = "User"; break;
// 	}
//
// 	r += "C";
// 	r += (chans+'0');
//
// 	std::cout << r << std::endl;
// }
