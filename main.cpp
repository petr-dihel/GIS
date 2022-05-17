#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stack>

#include <cmath>

#include <opencv2/opencv.hpp>

#include <proj.h>

#include "defines.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <filesystem>
#include <algorithm>    // std::min

#define STEP1_WIN_NAME "Heightmap"
#define STEP2_WIN_NAME "Edges"
#define ZOOM           1


struct MouseProbe {
    cv::Mat & heightmap_8uc1_img_;
    cv::Mat & heightmap_show_8uc3_img_;
    cv::Mat & edgemap_8uc1_img_;

    MouseProbe( cv::Mat & heightmap_8uc1_img, cv::Mat & heightmap_show_8uc3_img, cv::Mat & edgemap_8uc1_img )
     : heightmap_8uc1_img_( heightmap_8uc1_img ), heightmap_show_8uc3_img_( heightmap_show_8uc3_img ), edgemap_8uc1_img_( edgemap_8uc1_img )
    {
    }
};

// variables

// function declarations
void flood_fill( cv::Mat & src_img, cv::Mat & dst_img, const int x, const int y );


/**
 * Mouse clicking callback.
 */
void mouse_probe_handler( int event, int x, int y, int flags, void* param ) {
    MouseProbe *probe = (MouseProbe*)param;

    switch ( event ) {

    case cv::EVENT_LBUTTONDOWN:
        printf( "Clicked LEFT at: [ %d, %d ]\n", x, y );
        flood_fill( probe->edgemap_8uc1_img_, probe->heightmap_show_8uc3_img_, x, y );
        break;

    case cv::EVENT_RBUTTONDOWN:
        printf( "Clicked RIGHT at: [ %d, %d ]\n", x, y );
        break;
    }
}


void create_windows( const int width, const int height ) {
    cv::namedWindow( STEP1_WIN_NAME, 0 );
    cv::namedWindow( STEP2_WIN_NAME, 0 );

    cv::resizeWindow( STEP1_WIN_NAME, width*ZOOM, height*ZOOM );
    cv::resizeWindow( STEP2_WIN_NAME, width*ZOOM, height*ZOOM );

} // create_windows


/**
 * Perform flood fill from the specified point (x, y) for the neighborhood points if they contain the same value,
 * as the one that came in argument 'value'. Function recursicely call itself for its 4-neighborhood.
 * 
 * edgemap_8uc1_img - image, in which we perform flood filling
 * heightmap_show_8uc3_img - image, in which we display the filling
 * value - value, for which we'll perform flood filling
 */
void fill_step( cv::Mat & edgemap_8uc1_img, cv::Mat & heightmap_show_8uc3_img, int y, int x, const uchar value ) {
    int width = edgemap_8uc1_img.cols , height = edgemap_8uc1_img.rows;

    std::stack<cv::Point> *points = new std::stack<cv::Point>();
    points->push(cv::Point(x, y)); 
    while (points->size() > 0)
    {
        printf("debug01 \n");
        cv::Point point = points->top();
        points->pop();
        
        if (point.x < 0 || point.x > width || point.y < 0 || point.y > height)
        {
            continue;
        }
        uchar currentPixelValue = edgemap_8uc1_img.at<uchar>(point.y, point.x);
        if (currentPixelValue != value) {
            continue;
        }

        edgemap_8uc1_img.at<uchar>(point.y, point.x) = 1;
        heightmap_show_8uc3_img.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 0, 255);
     printf("debug02 \n");
        points->push(cv::Point(point.x+1, point.y)); 
        points->push(cv::Point(point.x, point.y+1)); 
        points->push(cv::Point(point.x-1, point.y)); 
        points->push(cv::Point(point.x, point.y-1));
    }
    
} //fill_step



void flood_fill_inner(cv::Mat & edgemap_8uc1_img, cv::Mat & heightmap_show_8uc3_img, const int x, const int y, const float pixelValue) {
    
    if (x < 0 || x > edgemap_8uc1_img.cols || y < 0 || y > edgemap_8uc1_img.rows)
    {
        return;
    }
    uchar currentPixelValue = edgemap_8uc1_img.at<uchar>(y, x);
    if (currentPixelValue != pixelValue) {
        return;
    }

    edgemap_8uc1_img.at<uchar>(y, x) = 1;
    heightmap_show_8uc3_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,255);
    flood_fill_inner(edgemap_8uc1_img, heightmap_show_8uc3_img, x+1, y, pixelValue);
    flood_fill_inner(edgemap_8uc1_img, heightmap_show_8uc3_img, x, y+1, pixelValue);
    flood_fill_inner(edgemap_8uc1_img, heightmap_show_8uc3_img, x-1, y, pixelValue);
    flood_fill_inner(edgemap_8uc1_img, heightmap_show_8uc3_img, x, y-1, pixelValue);  
    //pixels = [(x+1, y), ]
    //flood_fill_inner(edgemap_8uc1_img, heightmap_show_8uc3_img, [], pixelValue); 
}


/**
 * Perform flood fill from the specified point (x, y). The function remembers the value at the coordinate (x, y)
 * and fill the neighborhood using 'fill_step' function so long as the value in the neighborhood points are the same.
 * Execute the fill on a temporary image to prevent the original image from being repainted.

 * edgemap_8uc1_img - image, in which we perform flood filling
 * heightmap_show_8uc3_img - image, in which we display the filling
 */
void flood_fill( cv::Mat & edgemap_8uc1_img, cv::Mat & heightmap_show_8uc3_img, const int x, const int y ) {
    cv::Mat tmp_edgemap_8uc1_img;
    printf("debug0 \n");
    // todo
    edgemap_8uc1_img.copyTo(tmp_edgemap_8uc1_img);

    heightmap_show_8uc3_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,255);
    
    uchar pixelValue = tmp_edgemap_8uc1_img.at<uchar>(y, x);
    uchar tes = pixelValue;
    if (pixelValue = 1) {
        //return;
    }

    fill_step(tmp_edgemap_8uc1_img, heightmap_show_8uc3_img, y, x, tes);
} //flood_fill


/**
 * Find the minimum and maximum coordinates in the file.
 * Note that the file is the S-JTSK coordinate system.
 */
void get_min_max( const char *filename, float *a_min_x, float *a_max_x, float *a_min_y, float *a_max_y, float *a_min_z, float *a_max_z, int current_l_type = -1) {
    FILE *f = NULL;
    float x, y, z;
    float min_x, min_y, min_z, max_x, max_y, max_z;
    int l_type;
    
    std::ifstream iFile(filename, std::ios::in | std::ios::binary);

    if(!iFile.is_open()) {
        std::cout<<"File not opened "<< filename <<std::endl;
    } 

    while (true) {
        iFile.read(reinterpret_cast<char*>(&min_x), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&min_y), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&min_z), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&l_type), sizeof(int));
        if (current_l_type >= 0) {
            if (current_l_type == l_type) {
                max_x = min_x;
                max_y = min_y;
                max_z = min_z;
                break;
            }
        } else {
            max_x = min_x;
            max_y = min_y;
            max_z = min_z;
            break;
        }
    }
    while(!iFile.fail()) {
        iFile.read(reinterpret_cast<char*>(&x), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&y), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&z), sizeof(float));
        iFile.read(reinterpret_cast<char*>(&l_type), sizeof(int));

        if (current_l_type >= 0 && current_l_type != l_type) {
            continue;
        } 

        if (x < min_x) {
            min_x = x;
        } else {
            if (x > max_x) {
                max_x = x;
            }
        }

        if (y < min_y) {
            min_y = y;
        } else {
            if (y > max_y) {
                max_y = y;
            }
        }

        if (z < min_z) {
            min_z = z;
        } else {
            if (z > max_z) {
                max_z = z;
            }
        }
    }    
    iFile.close();
    *a_min_x = min_x;
    *a_min_y = min_y;
    *a_min_z = min_z;
    *a_max_x = max_x;
    *a_max_y = max_y;
    *a_max_z = max_z;
}


/**
 * Fill the image by data from lidar.
 * All lidar points are stored in a an array that has the dimensions of the image. Then the pixel is assigned
 * a value as an average value range from at the corresponding array element. However, with this simple data access, you will lose data precission.
 * filename - file with binarny data
 * img - input image
 */
void fill_image( const char *filename, cv::Mat & heightmap_8uc1_img, float min_x, float max_x, float min_y, float max_y, float min_z, float max_z, int current_l_type = -1 ) {
    FILE *f = NULL;
    int delta_x, delta_y, delta_z;
    float fx, fy, fz;
    int x, y, l_type;
    int stride;
    int num_points = 1;
    float range = 0.0f;
    float *sum_height = NULL;
    int *sum_height_count = NULL;

    // zjistime sirku a vysku obrazu
    delta_x = round( max_x - min_x + 0.5f );
    delta_y = round( max_y - min_y + 0.5f );
    delta_z = round( max_z - min_z + 0.5f );

    stride = delta_x;

    int arraySize = heightmap_8uc1_img.rows * heightmap_8uc1_img.cols;
    // 1:
    // We allocate helper arrays, in which we store values from the lidar
    // and the number of these values for each pixel
    sum_height = new float[arraySize];
    sum_height_count = new int[arraySize];
    for (unsigned int i = 0; i < arraySize; i++)
    {
        sum_height[i] = 0;
        sum_height_count[i] = 0;
    }
    
    // 2:
    // go through the file and assign values to the field
    // beware that in S-JTSK the beginning of the co-ordinate system is at the bottom left,
    // while in the picture it is top left
    f = fopen(filename, "rb");
    if (!f)
    {
        std::cout << "file not opened" << std::endl; 
        return;
    }
    
    while (!feof(f))
    {
        fread(&fx, sizeof(float), 1, f);
        fread(&fy, sizeof(float), 1, f);
        fread(&fz, sizeof(float), 1, f);
        fread(&l_type, sizeof(float), 1, f);
        if (current_l_type >= 0 && current_l_type != l_type) {
            continue;
        }

        x = (fx + abs(min_x));
        y = (fy + abs(min_y));
        int index = heightmap_8uc1_img.cols*y+x;
        sum_height[index] += fz;
        sum_height_count[index]++;
    }
    fclose(f);
    
    range = 255/(max_z - min_z);
    // 3:
    // assign values from the helper field into the image
    for (unsigned int i = 0; i < heightmap_8uc1_img.rows; i++)
    {
        for (unsigned int j = 0; j < heightmap_8uc1_img.cols; j++)
        {
            int index = heightmap_8uc1_img.cols*i+j;
            float sum = sum_height[index];
            int count = sum_height_count[index];
            float value = sum_height[index]/sum_height_count[index];
            value = std::max(0, (int)((value-min_z) * range));
            
            heightmap_8uc1_img.at<uchar>(i,j) = (int)value;
        }

    }
    
}

void make_edges( const cv::Mat & src_8uc1_img, cv::Mat & edgemap_8uc1_img ) {
    cv::Canny( src_8uc1_img, edgemap_8uc1_img, 1, 80 );
}

/**
 * Transforms the image so it contains only two values.
 * Threshold may be set experimentally.
 */
void binarize_image( cv::Mat & src_8uc1_img ) {
    int x, y;
    uchar value;

    for (y = 0; y < src_8uc1_img.rows; y++)
    {
        for (x = 0; x < src_8uc1_img.cols; x++)
        {
            value = src_8uc1_img.at<uchar>(y,x) > 128 ? 255 : 0; 
            src_8uc1_img.at<uchar>(y,x) = value;
        }
    }
    //cv::imshow("BIN", src_8uc1_img);
}


void dilate_and_erode_edgemap( cv::Mat & edgemap_8uc1_img ) {
    //cv3
    cv::Mat result_dir = cv::Mat::zeros(edgemap_8uc1_img.rows, edgemap_8uc1_img.cols, CV_8UC1);
    
    cv::Mat filter = cv::Mat::zeros(3,3, CV_8UC1);
    // dilatace
    for (int y = 0; y < edgemap_8uc1_img.rows; y++)
    {
        for (int x = 0; x < edgemap_8uc1_img.cols; x++)
        {
            
            uchar value = edgemap_8uc1_img.at<uchar>(y, x);
            if (value == 255) {
                for (int filterX = -1; filterX < (filter.rows - 1); filterX++)
                {
                    for (int filterY = -1; filterY < (filter.cols - 1); filterY++)
                    {
                        int shiftX = x + filterX;
                        int shiftY = y + filterY;
                        if (shiftX < 0 || shiftX > edgemap_8uc1_img.cols || shiftY < 0 || shiftY > edgemap_8uc1_img.rows) {
                          continue;  
                        } 
                        result_dir.at<uchar>(shiftY, shiftX) = 255;
                    }
                }
            }
            
        }
    }

    //cv::imshow("dil", result_dir);
    // eroze
    
    cv::Mat result_eroz = result_dir.clone();

    for (int y = 0; y < edgemap_8uc1_img.rows; y++)
    {
        for (int x = 0; x < edgemap_8uc1_img.cols; x++)
        {
            
            uchar value = result_dir.at<uchar>(y, x);
            bool isDifferent = false;
            if (value == 255) {
                for (int filterX = -1; filterX < (filter.rows - 1); filterX++)
                {
                    for (int filterY = -1; filterY < (filter.cols - 1); filterY++)
                    {
                        int shiftX = x + filterX;
                        int shiftY = y + filterY;
                        if (shiftX < 0 || shiftX > edgemap_8uc1_img.cols || shiftY < 0 || shiftY > edgemap_8uc1_img.rows) {
                          continue;  
                        } 
                        if (result_dir.at<uchar>(shiftY, shiftX) != 255) {
                            isDifferent = true;
                            break;
                        }
                    }
                }
            }
            if (isDifferent) {
                result_eroz.at<uchar>(y, x) = 0;
            }
            
        }
    }
    //cv::imshow("dil_ero", result_eroz);
    edgemap_8uc1_img = result_eroz;
}

void heigth_maps(const char *bin_filename) {
    std::cout << "heigth_maps" << std::endl;
    const int max_l_type = 3;
    cv::Mat h_maps[max_l_type];
    for (int i = 0; i < max_l_type; i++)
    {
        float min_x, max_x, min_y, max_y, min_z, max_z;
        float delta_x, delta_y, delta_z;
        get_min_max( bin_filename, &min_x, &max_x, &min_y, &max_y, &min_z, &max_z, i);

        delta_x = max_x - min_x;
        delta_y = max_y - min_y;
        delta_z = max_z - min_z;

        h_maps[i] = cv::Mat::zeros( cv::Size( cvRound( delta_x + 0.5f ), cvRound( delta_y + 0.5f ) ), CV_8UC1);
        fill_image( bin_filename, h_maps[i], min_x, max_x, min_y, max_y, min_z, max_z, i);
    }
    
    cv::imshow("ltype0", h_maps[0]);
    cv::imshow("ltype1", h_maps[1]);
    cv::imshow("ltype2", h_maps[2]);
    cv::imwrite("./ltype0.jpg", h_maps[0]);
    cv::imwrite("./ltype1.jpg", h_maps[1]);
    cv::imwrite("./ltype2.jpg", h_maps[2]);
    cv::waitKey(0);
}

int krovak(cv::Point2d min,  cv::Point2d max) {
    PJ *P;
    PJ_COORD c, c_out;
    P = proj_create_crs_to_crs(PJ_DEFAULT_CTX,
    
                               "+proj=krovak +ellps=bessel +towgs84=570.8,85.7,462.8,4.998,1.587,5.261,3.56",
                               "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                               NULL);
    if (P==0)
        return 1;
    {
        /* For that particular use case, this is not needed. */
        /* proj_normalize_for_visualization() ensures that the coordinate */
        /* order expected and returned by proj_trans() will be longitude, */
        /* latitude for geographic CRS, and easting, northing for projected */
        /* CRS. If instead of using PROJ strings as above, "EPSG:XXXX" codes */
        /* had been used, this might had been necessary. */
        PJ* P_for_GIS = proj_normalize_for_visualization(PJ_DEFAULT_CTX, P);
        if( 0 == P_for_GIS )  {
            proj_destroy(P);
            return 1;
        }
        proj_destroy(P);
        P = P_for_GIS;
    }

    /* For reliable geographic <--> geocentric conversions, z shall not */
    /* be some random value. Also t shall be initialized to HUGE_VAL to */
    /* allow for proper selection of time-dependent operations if one of */
    /* the CRS is dynamic. */
    c.lpzt.z = 0.0;
    c.lpzt.t = HUGE_VAL;

    c.lpzt.lam =  min.x;
    c.lpzt.phi = min.y;
    c_out = proj_trans(P, PJ_FWD, c);
    printf("MIN wgs84: %.5f\t%.5f\n", c_out.xy.x, c_out.xy.y);
   
    c.lpzt.lam =  max.x;
    c.lpzt.phi = max.y;
    c_out = proj_trans(P, PJ_FWD, c);
    printf("MAX wgs84: %f \t %f \n", c_out.xy.x, c_out.xy.y);

/*
    while (scanf("%lf %lf", &c.lpzt.lam, &c.lpzt.phi) == 2) {
   
        c_out = proj_trans(P, PJ_FWD, c);
        printf("%.2f\t%.2f\n", c_out.xy.x, c_out.xy.y);
    }*/

    proj_destroy(P);

    return 0;
}

void process_lidar( const char *txt_filename, const char *bin_filename, const char *img_filename ) {
    float min_x, max_x, min_y, max_y, min_z, max_z;
    float delta_x, delta_y, delta_z;
    MouseProbe *mouse_probe;

    cv::Mat heightmap_8uc1_img;      // image of source of lidar data
    cv::Mat heightmap_show_8uc3_img; // image to detected areas
    cv::Mat edgemap_8uc1_img;        // image for edges

    get_min_max( bin_filename, &min_x, &max_x, &min_y, &max_y, &min_z, &max_z );

    
    printf( "min x: %f, max x: %f\n", min_x, max_x );
    printf( "min y: %f, max y: %f\n", min_y, max_y );
    printf( "min z: %f, max z: %f\n", min_z, max_z );

    krovak(cv::Point2d(min_x, min_y), cv::Point2d(max_x, max_y));

    delta_x = max_x - min_x;
    delta_y = max_y - min_y;
    delta_z = max_z - min_z;

    printf( "delta x: %f\n", delta_x );
    printf( "delta y: %f\n", delta_y );
    printf( "delta z: %f\n", delta_z );

    // create images according to data from the source file
    
    heightmap_8uc1_img = cv::Mat( cv::Size( cvRound( delta_x + 0.5f ), cvRound( delta_y + 0.5f ) ), CV_8UC1 );
    heightmap_show_8uc3_img = cv::Mat( cv::Size( cvRound( delta_x + 0.5f ), cvRound( delta_y + 0.5f ) ), CV_8UC3 );
    edgemap_8uc1_img = cv::Mat( cv::Size( cvRound( delta_x + 0.5f ), cvRound( delta_y + 0.5f ) ), CV_8UC3 );

    create_windows( heightmap_8uc1_img.cols, heightmap_8uc1_img.rows );
    // edgemap or after
    mouse_probe = new MouseProbe( heightmap_8uc1_img, heightmap_show_8uc3_img, edgemap_8uc1_img );

    cv::setMouseCallback( STEP1_WIN_NAME, mouse_probe_handler, mouse_probe );
    cv::setMouseCallback( STEP2_WIN_NAME, mouse_probe_handler, mouse_probe );

    printf( "Image w=%d, h=%d\n", heightmap_8uc1_img.cols, heightmap_8uc1_img.rows );
    
    // fill the image with data from lidar scanning
    fill_image( bin_filename, heightmap_8uc1_img, min_x, max_x, min_y, max_y, min_z, max_z );
    cv::cvtColor( heightmap_8uc1_img, heightmap_show_8uc3_img, cv::COLOR_GRAY2RGB);

    // create edge map from the height image
    make_edges( heightmap_8uc1_img, edgemap_8uc1_img );

    // binarize image, so we can easily process it in the next step
    binarize_image( edgemap_8uc1_img );
    
    // implement image dilatation and erosion
    dilate_and_erode_edgemap( edgemap_8uc1_img );


    cv::Mat save_8uc4_img;
    cv::cvtColor( heightmap_8uc1_img, save_8uc4_img, cv::COLOR_GRAY2BGRA);

    cv::Mat flipped_8uc4 = cv::Mat::zeros(save_8uc4_img.size(), CV_8UC4);
    for (int y = 0; y < save_8uc4_img.rows; y++)
    {
        for (int x = 0; x < save_8uc4_img.cols; x++)
        {

            cv::Vec4b srcPixel = save_8uc4_img.at<cv::Vec4b>(y, x);
            
            int flippedY = save_8uc4_img.rows - y;
            int flippedX = save_8uc4_img.cols - x;
            
            //cv::Vec4b destPixel = save_8uc4_img.at<cv::Vec4b>(flippedY, flippedX);

            if (srcPixel[0] == 0 && srcPixel[1] == 0 && srcPixel[2] == 0 ) {
                srcPixel[3] = 0;
            }


            flipped_8uc4.at<cv::Vec4b>(flippedY, x) = srcPixel;
        }
    }
    
    cv::imwrite( img_filename, flipped_8uc4 );

    // wait here for user input using (mouse clicking)
    

    while ( 1 ) {
        cv::imshow( STEP1_WIN_NAME, heightmap_show_8uc3_img );
        //cv::imshow( STEP2_WIN_NAME, edgemap_8uc1_img );
        int key = cv::waitKey( 10 );
        if ( key == 'q' ) {
            break;
        }
    }
    
}

int main( int argc, char *argv[] ) {
    char *txt_file, *bin_file, *img_file;

    printf("argc: %d\n", argc );
    
    //./build/gis test.txt pt000023.bin pt000023.png
    if ( argc < 4 ) {
        printf( "Not enough command line parameters.\n" );
        //txt_file = "../test.txt";
        //bin_file = "../pt000023.bin";
        //img_file = "../pt000023.png";
    } else {        
        txt_file = argv[ 1 ];
        bin_file = argv[ 2 ];
        img_file = argv[ 3 ];
    }
    
    char tmp[256];
    getcwd(tmp, 256);

    std::cout << "Current path " << tmp << std::endl;
    process_lidar( txt_file, bin_file, img_file );
    
    // task 5
    //heigth_maps(bin_file);

    return 0;
}
