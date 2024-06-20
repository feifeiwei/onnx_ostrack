#include <iostream>
#include <opencv2/opencv.hpp>
// #include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace nvinfer1;
// using namespace onnxruntime;


// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out info messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


// Load TRT engine from file
ICudaEngine* loadEngine(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "ERROR: Unable to load engine file: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size, nullptr);
    return engine;
}

float sample_target(const cv::Mat& im, cv::Mat& im_crop_pad, const cv::Rect& target_box, float search_area_factor, int output_wh)
{
    /* Extracts a square crop centered at target_bb box, of area search_area_factor ^ 2 times target_bb area

        args:
        im - cv image
        target_bb - target box[x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float)Size to which the extracted crop is resized(always square).If None, no resizing is done.

        returns :
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
      */

    int im_width = im.cols;
    int im_height = im.rows;

    int x = target_box.x;
    int y = target_box.y;
    int w = target_box.width;
    int h = target_box.height;

    int crop_sz = std::ceil(std::sqrt(w * h) * search_area_factor);
     
    if (crop_sz < 1) return -1;

    int x1 =round( x + w / 2. - crop_sz / 2.);
    int x2 = x1 + crop_sz;
    int y1 = round(y + h / 2. - crop_sz / 2.);
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - im_width + 1, 0);
    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2 - im_height + 1, 0);

    // Crop target
    cv::Mat im_crop = im(cv::Rect(x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - (x1 + x1_pad), y2 - y2_pad -(y1 + y1_pad) ));
    cv::copyMakeBorder(im_crop, im_crop_pad, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT, 0);
    float resize_factor = (float)output_wh / crop_sz;
    cv::resize(im_crop_pad, im_crop_pad, cv::Size(output_wh, output_wh));

    return resize_factor;
}

vector<float> cal_bbox(Mat& score_map_ctr, Mat& size_map, Mat& offset_map) {
    // Implement cal_bbox function logic here
}



cv::Rect map_box_back(const cv::Rect& state, const cv::Rect2f& pred_box, const float resize_factor, const int search_size)
{
        int cx_prev = state.x + 0.5 * state.width;
        int cy_prev = state.y + 0.5 * state.height;

        // int search_size = 256;

        int cx = pred_box.x;
        int cy = pred_box.y;
        int w =  pred_box.width;
        int h =  pred_box.height;
        int half_side = search_size * 0.5 / resize_factor;

        int cx_real = cx + (cx_prev - half_side);
        int cy_real = cy + (cy_prev - half_side);

        cv::Rect ori_box(cx_real - w / 2, cy_real - h / 2, w, h);
 
        return ori_box;
}

vector<int> clip_box(vector<int>& box, int H, int W, int margin) {
    // Implement clip_box function logic here
}

int main() {
    string videofilepath = "../bag.avi";

    std::string engineFile = "../sim_os_track_fp32.engine";

    // parameters
    const int batchSize = 1;
    const int zWidth = 128, zHeight = 128;
    const int xWidth = 256, xHeight = 256;
    const int feat_size=16;

    const int template_factor = 2;
    const int template_size = 128;

    const int search_factor = 4;
    const int search_size = 256;

    cv::Rect init_box = {316, 138, 110, 118};


    VideoCapture cap(videofilepath);
    VideoWriter out("output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20.0, Size(480, 360));

    Mat frame;
    cap.read(frame);
    int H = frame.rows;
    int W = frame.cols;

    Mat z_patch_arr, x_patch_arr, x_patch_arr_cp, z_amask_arr, x_amask_arr;

    bool cudaEnabled{};

    // cv::dnn::Net net = cv::dnn::readNetFromONNX("../sim_cnn_track.onnx");
    // if (cudaEnabled)
    // {
    //     std::cout << "\nRunning on CUDA" << std::endl;
    //     net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //     net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // }
    // else
    // {
    //     std::cout << "\nRunning on CPU" << std::endl;
    //     net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //     net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // }

    
    cv::Mat z_template, x_search;

    


    float resize_factor = sample_target(frame, z_template, init_box, template_factor, template_size);

    cv::imwrite("test_tmeplate.jpg", z_template);

    cv::Rect state = init_box;

    cv::Rect2f pred_bbox; // 预测框
    int counts=0;


    // Load TensorRT engine from file
    
    ICudaEngine* engine = loadEngine(engineFile);
    if (!engine) {
        return -1;
    }

    // Create execution context from the engine
    IExecutionContext* context = engine->createExecutionContext();

    // Allocate GPU buffers for inputs and outputs
    

    const int scoreMapSize  = 1*1*feat_size*feat_size; // Size of output for "score_map"
    const int sizeMapSize   = 1*2*feat_size*feat_size;  // Size of output for "size_map"
    const int offsetMapSize = 1*2*feat_size*feat_size; // Size of output for "offset_map"

    // float* zInput = new float[zWidth * zHeight * 3];
    // float* xInput = new float[xWidth * xHeight * 3];

    cv::Mat z_template_out;
    cv::Mat x_search_out;

    z_template_out.create({1, 3, (int)zWidth, (int)zWidth}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(z_template, channels);

    cv::Mat c0((int)zWidth, (int)zWidth, CV_32F, (float*)z_template_out.data);
    cv::Mat c1((int)zWidth, (int)zWidth, CV_32F, (float*)z_template_out.data + (int)zWidth * (int)zWidth);
    cv::Mat c2((int)zWidth, (int)zWidth, CV_32F, (float*)z_template_out.data + (int)zWidth * (int)zWidth * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    // if (z_template.type() != CV_32F) {
    //     z_template.convertTo(z_template, CV_32F, 1.0 / 255.0); // 先进行归一化，除以255
    // }

    // std::cout << z_template << std::endl;
    float* zInput = reinterpret_cast<float*>(z_template_out.data);

    float* d_zInput, * d_xInput;
    float* d_scoreMap, * d_sizeMap, * d_offsetMap;

    while (true) {

        cap.read(frame);
        if (frame.empty()) break;

    //     Mat img_255 = frame.clone();

    //     // Sample target patch
    //  
        auto start = std::chrono::high_resolution_clock::now();

    //     // Prepare search patch
    //     sample_target(frame, init_bbox, 4, 256, x_patch_arr, resize_factor, x_amask_arr);
        float resize_factor = sample_target(frame, x_search, state, search_factor, search_size);

        // if (x_search.type() != CV_32F) {
        //     x_search.convertTo(x_search, CV_32F, 1.0 / 255.0); // 先进行归一化，除以255
        // }

        x_search_out.create({1, 3, (int)xWidth, (int)xWidth}, CV_32F);

        std::vector<cv::Mat> channels;
        cv::split(x_search, channels);

        cv::Mat c0((int)xWidth, (int)xWidth, CV_32F, (float*)x_search_out.data);
        cv::Mat c1((int)xWidth, (int)xWidth, CV_32F, (float*)x_search_out.data + (int)xWidth * (int)xWidth);
        cv::Mat c2((int)xWidth, (int)xWidth, CV_32F, (float*)x_search_out.data + (int)xWidth * (int)xWidth * 2);

        channels[0].convertTo(c2, CV_32F, 1 / 255.f);
        channels[1].convertTo(c1, CV_32F, 1 / 255.f);
        channels[2].convertTo(c0, CV_32F, 1 / 255.f);

        float* xInput = reinterpret_cast<float*>(x_search_out.data);


         // // Allocate GPU memory
        
        cudaMalloc((void**)&d_zInput, batchSize * zWidth * zHeight * 3 * sizeof(float));
        cudaMalloc((void**)&d_xInput, batchSize * xWidth * xHeight * 3 * sizeof(float));

        // Copy input data to GPU
        cudaMemcpy(d_zInput, zInput, batchSize * zWidth * zHeight * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xInput, xInput, batchSize * xWidth * xHeight * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate GPU memory for outputs
        cudaMalloc((void**)&d_scoreMap, batchSize * scoreMapSize * sizeof(float));
        cudaMalloc((void**)&d_sizeMap, batchSize * sizeMapSize * sizeof(float));
        cudaMalloc((void**)&d_offsetMap, batchSize * offsetMapSize * sizeof(float));

        int numBindings = engine->getNbBindings();

        void* buffers[numBindings];
        

        // 设置输入和输出张量的绑定顺序
        int zInputIndex = engine->getBindingIndex("z"); // 示例：获取zInput的索引
        int xInputIndex = engine->getBindingIndex("x"); // 示例：获取xInput的索引

        // int scoreMapIndex = engine->getBindingIndex("577"); // 示例：获取scoreMap的索引
        // int sizeMapIndex = engine->getBindingIndex("579"); // 示例：获取sizeMap的索引
        // int offsetMapIndex = engine->getBindingIndex("575"); // 示例：获取offsetMap的索引

        int scoreMapIndex = engine->getBindingIndex("1271"); // 示例：获取scoreMap的索引
        int sizeMapIndex = engine->getBindingIndex("1272"); // 示例：获取sizeMap的索引
        int offsetMapIndex = engine->getBindingIndex("1257"); // 示例：获取offsetMap的索引

        // cout <<"numBindings: " << numBindings << endl;
        // cout <<"zInputIndex: " << zInputIndex << endl;
        // cout <<"xInputIndex: " << xInputIndex << endl;
        // cout <<"scoreMapIndex: " << scoreMapIndex << endl;
        // cout <<"sizeMapIndex: " << sizeMapIndex << endl;
        // cout <<"offsetMapIndex: " << offsetMapIndex << endl;

        // Bind input and output buffers to the execution context
        buffers[zInputIndex] = d_zInput;
        buffers[xInputIndex] = d_xInput;
        buffers[scoreMapIndex] = d_scoreMap;
        buffers[sizeMapIndex] = d_sizeMap;
        buffers[offsetMapIndex] = d_offsetMap;

        // const char* inputBindings[] = { "z", "x" }; // Names of input bindings in the engine
        // const char* outputBindings[] = { "577", "579", "575" }; // Names of output bindings in the engine

        // Execute inference
        context->execute(batchSize, buffers);

        // Copy output data from GPU
        float* scoreMapOutput = new float[batchSize * scoreMapSize];
        float* sizeMapOutput = new float[batchSize * sizeMapSize];
        float* offsetMapOutput = new float[batchSize * offsetMapSize]; //0-1?

        cudaMemcpy(scoreMapOutput, d_scoreMap, batchSize * scoreMapSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(sizeMapOutput, d_sizeMap, batchSize * sizeMapSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(offsetMapOutput, d_offsetMap, batchSize * offsetMapSize * sizeof(float), cudaMemcpyDeviceToHost);


        int idx_x=0;
        int idx_y=0;

        float max_conf = 0;
        float w = 0.;
        float h = 0.;
        float x_offset = 0.0;
        float y_offset = 0.0;

        for(int i=0; i <feat_size; i++)
            for(int j=0; j<feat_size; j++)
            {
                 float cur_conf = *(scoreMapOutput + i*feat_size + j);

                 if (cur_conf > max_conf)
                 {

                    max_conf = cur_conf;
      
                    idx_x = j;
                    idx_y = i;
                 }

            }


        w = *(sizeMapOutput + 0*feat_size*feat_size + idx_y*feat_size + idx_x);
        h = *(sizeMapOutput + 1*feat_size*feat_size + idx_y*feat_size + idx_x);

        x_offset = *(offsetMapOutput + 0*feat_size*feat_size + idx_y*feat_size + idx_x);
        y_offset = *(offsetMapOutput + 1*feat_size*feat_size + idx_y*feat_size + idx_x);


        // cout <<"max_conf: " << max_conf << endl;
        // cout <<"idx_x: " << idx_x << endl;
        // cout <<"idx_y: " << idx_y << endl;
        // cout <<"w: " << w << endl;
        // cout <<"h: " << h << endl;
        // cout <<"x_offset: " << x_offset << endl;
        // cout <<"y_offset: " << y_offset << endl;


        float cx = ((float)idx_x + x_offset) / feat_size;
        float cy = ((float)idx_y + y_offset) / feat_size;


        pred_bbox.x = cx;
        pred_bbox.y = cy;
        pred_bbox.width = w;
        pred_bbox.height = h;

        // cout <<"search_size: " << search_size << endl;
        // cout <<"resize_factor: " << resize_factor << endl;

        pred_bbox.x      = pred_bbox.x * search_size / resize_factor;
        pred_bbox.y      = pred_bbox.y * search_size / resize_factor;
        pred_bbox.width  = pred_bbox.width * search_size / resize_factor;
        pred_bbox.height = pred_bbox.height * search_size / resize_factor;

        // cout <<"pred_bbox: " << pred_bbox.x <<" " << pred_bbox.y <<" " << pred_bbox.width <<" " <<pred_bbox.height << endl;
        state = map_box_back(state, pred_bbox, resize_factor, search_size);

        rectangle(frame, state, Scalar(0, 255, 0), 5);
        // rectangle(frame, Point(state[0], state[1]), Point(state[0] + state[2], state[1] + state[3]), Scalar(0, 255, 0), 5);
        // cv::imwrite("res.jpg", frame);
        // break;
        out.write(frame);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        // std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

        std::cout << "\ncounts: " << counts++ << ". time: " << duration.count() << " ms"  << std::endl;

    }

    out.release();
    cap.release();

    // delete[] zInput;
    // delete[] xInput;
    // delete[] scoreMapOutput;
    // delete[] sizeMapOutput;
    // delete[] offsetMapOutput;

    cudaFree(d_zInput);
    cudaFree(d_xInput);
    cudaFree(d_scoreMap);
    cudaFree(d_sizeMap);
    cudaFree(d_offsetMap);

    context->destroy();
    engine->destroy();
    // shutdownProtobufLibrary();

    return 0;
}

// score:  0.2220985 index x:  7 7 0.24480644 0.2568841 -0.3531289 0.34548384
// search_size:  256
// resize_factor:  0.5614035087719298
// 256 0.5614035087719298 [189.43582606315613, 209.34628942608833, 111.63173604011536, 117.1391487121582]
// state:  [276, 119, 111, 117]

