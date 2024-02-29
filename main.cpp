#include <iostream>
#include <opencv2/opencv.hpp>
// #include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
// using namespace onnxruntime;

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



cv::Rect map_box_back(const cv::Rect& state, const cv::Rect2f& pred_box, const float resize_factor)
{
        int cx_prev = state.x + 0.5 * state.width;
        int cy_prev = state.y + 0.5 * state.height;

        int search_size = 256;

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
    string model_path = "sim_cnn_track.onnx";

    // vector<int> init_bbox = {316, 138, 110, 118};
    int feat_sz = 16;
    int search_size = 256;

    int fmp_size = 16;

    VideoCapture cap(videofilepath);
    VideoWriter out("output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 20.0, Size(480, 360));

    Mat frame;
    cap.read(frame);
    int H = frame.rows;
    int W = frame.cols;

    Mat z_patch_arr, x_patch_arr, x_patch_arr_cp, z_amask_arr, x_amask_arr;

    bool cudaEnabled{};

    cv::dnn::Net net = cv::dnn::readNetFromONNX("../sim_cnn_track.onnx");
    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    cv::Rect init_box = {316, 138, 110, 118};
    cv::Mat z_template, x_search;

    int template_factor = 2;
    int template_size = 128;

    float resize_factor = sample_target(frame, z_template, init_box, template_factor, template_size);

    cv::imwrite("test_tmeplate.jpg", z_template);

    cv::Rect state = init_box;

    cv::Rect2f pred_bbox; // 预测框

    while (true) {

        cap.read(frame);
        if (frame.empty()) break;

    //     Mat img_255 = frame.clone();

    //     // Sample target patch
    //     

    //     // Prepare search patch
    //     sample_target(frame, init_bbox, 4, 256, x_patch_arr, resize_factor, x_amask_arr);
        float resize_factor = sample_target(frame, x_search, state, 4, 256);

        cv::Mat blob_z;
        cv::dnn::blobFromImage(z_template, blob_z, 1.0/255.0, cv::Size(128, 128), cv::Scalar(), true, false);

        cv::Mat blob_x;
        cv::dnn::blobFromImage(x_search, blob_x, 1.0/255.0, cv::Size(256, 256), cv::Scalar(), true, false);

        net.setInput(blob_z, "z"); // "input1" 是网络中定义的输入层名称
        net.setInput(blob_x, "x"); // "input2" 是网络中定义的输入层名称

        // std::vector<cv::Mat> outputs;
        // net.forward(outputs, net.getUnconnectedOutLayersNames());
        cv::Mat score_map = net.forward("577"); // "output1" 是网络中定义的输出层名称
        cv::Mat size_map = net.forward("579"); // "output2" 是网络中定义的输出层名称
        cv::Mat offset_map = net.forward("575"); // "output2" 是网络中定义的输出层名称

        // cv::Mat score_map = outputs[0];
        // cv::Mat size_map = outputs[1];
        // cv::Mat offset_map = outputs[2];

        float* score_ptr  =   (float *)score_map.data;
        float* size_ptr   =   (float *)size_map.data;
        float* offset_ptr =   (float *)offset_map.data;

        // cout <<"cols: " << score_map.cols << endl;
        // cout <<"rows: " << score_map.rows << endl;
        // cout <<"size: " << score_map.size() << endl;
        int idx_x=0;
        int idx_y=0;

        float max_conf = 0;
        float w = 0.;
        float h = 0.;
        float x_offset = 0.0;
        float y_offset = 0.0;

        for(int i=0; i <16; i++)
            for(int j=0; j<16; j++)
            {
                 float cur_conf = *(score_ptr + i*16 + j);
                 if (cur_conf > max_conf)
                 {
                    max_conf = cur_conf;
                    // cout <<"max_conf: " << max_conf << endl;
                    idx_x = j;
                    idx_y = i;
                 }

            }
        

        w = *(size_ptr + 0*16*16 + idx_y*16 + idx_x);
        h = *(size_ptr + 1*16*16 + idx_y*16 + idx_x);

        x_offset = *(offset_ptr + 0*16*16 + idx_y*16 + idx_x);
        y_offset = *(offset_ptr + 1*16*16 + idx_y*16 + idx_x);

        // cout <<"max_conf: " << max_conf << endl;
        // cout <<"idx_x: " << idx_x << endl;
        // cout <<"idx_y: " << idx_y << endl;
        // // cout <<"cx: " << cx << endl;
        // // cout <<"cy: " << cy << endl;
        // cout <<"w: " << w << endl;
        // cout <<"h: " << h << endl;
        // cout <<"x_offset: " << x_offset << endl;
        // cout <<"y_offset: " << y_offset << endl;



        float cx = ((float)idx_x + x_offset) / fmp_size;
        float cy = ((float)idx_y + y_offset) / fmp_size;


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

        state = map_box_back(state, pred_bbox, resize_factor);

        

        // cout <<"resize_factor: " << resize_factor << endl;
        // cout <<"state: " << state.x <<" " << state.y <<" " << state.width <<" " <<state.height << endl;
        // cout <<"------------------ "  << endl;

        rectangle(frame, state, Scalar(0, 255, 0), 5);
        // rectangle(frame, Point(state[0], state[1]), Point(state[0] + state[2], state[1] + state[3]), Scalar(0, 255, 0), 5);

        out.write(frame);
    }

    out.release();
    cap.release();

    return 0;
}

// score:  0.2220985 index x:  7 7 0.24480644 0.2568841 -0.3531289 0.34548384
// search_size:  256
// resize_factor:  0.5614035087719298
// 256 0.5614035087719298 [189.43582606315613, 209.34628942608833, 111.63173604011536, 117.1391487121582]
// state:  [276, 119, 111, 117]

