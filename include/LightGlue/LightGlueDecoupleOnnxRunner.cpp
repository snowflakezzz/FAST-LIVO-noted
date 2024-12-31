#include "LightGlueDecoupleOnnxRunner.h"

namespace Lightglue{
LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(Configuation &config){
    scale[0] = 0.0; scale[1] = 0.0;
    cfg = config;
    if(cfg.lightglue_path.empty() || cfg.extractor_path.empty()){
        std::cout << "[ERROR] need lightglue and extractor weight path" << std::endl;
        return;
    }

    this->init_ortenv();
}

int LightGlueDecoupleOnnxRunner::init_ortenv(){
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try{
        env0 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");
        env1 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Matcher");

        session_options0 = Ort::SessionOptions();
        session_options0.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_options1 = Ort::SessionOptions();
        session_options1.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda")
        {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;     // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options0.AppendExecutionProvider_CUDA(cuda_options);
            session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options1.AppendExecutionProvider_CUDA(cuda_options);
            session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

        extract_session = std::make_unique<Ort::Session>(env0, cfg.extractor_path.c_str(), session_options0);
        matcher_session = std::make_unique<Ort::Session>(env1, cfg.lightglue_path.c_str(), session_options1);

        // Initial Extractor
        size_t numInputNodes = extract_session->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            ExtractorInputNodeNames.emplace_back(strdup(extract_session->GetInputNameAllocated(i, allocator).get()));
            ExtractorInputNodeShapes.emplace_back(extract_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        size_t numOutputNodes = extract_session->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            ExtractorOutputNodeNames.emplace_back(strdup(extract_session->GetOutputNameAllocated(i, allocator).get()));
            ExtractorOutputNodeShapes.emplace_back(extract_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numInputNodes = 0;
        numOutputNodes = 0;
        // Initial Matcher
        numInputNodes = matcher_session->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            MatcherInputNodeNames.emplace_back(strdup(matcher_session->GetInputNameAllocated(i, allocator).get()));
            MatcherInputNodeShapes.emplace_back(matcher_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        numOutputNodes = matcher_session->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            MatcherOutputNodeNames.emplace_back(strdup(matcher_session->GetOutputNameAllocated(i, allocator).get()));
            MatcherOutputNodeShapes.emplace_back(matcher_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> 
        LightGlueDecoupleOnnxRunner::InferenceImage(const cv::Mat &srcImage, const cv::Mat &destImage){
    if (srcImage.empty() || destImage.empty())
    {
        throw "[ERROR] ImageEmptyError ";
    }
    cv::Mat srcImage_copy(srcImage);
    cv::Mat destImage_copy(destImage);

    // step1 图像预处理
    cv::Mat src = PreProcess(srcImage_copy, scale[0]);
    cv::Mat dest = PreProcess(destImage_copy, scale[1]);

    // step2 特征提取
    extractor_outputtensors0 = extractor_inference(src);
    extractor_outputtensors1 = extractor_inference(dest);

    // step3 将tensor结果转为cv::Point2f并进行坐标归一化
    auto src_extract = postprocess(extractor_outputtensors0, src.cols, src.rows);
    auto dest_extract = postprocess(extractor_outputtensors1, dest.cols, dest.rows);

    // step4 特征匹配
    match_inference(src_extract.first, dest_extract.first, src_extract.second, dest_extract.second);

    // step5 将匹配的tensor结果转为cv::Point2f
    match_postprocess(src_extract.first, dest_extract.first);

    return keypoints_result;
}

void LightGlueDecoupleOnnxRunner::match_postprocess(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1){
    // 匹配情况
    std::vector<int64_t> matches_Shape = matcher_outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t *matches = (int64_t *)matcher_outputtensors[0].GetTensorMutableData<void>();
    
    // 匹配得分
    std::vector<int64_t> mscore_Shape = matcher_outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
    float *mscores = (float *)matcher_outputtensors[1].GetTensorMutableData<void>();

    std::vector<cv::Point2f> m_kpts0, m_kpts1;
    m_kpts0.reserve(matches_Shape[0]);
    m_kpts1.reserve(matches_Shape[0]);
    for (int i = 0; i < matches_Shape[0]; i++){
        if (mscores[i] > cfg.threshold){
            auto kpt0 = kpts0[matches[i * 2]];
            kpt0.x = (kpt0.x + 0.5) / scale[0] - 0.5;
            kpt0.y = (kpt0.y + 0.5) / scale[1] - 0.5;
            auto kpt1 = kpts1[matches[i * 2 + 1]];
            kpt1.x = (kpt1.x + 0.5) / scale[1] - 0.5;
            kpt1.y = (kpt1.y + 0.5) / scale[1] - 0.5;
            m_kpts0.emplace_back(kpt0);
            m_kpts1.emplace_back(kpt1);
        }
    }

    keypoints_result.first = m_kpts0;
    keypoints_result.second = m_kpts1;
}

void LightGlueDecoupleOnnxRunner::match_inference(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1){
    MatcherInputNodeShapes[0] = {1, static_cast<int>(kpts0.size()), 2};
    MatcherInputNodeShapes[1] = {1, static_cast<int>(kpts1.size()), 2};
    if (cfg.extractor_type == "superpoint")
    {
        MatcherInputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 256};
        MatcherInputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 256};
    }
    else
    {
        MatcherInputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 128};
        MatcherInputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 128};
    }

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    float *kpts0_data = new float[kpts0.size() * 2];
    float *kpts1_data = new float[kpts1.size() * 2];
    for (size_t i = 0; i < kpts0.size(); ++i)
    {
        kpts0_data[i * 2] = kpts0[i].x;
        kpts0_data[i * 2 + 1] = kpts0[i].y;
    }
    for (size_t i = 0; i < kpts1.size(); ++i)
    {
        kpts1_data[i * 2] = kpts1[i].x;
        kpts1_data[i * 2 + 1] = kpts1[i].y;
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, kpts0_data, kpts0.size() * 2 * sizeof(float),
        MatcherInputNodeShapes[0].data(), MatcherInputNodeShapes[0].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, kpts1_data, kpts1.size() * 2 * sizeof(float),
        MatcherInputNodeShapes[1].data(), MatcherInputNodeShapes[1].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc0, kpts0.size() * 256 * sizeof(float),
        MatcherInputNodeShapes[2].data(), MatcherInputNodeShapes[2].size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, desc1, kpts1.size() * 256 * sizeof(float),
        MatcherInputNodeShapes[3].data(), MatcherInputNodeShapes[3].size()));

    auto output_tensor = matcher_session->Run(Ort::RunOptions{nullptr}, MatcherInputNodeNames.data(), input_tensors.data(),
                                                input_tensors.size(), MatcherOutputNodeNames.data(), MatcherOutputNodeNames.size());

    for (auto &tensor : output_tensor)
    {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
            std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
        }
    }
    matcher_outputtensors = std::move(output_tensor);
}

std::pair<std::vector<cv::Point2f>, float*> LightGlueDecoupleOnnxRunner::postprocess(std::vector<Ort::Value> &tensor, int w, int h){
    std::pair<std::vector<cv::Point2f>, float *> extractor_result;

    std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t *kpts = (int64_t *)tensor[0].GetTensorMutableData<void>();

    // 提取分数
    std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
    float *scores = (float *)tensor[1].GetTensorMutableData<void>();

    std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
    float *desc = (float *)tensor[2].GetTensorMutableData<void>();

    // step2 特征点坐标归一化
    cv::Point2f shift(static_cast<float>(w)/2, static_cast<float>(h)/2);
    float scale = static_cast<float>(std::max(w, h))/2;

    std::vector<cv::Point2f> kpts_f;
    for (int i = 0; i < kpts_Shape[1] * 2; i += 2)
    {
        cv::Point2f kpt(kpts[i], kpts[i + 1]);
        cv::Point2f normalizedKpt = (kpt - shift) / scale;
        kpts_f.emplace_back(normalizedKpt);
    }

    extractor_result.first = kpts_f;
    extractor_result.second = desc;
    return extractor_result;
}

std::vector<Ort::Value> LightGlueDecoupleOnnxRunner::extractor_inference(cv::Mat &img){
    int srcInputTensorSize, destInputTensorSize;
    int height = img.rows; int width = img.cols;

    if (cfg.extractor_type == "superpoint")
        ExtractorInputNodeShapes[0] = {1, 1, height, width};
    else if (cfg.extractor_type == "disk")
        ExtractorInputNodeShapes[0] = {1, 3, height, width};
    
    srcInputTensorSize = ExtractorInputNodeShapes[0][0] * ExtractorInputNodeShapes[0][1] * ExtractorInputNodeShapes[0][2] * ExtractorInputNodeShapes[0][3];
    
    if(img.type() != CV_32F) img.convertTo(img, CV_32F);
    std::vector<float> srcInputTensorValues(srcInputTensorSize);
    if (cfg.extractor_type == "superpoint")
    {
        srcInputTensorValues.assign(img.begin<float>(), img.end<float>());
    }
    else
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Vec3f pixel = img.at<cv::Vec3f>(y, x); // RGB
                srcInputTensorValues[y * width + x] = pixel[2];
                srcInputTensorValues[height * width + y * width + x] = pixel[1];
                srcInputTensorValues[2 * height * width + y * width + x] = pixel[0];
            }
        }
    }

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                                              OrtMemType::OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(),
        ExtractorInputNodeShapes[0].data(), ExtractorInputNodeShapes[0].size()));

    auto output_tensor = extract_session->Run(Ort::RunOptions{nullptr}, ExtractorInputNodeNames.data(), input_tensors.data(),
                                                input_tensors.size(), ExtractorOutputNodeNames.data(), ExtractorOutputNodeNames.size());

    for (auto &tensor : output_tensor)
    {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
            std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
        }
    }

    return output_tensor;
}

cv::Mat LightGlueDecoupleOnnxRunner::PreProcess(cv::Mat &img, float &scale){
    int heigh = img.rows; int width = img.cols;

    // step1 图像缩放
    int h_new, w_new, size;
    size = cfg.image_size;
    if(size == 512 || size == 1024 || size == 2048){
        scale = static_cast<float>(size) / static_cast<float>(std::max(heigh, width));
        h_new = static_cast<int>(round(heigh * scale));
        w_new = static_cast<int>(round(width * scale));
    }
    else 
        throw std::invalid_argument("Incorrect new size: " + std::to_string(size));

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(w_new, h_new), 0, 0, cv::INTER_AREA);

    // step2 图像归一化
    cv::Mat normalized_img;
    if(resized_img.channels() == 3){
        cv::cvtColor(resized_img, normalized_img, cv::COLOR_BGR2RGB);
        normalized_img.convertTo(normalized_img, CV_32F, 1.0 / 255.0);
    }
    else if(resized_img.channels() == 1)
        resized_img.convertTo(normalized_img, CV_32F, 1.0 / 255.0);
    else
        throw std::invalid_argument("[ERROR] Not an image");

    cv::Mat result_img(normalized_img);
    if(cfg.extractor_type == "superpoint" && !cfg.isgray)
        cv::cvtColor(result_img, result_img, cv::COLOR_RGB2GRAY);
    
    return result_img;
}
}