// ═════════════════════════════════════════════════════════════════════════════
//  Violence Detection — Raspberry Pi Cluster
//  Master (PC): loads video, extracts 16 frames exactly as training did,
//               sends the FULL 16-frame tensor to ALL 4 Pis simultaneously.
//  Workers (Pi 1-4): each run full inference independently on the same input.
//  Decision: if ANY Pi exceeds threshold (0.65) → VIOLENCE DETECTED.
//
//  Why full 16 frames to every Pi?
//  The model was trained on 16-frame sequences. Sending sub-clips (5 frames)
//  breaks the temporal patterns the model learned, causing wrong predictions.
//  Each Pi does independent inference → averaging reduces noise.
// ═════════════════════════════════════════════════════════════════════════════

#include <mpi.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Constants — must match training config exactly
// ─────────────────────────────────────────────────────────────────────────────
const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_STD[3]  = {0.229f, 0.224f, 0.225f};
const int   NUM_FRAMES       = 16;    // from config.py: NUM_FRAMES = 16
const int   FRAME_H          = 224;   // from config.py: FRAME_SIZE = (224,224)
const int   FRAME_W          = 224;
const int   NUM_WORKERS      = 4;

// MPI tags
const int TAG_TENSOR_NDIM  = 10;
const int TAG_TENSOR_SHAPE = 11;
const int TAG_TENSOR_DATA  = 12;
const int TAG_RESULT_PROB  = 20;
const int TAG_RESULT_UNC   = 21;
const int TAG_DONE         = 99;

// ─────────────────────────────────────────────────────────────────────────────
// Load exactly NUM_FRAMES from a video using np.linspace logic
// Matches training code: frame_idxs = np.linspace(0, total-1, 16).astype(int)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor load_video_frames(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "[Master] ERROR: Cannot open: " << path << "\n";
        return torch::Tensor();
    }

    int    total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps   = cap.get(cv::CAP_PROP_FPS);
    double dur   = (fps > 0) ? total / fps : 0.0;

    std::cout << "[Master] Video : " << fs::path(path).filename().string() << "\n"
              << "         Frames: " << total
              << "  FPS: " << std::fixed << std::setprecision(1) << fps
              << "  Duration: " << dur << "s\n";

    if (total <= 0) { cap.release(); return torch::Tensor(); }

    // np.linspace(0, total-1, NUM_FRAMES).astype(int) — exact training match
    std::vector<int> idxs(NUM_FRAMES);
    int last = std::max(total - 1, 0);
    for (int i = 0; i < NUM_FRAMES; i++)
        idxs[i] = static_cast<int>(std::round(i * last / (NUM_FRAMES - 1.0)));

    auto tensor = torch::zeros({1, NUM_FRAMES, 3, FRAME_H, FRAME_W});
    cv::Mat last_frame(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(0));

    for (int i = 0; i < NUM_FRAMES; i++) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idxs[i]);
        cv::Mat frame;
        bool ok = cap.read(frame) && !frame.empty();
        if (!ok) {
            frame = last_frame.clone();
        } else {
            // Match training: grayscale → RGB, then resize, then normalize
            if (frame.channels() == 1)
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2RGB);
            else
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            cv::resize(frame, frame, cv::Size(FRAME_W, FRAME_H));
            last_frame = frame.clone();
        }

        // Normalize: (pixel/255 - mean) / std  — matches training
        for (int h = 0; h < FRAME_H; h++)
            for (int w = 0; w < FRAME_W; w++) {
                auto px = frame.at<cv::Vec3b>(h, w);
                for (int c = 0; c < 3; c++)
                    tensor[0][i][c][h][w] =
                        (px[c] / 255.0f - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
    }

    cap.release();
    return tensor;  // [1, 16, 3, 224, 224]
}

// ─────────────────────────────────────────────────────────────────────────────
// MPI helpers
// ─────────────────────────────────────────────────────────────────────────────
void mpi_send_tensor(const torch::Tensor& t, int dest) {
    auto ct  = t.contiguous();
    int ndim = ct.dim();
    MPI_Send(&ndim, 1, MPI_INT, dest, TAG_TENSOR_NDIM, MPI_COMM_WORLD);
    std::vector<long> shape(ndim);
    for (int i = 0; i < ndim; i++) shape[i] = ct.size(i);
    MPI_Send(shape.data(), ndim, MPI_LONG, dest, TAG_TENSOR_SHAPE, MPI_COMM_WORLD);
    int numel = static_cast<int>(ct.numel());
    MPI_Send(ct.data_ptr<float>(), numel, MPI_FLOAT, dest, TAG_TENSOR_DATA, MPI_COMM_WORLD);
}

torch::Tensor mpi_recv_tensor(int src) {
    MPI_Status st;
    int ndim;
    MPI_Recv(&ndim, 1, MPI_INT, src, TAG_TENSOR_NDIM, MPI_COMM_WORLD, &st);
    std::vector<long> shape(ndim);
    MPI_Recv(shape.data(), ndim, MPI_LONG, src, TAG_TENSOR_SHAPE, MPI_COMM_WORLD, &st);
    int numel = 1;
    for (long s : shape) numel *= static_cast<int>(s);
    auto tensor = torch::zeros(std::vector<int64_t>(shape.begin(), shape.end()));
    MPI_Recv(tensor.data_ptr<float>(), numel, MPI_FLOAT, src, TAG_TENSOR_DATA, MPI_COMM_WORLD, &st);
    return tensor;
}

// ─────────────────────────────────────────────────────────────────────────────
// MASTER — Rank 0 (your PC)
// Sends the SAME full 16-frame tensor to ALL 4 Pis
// Averages their probabilities for final decision
// ─────────────────────────────────────────────────────────────────────────────
void run_master(const std::string& video_path, float threshold) {

    std::cout << "\n========================================\n"
              << "  Violence Detection — Pi Cluster\n"
              << "========================================\n"
              << "  Workers   : " << NUM_WORKERS << " Raspberry Pis\n"
              << "  Frames    : " << NUM_FRAMES  << " (full sequence per Pi)\n"
              << "  Threshold : " << threshold   << "\n"
              << "  Decision  : ANY Pi prob > threshold → VIOLENCE\n"
              << "========================================\n\n";

    // Load full 16-frame tensor exactly as training did
    torch::Tensor input = load_video_frames(video_path);
    if (!input.defined()) {
        std::cerr << "[Master] Failed to load video. Aborting.\n";
        int done = 1;
        for (int w = 1; w <= NUM_WORKERS; w++)
            MPI_Send(&done, 1, MPI_INT, w, TAG_DONE, MPI_COMM_WORLD);
        return;
    }

    // Send the SAME [1, 16, 3, 224, 224] tensor to all 4 Pis
    int kb = static_cast<int>(input.numel() * 4 / 1024);
    std::cout << "[Master] Sending full 16-frame tensor to all Pis (" << kb << " KB each)...\n";
    for (int w = 1; w <= NUM_WORKERS; w++) {
        std::cout << "  => Pi " << w << "\n";
        mpi_send_tensor(input, w);
    }

    // Collect results and average
    float sum_prob = 0.0f, sum_unc = 0.0f, max_prob = 0.0f;

    std::cout << "\n[Master] Results from Raspberry Pis:\n";
    for (int w = 0; w < NUM_WORKERS; w++) {
        int worker_rank = w + 1;
        MPI_Status st;
        float prob, unc;
        MPI_Recv(&prob, 1, MPI_FLOAT, worker_rank, TAG_RESULT_PROB, MPI_COMM_WORLD, &st);
        MPI_Recv(&unc,  1, MPI_FLOAT, worker_rank, TAG_RESULT_UNC,  MPI_COMM_WORLD, &st);

        sum_prob += prob;
        sum_unc  += unc;
        if (prob > max_prob) max_prob = prob;

        std::cout << "  Pi " << worker_rank
                  << "  prob=" << std::fixed << std::setprecision(4) << prob
                  << "  unc="  << unc
                  << "  =>  "  << (prob > threshold ? "VIOLENCE   " : "NonViolence") << "\n";
    }

    float avg_prob = sum_prob / NUM_WORKERS;
    float avg_unc  = sum_unc  / NUM_WORKERS;
    // ANY single worker exceeding threshold is enough to flag violence
    bool  violence = (max_prob > threshold);

    std::cout << "\n========================================\n"
              << "  FINAL RESULT\n"
              << "========================================\n"
              << "  avg_prob  : " << std::fixed << std::setprecision(4) << avg_prob << "\n"
              << "  max_prob  : " << max_prob   << "\n"
              << "  avg_unc   : " << avg_unc    << "\n"
              << "  threshold : " << threshold  << "\n"
              << "  Decision  : ANY worker prob > threshold triggers alert\n"
              << "  DECISION  : "
              << (violence ? "*** VIOLENCE DETECTED ***" : "    NonViolence          ")
              << "\n========================================\n";

    int done = 1;
    for (int w = 1; w <= NUM_WORKERS; w++)
        MPI_Send(&done, 1, MPI_INT, w, TAG_DONE, MPI_COMM_WORLD);
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER — Ranks 1-4 (Raspberry Pis)
// Receives full [1, 16, 3, 224, 224] tensor, runs inference, returns result
// ─────────────────────────────────────────────────────────────────────────────
void run_worker(int rank, torch::jit::script::Module& model, float threshold) {
    std::cout << "[Pi " << rank << "] Ready\n";

    while (true) {
        MPI_Status probe_st;
        int flag = 0;

        MPI_Iprobe(0, TAG_DONE, MPI_COMM_WORLD, &flag, &probe_st);
        if (flag) {
            int done;
            MPI_Recv(&done, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD, &probe_st);
            std::cout << "[Pi " << rank << "] Shutting down\n";
            break;
        }

        MPI_Iprobe(0, TAG_TENSOR_NDIM, MPI_COMM_WORLD, &flag, &probe_st);
        if (!flag) {
            struct timespec ts = {0, 1000000};
            nanosleep(&ts, nullptr);
            continue;
        }

        // Receive [1, 16, 3, 224, 224] — same shape the model was trained on
        torch::Tensor input = mpi_recv_tensor(0);

        float prob = 0.5f, unc = 0.5f;
        try {
            torch::NoGradGuard no_grad;
            auto output = model.forward({input}).toTuple();
            float logit = output->elements()[0].toTensor().item<float>();
            unc         = output->elements()[1].toTensor().item<float>();
            prob        = 1.0f / (1.0f + std::exp(-logit));
        } catch (const c10::Error& e) {
            std::cerr << "[Pi " << rank << "] Inference error: " << e.what() << "\n";
        }

        std::cout << "[Pi " << rank << "]"
                  << "  prob=" << std::fixed << std::setprecision(4) << prob
                  << "  unc=" << unc
                  << "  =>  " << (prob > threshold ? "VIOLENCE" : "NonViolence") << "\n";

        MPI_Send(&prob, 1, MPI_FLOAT, 0, TAG_RESULT_PROB, MPI_COMM_WORLD);
        MPI_Send(&unc,  1, MPI_FLOAT, 0, TAG_RESULT_UNC,  MPI_COMM_WORLD);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_WORKERS + 1) {
        if (rank == 0)
            std::cerr << "ERROR: Need exactly " << NUM_WORKERS + 1
                      << " processes (1 master PC + 4 Raspberry Pis).\n";
        MPI_Finalize();
        return 1;
    }

    if (argc < 3) {
        if (rank == 0)
            std::cerr << "Usage: mpiexec -n 5 ./violence_mpi_master <model.pt> <video.mp4> [threshold]\n";
        MPI_Finalize();
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    float       threshold  = (argc >= 4) ? std::stof(argv[3]) : 0.65f;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path, torch::kCPU);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[Rank " << rank << "] Model load error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        run_master(video_path, threshold);
    else
        run_worker(rank, model, threshold);

    MPI_Finalize();
    return 0;
}
