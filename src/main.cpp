#include <ctime>
#include <filesystem>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

int main() {
  cv::CascadeClassifier face_cascade;
  // Especifique o caminho completo para o arquivo XML
  std::string face_cascade_path =
      "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

  if (!face_cascade.load(face_cascade_path)) {
    std::cerr << "Erro: Não foi possível carregar o classificador de rostos."
              << std::endl;
    return -1;
  }

  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Erro: Não foi possível abrir a câmera." << std::endl;
    return -1;
  }

  int face_id = 0;
  std::time_t last_time = std::time(0);

  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
      std::cerr << "Erro: Não foi possível ler o frame." << std::endl;
      break;
    }

    cv::flip(frame, frame, 1);
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

    for (const auto &face : faces) {
      cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);

      cv::Mat face_roi = frame(face);

      if (std::difftime(std::time(0), last_time) >= 1.5) {
        last_time = std::time(0);
        face_id++;
        std::string file_path = "imgs/face_" + std::to_string(face_id) + ".jpg";
        cv::imwrite(file_path, face_roi);
        std::cout << "Rosto salvo como " << file_path << std::endl;
      }
    }

    cv::imshow("Camera", frame);

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
