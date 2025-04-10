import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import joblib 
from multiprocessing import freeze_support
# Thêm thư viện để hiển thị ảnh
import matplotlib.pyplot as plt
from PIL import Image
import math # Để dùng ceil tính số hàng subplot

# --- Định nghĩa lại hàm trích xuất đặc trưng ---
def extract_features(dataloader, model):
    features_list = []
    labels_list = []
    print(f"Bắt đầu trích xuất đặc trưng cho dự đoán...")
    model.eval() 
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            print(f".", end="")
    print("\nHoàn tất trích xuất đặc trưng.")
    features_np = np.concatenate(features_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    return features_np, labels_np

# ============================================
# === Phần thực thi chính cho việc Predict ===
# ============================================
if __name__ == '__main__':
    freeze_support()

    # --- Cấu hình ---
    data_dir = '.'
    test_dir = os.path.join(data_dir, 'test_set') 
    svm_model_path = 'svm_catdog_resnet18_features.joblib' 

    img_size = 224
    batch_size = 32 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 1. Tải Mô hình CNN Feature Extractor ---
    print("Đang tải mô hình ResNet pre-trained...")
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        feature_extractor = models.resnet18(weights=weights)
        num_ftrs = feature_extractor.fc.in_features
        feature_extractor.fc = nn.Identity() 
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval() 
        print(f"Mô hình feature extractor sẵn sàng. Output features: {num_ftrs}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình feature extractor: {e}")
        exit()

    # --- 2. Chuẩn bị Dữ liệu Test ---
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Đang tải dữ liệu từ: {test_dir}")
    try:
        image_dataset_test = datasets.ImageFolder(test_dir, data_transforms)
        dataloader_test = DataLoader(image_dataset_test, 
                                     batch_size=batch_size, 
                                     shuffle=False, # QUAN TRỌNG: shuffle=False để thứ tự khớp nhau
                                     num_workers=4)                      
        class_names = image_dataset_test.classes
        # Lấy danh sách đường dẫn ảnh và nhãn thật từ dataset
        # image_dataset_test.samples là list các tuple (đường_dẫn_ảnh, chỉ_số_lớp)
        image_paths_and_labels = image_dataset_test.samples 
        print(f"Các lớp tìm thấy trong tập test: {class_names}")
        if not class_names:
             print("Cảnh báo: Không tìm thấy lớp nào trong thư mục test_set.")
    except FileNotFoundError:
         print(f"Lỗi: Không tìm thấy thư mục dữ liệu test tại '{test_dir}'. Vui lòng kiểm tra đường dẫn.")
         exit()
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu test: {e}")
        exit()
        
    # --- 3. Tải Mô hình SVM đã huấn luyện ---
    print(f"Đang tải mô hình SVM đã huấn luyện từ: {svm_model_path}")
    try:
        loaded_svm_model = joblib.load(svm_model_path)
        print("Tải mô hình SVM thành công.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file mô hình SVM tại '{svm_model_path}'.")
        exit()
    except Exception as e:
        print(f"Lỗi khi tải mô hình SVM: {e}")
        exit()

    # --- 4. Trích xuất Đặc trưng cho Tập Test ---
    try:
        # y_test ở đây chứa chỉ số lớp thực tế, theo đúng thứ tự của dataloader (vì shuffle=False)
        X_test, y_test = extract_features(dataloader_test, feature_extractor) 
        print(f"Shape của đặc trưng tập test (X_test): {X_test.shape}")
        print(f"Shape của nhãn tập test (y_test): {y_test.shape}")
    except Exception as e:
        print(f"Lỗi trong quá trình trích xuất đặc trưng: {e}")
        exit()
        
    # --- 5. Thực hiện Dự đoán bằng SVM đã tải ---
    print("\nBắt đầu dự đoán trên tập test bằng mô hình SVM đã tải...")
    try:
        # y_pred chứa chỉ số lớp dự đoán, cùng thứ tự với y_test
        y_pred = loaded_svm_model.predict(X_test)
        print("Dự đoán hoàn tất.")
    except Exception as e:
        print(f"Lỗi khi thực hiện predict bằng SVM: {e}")
        exit()

    # --- 6. Đánh giá Kết quả Dự đoán ---
    # print("\nĐánh giá kết quả dự đoán:")
    # try:
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    #     print("\nClassification Report:")
    #     print(classification_report(y_test, y_pred, target_names=class_names)) 
    # except Exception as e:
    #     print(f"Lỗi khi đánh giá kết quả: {e}")
        
    # --- 7. Hiển thị một vài dự đoán mẫu ---
    print("\nHiển thị một vài dự đoán mẫu...")
    num_samples_to_show = 10 # Số lượng mẫu muốn hiển thị
    # Đảm bảo không hiển thị nhiều hơn số mẫu có trong tập test
    num_samples_to_show = min(num_samples_to_show, len(y_test)) 
    
    if num_samples_to_show > 0:
        # Tính toán số hàng và cột cho subplot
        num_cols = 5 
        num_rows = math.ceil(num_samples_to_show / num_cols)
        
        plt.figure(figsize=(num_cols * 3, num_rows * 3.5)) # Điều chỉnh kích thước figure
        j = 0
        while j!= 10:
            # Lấy đường dẫn ảnh từ list đã lấy ở bước 2
            i = np.random.randint(0,2000)
            img_path, _ = image_paths_and_labels[i] 
            # Lấy nhãn thật và nhãn dự đoán từ kết quả predict
            true_label_idx = y_test[i]
            predicted_label_idx = y_pred[i]
            true_label = class_names[true_label_idx]
            predicted_label = class_names[predicted_label_idx]

            # Xác định màu cho tiêu đề (đúng: xanh lá, sai: đỏ)
            title_color = 'green' if true_label_idx == predicted_label_idx else 'red'

            # Tải ảnh bằng PIL
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Lỗi khi mở ảnh {img_path}: {e}")
                continue # Bỏ qua ảnh này nếu không mở được

            # Hiển thị ảnh trên subplot
            plt.subplot(num_rows, num_cols, j + 1)
            plt.imshow(img)
            plt.title(f"True: {true_label}\nPredicted: {predicted_label}", color=title_color, fontsize=10)
            plt.axis('off') # Ẩn trục tọa độ
            j += 1
        plt.tight_layout() # Tự động điều chỉnh khoảng cách subplot
        plt.show() # Hiển thị figure chứa tất cả subplot
    else:
        print("Không có mẫu nào để hiển thị.")

    print("\nHoàn tất quá trình dự đoán!")