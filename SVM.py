import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import time
from multiprocessing import freeze_support # <<-- Thêm import này
import joblib
# --- Định nghĩa các hàm (ví dụ: extract_features) ---
def extract_features(dataloader, model):
    features_list = []
    labels_list = []
    print(f"Bắt đầu trích xuất đặc trưng...")
    with torch.no_grad():
        for inputs, labels in dataloader: 
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            print(f".", end="")
    print("\nHoàn tất trích xuất.")
    features_np = np.concatenate(features_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    print(features_np.shape, labels_np.shape)
    return features_np, labels_np



if __name__ == '__main__':
    # Gọi hàm này đầu tiên trên Windows khi dùng multiprocessing
    freeze_support()

    # --- Cấu hình ---
    data_dir = '.'
    train_dir = os.path.join(data_dir, 'training_set')
    test_dir = os.path.join(data_dir, 'test_set')
    img_size = 224
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- 1. Tải Mô hình CNN Pre-trained ---
    print("Đang tải mô hình ResNet pre-trained...")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    feature_extractor = models.resnet18(weights=weights)
    num_ftrs = feature_extractor.fc.in_features
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    print(f"Mô hình feature extractor sẵn sàng. Output features: {num_ftrs}")

    # --- 2. Chuẩn bị Dữ liệu ---
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("Đang tải datasets...")
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms),
        'test': datasets.ImageFolder(test_dir, data_transforms)
    }
    # Tạo DataLoaders với num_workers > 0
    dataloaders = {
        # <<-- Đảm bảo dòng này nằm trong if __name__ == '__main__'
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
        for x in ['train', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Các lớp: {class_names}")

    # --- 3 & 4. Trích xuất Đặc trưng ---
    # <<-- Các lời gọi hàm này phải nằm trong if __name__ == '__main__'
    X_train, y_train = extract_features(dataloaders['train'], feature_extractor)
    X_test, y_test = extract_features(dataloaders['test'], feature_extractor)

    # --- 5. Huấn luyện SVM ---
    # <<-- Phần này phải nằm trong if __name__ == '__main__'
    print("\nBắt đầu huấn luyện SVM...")
    start_time = time.time()
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    print(f"Huấn luyện SVM hoàn tất trong {time.time() - start_time:.2f} giây.")

    # --- 6. Đánh giá SVM ---
    # <<-- Phần này phải nằm trong if __name__ == '__main__'
    print("\nĐánh giá mô hình SVM trên tập test...")
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    # svm_model_path = 'svm_catdog_resnet18_features.joblib'
    # joblib.dump(svm_model, svm_model_path)
    # print(f"Mô hình SVM đã được lưu tại: {svm_model_path}")
    #Để tải lại model: loaded_svm = joblib.load(svm_model_path)
    print("\nHoàn tất!")