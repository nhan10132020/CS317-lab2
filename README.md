# Wine Classification 

## Mô tả
Dự án này xây dựng một API sử dụng FastAPI để phân loại rượu vang dựa trên mô hình học máy đã huấn luyện. Mô hình và scaler đã được train trong lab 1 và lưu dưới dạng file `.pkl` và `.joblib` trong `src/model/`.

## Pull Docker image from docker hub
Docker hub: https://hub.docker.com/r/nhan0612/cs317/tags

```sh
docker pull nhan0612/cs317:latest
```

## Video demo
- Deploy API lên server được cung cấp bằng cách pull docker image đã được build trên Docker hub
- Chạy container trên server trường
- Call request trong server trường và call request từ local -> remote server

https://github.com/user-attachments/assets/ece3fb1c-398c-4357-bac1-e357eee29bcd

---

## Cài đặt môi trường

### 1. Clone repository

```sh
git clone <repository-url>
cd <tên-thư-mục-dự-án>
```

### 2. Tạo virtual environment 
```sh
python3 -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

### 3. Cài đặt các thư viện phụ thuộc
**Lưu ý:** Tất cả các thư viện đều được khai báo cụ thể phiên bản trong file [`requirements.txt`](requirements.txt).  

```sh
pip install -r requirements.txt
```

---

## Chạy ứng dụng

### 0. Training với mlflows (Optional)

```sh
python train.py
```

### 1. Chạy trực tiếp bằng FastAPI/Uvicorn

```sh
uvicorn src.main:app --reload
```
API sẽ chạy tại địa chỉ: http://127.0.0.1:8000 

### 2. Chạy bằng Docker

#### Build image:
```sh
docker build -t wine-api .
```

#### Run container:
```sh
docker run -p 8000:8000 wine-api
```

### 3. Chạy bằng Docker Compose

```sh
docker-compose up -d
```

---

## Cấu trúc thư mục

```
.
├── src/
│   ├── main.py
│   └── model/
│       ├── scaler.joblib
│       └── wine_classification.pkl
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── mlruns/
```

---
