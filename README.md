# AIOps cho Microservices trên Kubernetes

Đây là đồ án tốt nghiệp AIOps được xây dựng dựa trên nền tảng `microservices-demo`.  
 **nền tảng AIOps** chạy trên **Minikube/Kubernetes** với các chức năng chính:

- phát hiện bất thường theo thời gian gần thực
- RCA (Root Cause Analysis - phân tích nguyên nhân gốc)
- thực thi hành động khôi phục trên Kubernetes
- vòng lặp phản hồi để đánh giá sự cố
- lựa chọn mô hình và nền tảng quản lý model registry cơ bản
- triển khai theo GitOps với **GitHub Actions + Argo CD**
- hỗ trợ giám sát nhiều hệ thống

## Mục tiêu dự án

Mục tiêu của dự án là xây dựng một quy trình AIOps đầu cuối cho hệ thống microservices cloud-native:

1. thu thập traces, metrics, logs và trạng thái sức khỏe Kubernetes
2. phát hiện bất thường khi hệ thống đang chạy
3. dự đoán top-k service có khả năng là nguyên nhân gốc
4. đề xuất hoặc thực thi hành động khôi phục
5. lưu trữ sự kiện giám sát và phản hồi từ người vận hành
6. hỗ trợ vòng đời mô hình và khả năng tái huấn luyện trong tương lai

## Kiến trúc hiện tại

Hệ thống hiện bao gồm các lớp chính sau:

### 1. Business microservices

- **Online Boutique**
- **Online Boutique Staging**:
- **Sock Shop**

### 2. Observability

- **Jaeger** dùng cho tracing
- **Prometheus** dùng cho metrics
- **Grafana** dùng cho dashboard giám sát
- `kubectl logs` dùng cho bước kiểm tra log ban đầu

### 3. Các dịch vụ AIOps

- **dashboard**
  - đăng nhập/xác thực
  - giao diện RBAC
  - audit logs
  - Live Analyze
  - RCA top-k
  - thao tác phản hồi
  - kiểm tra log
- **orchestrator**
  - điều phối luồng anomaly -> RCA -> recommendation
- **anomaly-service**
  - phục vụ các mô hình phát hiện bất thường
- **rca-service**
  - phục vụ các mô hình RCA

### 4. Triển khai và vận hành

- **Minikube** dùng làm môi trường Kubernetes cục bộ
- **GitHub Actions** dùng cho CI/CD
- **GHCR** dùng để lưu container images
- **Argo CD** dùng cho triển khai GitOps

## Các chức năng AIOps chính

### Dashboard và vận hành

- Live Analyze cho traces/metrics runtime
- xếp hạng RCA top-k service
- thực thi hành động khôi phục:
  - restart deployment
  - scale deployment
- kiểm tra log từ các service ứng viên RCA
- lịch sử sự kiện giám sát
- lịch sử audit log

### RBAC và xác thực

Các vai trò được hỗ trợ:

- `admin`
- `operator`
- `viewer`
- `ml_engineer`

Hành vi hiện tại:

- `admin` / `ml_engineer`
  - lựa chọn mô hình
  - hook cho luồng promote model
  - thao tác đánh giá incident
- `operator`
  - Live Analyze
  - thực thi hành động khôi phục
- `viewer`
  - chỉ có quyền xem dashboard

### Lưu trữ sự kiện giám sát

Mỗi lần chạy Live Analyze sẽ lưu một monitoring event, bao gồm:

- `system_id`
- kết quả anomaly
- kết quả RCA
- recommendation
- snapshot trace
- snapshot metrics
- thông tin mô hình
- trạng thái feedback

### Vòng lặp phản hồi

Dashboard hỗ trợ:

- `Accept Incident`
- `Reject False Positive`
- `Mark Unknown`

Các phản hồi này được lưu lại để phục vụ gán nhãn và tái huấn luyện trong tương lai.

### Kiểm tra log

Khi hệ thống phát hiện anomaly/RCA, dashboard có thể lấy log gần đây của các service ứng viên RCA bằng:

- `kubectl logs`

Giao diện sẽ làm nổi bật các từ khóa:

- `error`
- `warn`
- `exception`

Chức năng này giúp operator kiểm tra và loại bỏ các false positive có khả năng xảy ra.

## Giám sát nhiều hệ thống

Dự án hỗ trợ **system catalog** để dashboard có thể chuyển đổi giữa nhiều hệ thống được giám sát.

Catalog hiện tại gồm:

- `online-boutique`
- `online-boutique-staging`
- `sock_shop`: đã có metadata/catalog; phần triển khai runtime vẫn là bước phát triển tiếp theo

Mỗi hệ thống có thể định nghĩa:

- namespace
- entry services
- Jaeger services
- Prometheus labels
- model profile
- service catalog

### Trạng thái runtime hiện tại

Ở giai đoạn này, các runtime multi-system đã được triển khai và kiểm chứng gồm:

- `online-boutique`
- `online-boutique-staging`

`sock_shop` đã được biểu diễn trong system catalog, nhưng phần triển khai Kubernetes runtime đầy đủ vẫn chưa hoàn tất trong repository này.

## Mô hình

### Mô hình phát hiện bất thường

Artifact anomaly hiện tại:

- `anomaly_xgb_lgbm`

Ghi chú:

- sử dụng ensemble model
- hỗ trợ calibrated threshold
- đã tích hợp vào luồng Live Analyze trên dashboard

### Mô hình RCA

Nhóm mô hình RCA hiện tại gồm:

- `rf_ml_ranker`
- `gat_baseline`
- `hgnn_rca`

Hệ thống hỗ trợ lựa chọn mô hình trên dashboard và inference RCA runtime thông qua RCA service riêng.

## CI/CD và GitOps

Dự án sử dụng:

- **GitHub Actions** cho CI và build/push image có chọn lọc
- **Argo CD** cho đồng bộ triển khai

### Thiết kế CI/CD

Luồng CI/CD hiện tại được thiết kế như sau:

- nếu chỉ thay đổi code của **dashboard**, chỉ rebuild image dashboard
- nếu chỉ thay đổi **orchestrator**, chỉ rebuild orchestrator
- nếu chỉ thay đổi **anomaly-service**, chỉ rebuild anomaly-service
- nếu chỉ thay đổi **rca-service**, chỉ rebuild rca-service
- nếu thay đổi shared framework/pipeline/config, các service AIOps liên quan có thể được rebuild

Thiết kế này giúp quá trình triển khai hiệu quả hơn và gần với mô hình quản lý service ownership trong GitOps.

### GitOps

Việc triển khai được quản lý thông qua:

- `deploy/aiops/environments/dev`
- `deploy/argocd/applications/aiops-dev.yaml`

Argo CD theo dõi trạng thái Git và tự động đồng bộ AIOps stack.

## Cấu trúc repository

Các thư mục chính:

- [`aiops_framework/`](./aiops_framework)
  - logic của dashboard, orchestrator, anomaly-service, rca-service
- [`deploy/aiops/`](./deploy/aiops)
  - Kubernetes manifests cho AIOps
- [`deploy/argocd/`](./deploy/argocd)
  - các Argo CD applications
- [`deploy/systems/`](./deploy/systems)
  - các overlay triển khai runtime cho hệ thống được giám sát
- [`observability/`](./observability)
  - manifests cho Jaeger và các thành phần observability liên quan
- [`pipeline/`](./pipeline)
  - logic feature engineering và RCA data pipeline
- [`kubernetes-manifests/`](./kubernetes-manifests)
  - manifests của Online Boutique
- [`data_anomaly_balanced_v3/`](./data_anomaly_balanced_v3)
  - artifact cho anomaly
- [`data_rca_balanced_v3/`](./data_rca_balanced_v3)
  - artifact cho RCA

## Chạy nhanh trên Minikube

### 1. Khởi động Minikube

```powershell
minikube start --driver=docker --cpus=4 --memory=6144
kubectl get nodes
```

### 2. Triển khai Online Boutique

```powershell
kubectl apply -k kubernetes-manifests
```

### 3. Triển khai Jaeger

```powershell
kubectl apply -f observability\jaeger\jaeger.yaml
```

### 4. Cài đặt Prometheus và Grafana

```powershell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

kubectl create namespace monitoring

helm install prometheus prometheus-community/prometheus -n monitoring
helm install grafana grafana/grafana -n monitoring
```

### 5. Cài đặt Argo CD

```powershell
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl get pods -n argocd
```

### 6. Triển khai ứng dụng AIOps thông qua Argo CD

```powershell
kubectl apply -f deploy\argocd\applications\aiops-dev.yaml
kubectl get applications -n argocd
kubectl get pods -n aiops-dev
```

### 7. Tùy chọn: triển khai Online Boutique Staging runtime

```powershell
kubectl apply -k deploy\systems\online-boutique-staging
```

Hoặc triển khai thông qua Argo CD:

```powershell
kubectl apply -f deploy\argocd\applications\online-boutique-staging.yaml
```

## Truy cập hệ thống

### Argo CD

```powershell
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Mở trình duyệt:

- [https://127.0.0.1:8080](https://127.0.0.1:8080)

Lấy mật khẩu ban đầu:

```powershell
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | % { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
```

### AIOps dashboard

```powershell
kubectl port-forward svc/aiops-dashboard -n aiops-dev 8010:8010
```

Mở trình duyệt:

- [http://127.0.0.1:8010](http://127.0.0.1:8010)

Thông tin đăng nhập dev hiện tại:

- username: `admin`
- password: được cấu hình trong `deploy/aiops/environments/dev/dashboard-auth-secret.yaml`

### Jaeger

```powershell
kubectl port-forward svc/jaeger -n observability 16686:16686
```

Mở trình duyệt:

- [http://127.0.0.1:16686](http://127.0.0.1:16686)

### Grafana

```powershell
kubectl port-forward svc/grafana -n monitoring 3000:80
```

Mở trình duyệt:

- [http://127.0.0.1:3000](http://127.0.0.1:3000)

## Luồng demo đề xuất

1. mở dashboard
2. chọn hệ thống từ dropdown
3. chạy Live Analyze
4. kiểm tra anomaly score và RCA top-k
5. mở log của các service ứng viên
6. chọn accept/reject/mark unknown
7. tùy chọn thực thi hành động khôi phục

## Hạn chế hiện tại

- runtime deployment của `sock_shop` chưa hoàn thiện đầy đủ
- chất lượng live metrics phụ thuộc vào trạng thái hoạt động của Prometheus
- Live RCA dựa trên Jaeger vẫn phụ thuộc vào chất lượng trace và độ phủ service
- khả năng cô lập observability giữa các hệ thống staging/multi-system vẫn đang được cải thiện
- một số chức năng model lifecycle vẫn còn cơ bản so với các nền tảng MLOps hoàn chỉnh

## Hướng phát triển tiếp theo

- hoàn thiện runtime deployment cho `sock-shop`
- mở rộng vòng đời model registry:
  - candidate
  - production
  - archived
- bổ sung tái huấn luyện từ monitoring event feedback
- thay thế `kubectl logs` bằng các backend log có thể mở rộng như Loki
- tăng cường cô lập hệ thống cho multi-system observability
- cải thiện kiểm tra rollout health và tinh chỉnh tài nguyên trên Minikube

## Ghi chú

Repository này không còn chỉ là bản upstream `microservices-demo`.  
Dự án đã được tùy chỉnh thành một **đồ án tốt nghiệp AIOps** tập trung vào:

- phát hiện bất thường runtime
- RCA cho microservices
- xử lý incident có operator trong vòng lặp
- triển khai GitOps
- giám sát nhiều hệ thống trên Kubernetes
