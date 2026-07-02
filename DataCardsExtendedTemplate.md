# Dataset Overview

## Tên bộ dữ liệu

**Bộ dữ liệu Speech Anti-Spoofing tiếng Việt (xây dựng dựa trên VSASV Dataset)**

---

## Phiên bản

**v1.0**

---

## Loại bộ dữ liệu

Đây là **bộ dữ liệu dẫn xuất (Derived Dataset)** được xây dựng dựa trên **VSASV Dataset**. Bộ dữ liệu đã được tiền xử lý và chuẩn hóa nhằm phục vụ bài toán **phát hiện giọng nói giả mạo (Speech Anti-Spoofing)** theo mô hình phân loại nhị phân.

---

## Bộ dữ liệu gốc

Bộ dữ liệu này được xây dựng từ **VSASV Dataset**, một bộ dữ liệu tiếng Việt dành cho bài toán **Spoofing-Aware Speaker Verification (SASV)** do **HUST Excellent Students in Pattern Recognition Laboratory (HUST EP Lab)** phát triển và công bố trong bài báo:

> **VSASV: A Vietnamese Dataset for Spoofing-Aware Speaker Verification (INTERSPEECH 2024)**

VSASV bao gồm các mẫu giọng nói thật (Bonafide) và giọng nói giả mạo (Spoof) được thu thập từ nhiều người nói tiếng Việt, nhằm hỗ trợ nghiên cứu về xác thực người nói trong điều kiện tồn tại các hình thức giả mạo giọng nói.

---

## Tổng quan

Bộ dữ liệu này là phiên bản đã được xử lý từ VSASV Dataset nhằm phục vụ riêng cho bài toán **Speech Anti-Spoofing**. Mục tiêu của bộ dữ liệu là xây dựng mô hình có khả năng phân biệt giữa:

- **Bonafide:** giọng nói thật do con người phát âm.
- **Spoof:** giọng nói được tạo hoặc chỉnh sửa bằng các kỹ thuật giả mạo.

Khác với VSASV gốc được thiết kế cho bài toán **Spoofing-Aware Speaker Verification (SASV)**, bộ dữ liệu này chỉ tập trung vào **phát hiện giọng nói giả mạo** dưới dạng bài toán phân loại nhị phân.

Trong quá trình xây dựng, dữ liệu được giữ nguyên nhãn gốc nhưng được chuẩn hóa thông qua quy trình tiền xử lý để phù hợp với việc huấn luyện các mô hình học sâu. Toàn bộ quy trình xử lý dữ liệu sẽ được trình bày chi tiết trong mục **Data Preprocessing**.

---

## Bài toán hướng tới

Bộ dữ liệu được sử dụng cho bài toán:

**Speech Anti-Spoofing (Phân loại nhị phân)**

Với đầu vào là một đoạn âm thanh tiếng nói, mô hình sẽ dự đoán một trong hai nhãn:

- **Bonafide:** Âm thanh được phát ra trực tiếp từ người nói.
- **Spoof:** Âm thanh được tạo ra hoặc chỉnh sửa bằng các kỹ thuật giả mạo giọng nói.

---

## Ngôn ngữ

- Tiếng Việt

---

## Loại dữ liệu

- Dữ liệu âm thanh (Speech Audio)

---

## Lĩnh vực nghiên cứu

Bộ dữ liệu hướng đến các lĩnh vực nghiên cứu sau:

- Speech Anti-Spoofing
- Automatic Speaker Verification (ASV)
- Spoofing-Aware Speaker Verification (SASV)
- Speech Processing
- Deep Learning for Audio
- AI-generated Speech Detection

---

## Đối tượng sử dụng

Bộ dữ liệu phù hợp cho:

- Sinh viên và học viên nghiên cứu về xử lý tiếng nói.
- Nhà nghiên cứu trong lĩnh vực học máy và trí tuệ nhân tạo.
- Nhà phát triển các hệ thống xác thực người nói.
- Các nghiên cứu về phát hiện giọng nói giả mạo tiếng Việt.
- Đánh giá và so sánh hiệu năng các mô hình Speech Anti-Spoofing.

---

## Phạm vi sử dụng

Bộ dữ liệu được xây dựng nhằm phục vụ mục đích:

- Huấn luyện mô hình Speech Anti-Spoofing.
- Đánh giá hiệu năng các mô hình phát hiện giọng nói giả mạo.
- So sánh các phương pháp trích xuất đặc trưng và kiến trúc học sâu.
- Nghiên cứu về bảo mật hệ thống xác thực người nói.

Bộ dữ liệu không nhằm thay thế VSASV Dataset gốc mà đóng vai trò là phiên bản đã được tiền xử lý để phục vụ quá trình thực nghiệm trong dự án này.

---

## Giấy phép sử dụng

Bộ dữ liệu kế thừa các điều khoản sử dụng của **VSASV Dataset**. Người sử dụng cần tuân thủ giấy phép và các quy định do nhóm tác giả của bộ dữ liệu gốc công bố.

---

## Yêu cầu trích dẫn

Nếu sử dụng bộ dữ liệu này trong nghiên cứu hoặc phát triển hệ thống, cần trích dẫn bài báo giới thiệu **VSASV Dataset** cùng với kho mã nguồn của dự án (nếu có).

---

## Trạng thái

Đây là phiên bản đầu tiên của bộ dữ liệu đã được tiền xử lý phục vụ cho bài toán **Speech Anti-Spoofing**. Các quy trình tiền xử lý, chia tập dữ liệu, trích xuất đặc trưng và cấu hình thực nghiệm sẽ được mô tả trong các phần tiếp theo của Data Card.

# Dataset Composition

## Thành phần bộ dữ liệu

Bộ dữ liệu được xây dựng dựa trên **VSASV Dataset**, một bộ dữ liệu tiếng Việt được thiết kế cho bài toán **Spoofing-Aware Speaker Verification (SASV)**. Dữ liệu bao gồm các đoạn ghi âm giọng nói thật và giọng nói giả mạo, được tổ chức nhằm hỗ trợ nghiên cứu về phát hiện giả mạo giọng nói và xác thực người nói an toàn.

Trong dự án này, chỉ các tệp âm thanh và nhãn phục vụ cho bài toán **Speech Anti-Spoofing** được sử dụng. Các thông tin liên quan đến định danh người nói hoặc bài toán xác thực người nói không được khai thác trong quá trình huấn luyện mô hình.

---

## Quy mô dữ liệu

Theo công bố của bộ dữ liệu VSASV:

| Thành phần | Giá trị |
|------------|---------:|
| Tổng số người nói | 1.382 |
| Số mẫu Bonafide | Khoảng 164.000 |
| Số mẫu Spoof | Khoảng 174.000 |
| Tổng số mẫu âm thanh | Khoảng 338.000 |
| Ngôn ngữ | Tiếng Việt |

Bộ dữ liệu bao phủ nhiều người nói với các đặc điểm giọng nói khác nhau nhằm tăng tính đa dạng và khả năng tổng quát hóa của các mô hình học máy.

> **Lưu ý:** Các thống kê trên được lấy từ bộ dữ liệu VSASV gốc. Các thống kê sau khi tiền xử lý sẽ được trình bày trong các mục **Data Preprocessing** và **Data Splits**.

---

## Đơn vị dữ liệu

Đơn vị cơ bản của bộ dữ liệu là **một tệp âm thanh (audio file)**.

Mỗi tệp âm thanh tương ứng với một phát ngôn (utterance) và được gán một nhãn duy nhất, biểu thị loại của mẫu dữ liệu:

- **Bonafide**: phát ngôn được tạo ra trực tiếp bởi người nói.
- **Spoof**: phát ngôn được tạo hoặc biến đổi bằng kỹ thuật giả mạo.

Do đó, mỗi mẫu dữ liệu là một cặp:

```
(Audio, Label)
```

Trong đó:

- **Audio** là tín hiệu tiếng nói.
- **Label** là nhãn phân loại nhị phân.

---

## Thành phần dữ liệu

Bộ dữ liệu bao gồm hai nhóm dữ liệu chính:

### 1. Bonafide Speech

Đây là các đoạn ghi âm giọng nói thật do con người phát âm trong điều kiện thu thập của bộ dữ liệu VSASV.

Các mẫu Bonafide đóng vai trò là lớp dương (genuine speech), phản ánh đặc điểm tự nhiên của tín hiệu tiếng nói tiếng Việt.

---

### 2. Spoof Speech

Đây là các đoạn ghi âm được tạo hoặc chỉnh sửa bằng các kỹ thuật giả mạo giọng nói.

Các mẫu Spoof được xây dựng nhằm mô phỏng những hình thức tấn công có thể xảy ra đối với hệ thống xác thực người nói, giúp mô hình học được các đặc trưng phân biệt giữa tiếng nói thật và tiếng nói giả.

Các kỹ thuật giả mạo cụ thể được sử dụng là những kỹ thuật được nhóm tác giả VSASV công bố trong bộ dữ liệu gốc.

---

## Mức độ đa dạng của dữ liệu

Bộ dữ liệu được thiết kế nhằm phản ánh nhiều nguồn biến thiên thường gặp trong tiếng nói tiếng Việt, bao gồm:

- Nhiều người nói khác nhau.
- Khác biệt về đặc điểm giọng nói.
- Nhiều câu phát ngôn.
- Nhiều mẫu Bonafide và Spoof.
- Nhiều điều kiện tạo dữ liệu giả mạo theo thiết kế của VSASV.

Sự đa dạng này giúp tăng khả năng tổng quát hóa của các mô hình học sâu khi đánh giá trên dữ liệu chưa từng xuất hiện trong quá trình huấn luyện.

---

## Cấu trúc dữ liệu

Trong dự án này, mỗi mẫu dữ liệu bao gồm các thành phần sau:

| Thuộc tính | Mô tả |
|------------|------|
| Audio | Tệp âm thanh đầu vào |
| Label | Nhãn Bonafide hoặc Spoof |
| Filename | Tên tệp âm thanh |
| Speaker ID* | Định danh người nói (nếu được sử dụng) |

> *Speaker ID chỉ được giữ lại nhằm phục vụ việc quản lý dữ liệu. Bài toán Speech Anti-Spoofing trong dự án này không sử dụng thông tin định danh người nói làm đầu vào cho mô hình.*

---

## Định dạng dữ liệu

Các mẫu dữ liệu được lưu dưới dạng các tệp âm thanh số.

Trong quá trình thực nghiệm, dữ liệu được đọc trực tiếp từ các tệp âm thanh và được xử lý theo cùng một quy trình tiền xử lý trước khi đưa vào mô hình học sâu. Chi tiết về quá trình này sẽ được trình bày trong mục **Data Preprocessing**.

---

## Tính toàn vẹn của bộ dữ liệu

Mỗi mẫu dữ liệu đều có:

- Một tệp âm thanh hợp lệ.
- Một nhãn phân loại tương ứng.
- Thông tin định danh cần thiết để quản lý và chia tập dữ liệu.

Không có mẫu dữ liệu nào được gán đồng thời nhiều nhãn. Mỗi phát ngôn chỉ thuộc duy nhất một trong hai lớp **Bonafide** hoặc **Spoof**, đảm bảo tính nhất quán cho bài toán phân loại nhị phân.

# Collection Process

## Nguồn gốc dữ liệu

Bộ dữ liệu được sử dụng trong dự án này được xây dựng dựa trên **VSASV Dataset**, một bộ dữ liệu công khai do **HUST Excellent Students in Pattern Recognition Laboratory (HUST EP Lab)** phát triển nhằm phục vụ nghiên cứu về **Spoofing-Aware Speaker Verification (SASV)** và **Speech Anti-Spoofing** cho tiếng Việt.

Toàn bộ dữ liệu gốc được tải từ kho dữ liệu chính thức của VSASV. Dự án này không thực hiện thu thập thêm dữ liệu mới mà kế thừa dữ liệu gốc để xây dựng một phiên bản đã được tiền xử lý phục vụ quá trình thực nghiệm.

---

## Mục tiêu thu thập dữ liệu

VSASV được xây dựng nhằm cung cấp một bộ dữ liệu tiếng Việt quy mô lớn phục vụ nghiên cứu về:

- Phát hiện giọng nói giả mạo (Speech Anti-Spoofing).
- Xác thực người nói an toàn (Spoofing-Aware Speaker Verification).
- Đánh giá khả năng chống lại các cuộc tấn công giả mạo bằng giọng nói.
- Phát triển và so sánh các mô hình học máy trong lĩnh vực xử lý tiếng nói.

Trong dự án này, dữ liệu chỉ được sử dụng cho bài toán **Speech Anti-Spoofing**, trong đó mục tiêu là phân loại một đoạn âm thanh thành **Bonafide** hoặc **Spoof**.

---

## Đối tượng thu thập

Bộ dữ liệu bao gồm các phát ngôn của nhiều người nói tiếng Việt với sự đa dạng về đặc điểm giọng nói và cách phát âm.

Mỗi người nói có thể xuất hiện trong nhiều phát ngôn khác nhau nhằm tăng tính đa dạng của dữ liệu và giúp các mô hình học được nhiều đặc trưng ngữ âm hơn thay vì phụ thuộc vào một số ít người nói.

---

## Quy trình xây dựng dữ liệu

Theo mô tả của bộ dữ liệu VSASV, quy trình xây dựng dữ liệu bao gồm hai nhóm chính:

### 1. Thu thập dữ liệu Bonafide

Các đoạn ghi âm giọng nói thật được thu thập từ người nói tiếng Việt và đóng vai trò là dữ liệu gốc (Bonafide Speech).

Các mẫu này phản ánh đặc điểm tự nhiên của tín hiệu tiếng nói và được sử dụng làm chuẩn để so sánh với các mẫu giả mạo.

---

### 2. Tạo dữ liệu Spoof

Từ các đoạn ghi âm Bonafide, nhóm tác giả VSASV tạo ra các mẫu **Spoof Speech** bằng các kỹ thuật giả mạo giọng nói.

Các mẫu giả mạo được xây dựng nhằm mô phỏng các hình thức tấn công có thể xảy ra đối với hệ thống xác thực người nói trong thực tế.

Việc sử dụng nhiều phương pháp giả mạo giúp tăng tính đa dạng của lớp Spoof và tạo ra môi trường đánh giá sát với các tình huống ứng dụng thực tế.

---

## Gán nhãn dữ liệu

Sau khi hoàn thành quá trình xây dựng dữ liệu, mỗi đoạn âm thanh được gán duy nhất một nhãn:

| Nhãn | Ý nghĩa |
|------|---------|
| Bonafide | Giọng nói thật do con người phát âm |
| Spoof | Giọng nói được tạo hoặc chỉnh sửa bằng kỹ thuật giả mạo |

Việc gán nhãn được thực hiện dựa trên nguồn gốc của từng mẫu dữ liệu và được công bố trong bộ dữ liệu VSASV.

---

## Kiểm soát chất lượng trong quá trình xây dựng

Theo tài liệu của bộ dữ liệu gốc, dữ liệu được tổ chức và quản lý theo các siêu dữ liệu (metadata) đi kèm, bao gồm thông tin về người nói, tệp âm thanh và nhãn của từng mẫu.

Điều này giúp đảm bảo:

- Mỗi tệp âm thanh có một nhãn duy nhất.
- Không xảy ra trùng lặp nhãn trên cùng một mẫu dữ liệu.
- Dữ liệu có thể được quản lý và chia thành các tập huấn luyện, xác thực và kiểm thử một cách nhất quán.

---

## Vai trò của dự án này

Dự án **không can thiệp vào quá trình thu thập hoặc tạo dữ liệu gốc**.

Thay vào đó, dự án kế thừa VSASV Dataset và thực hiện các bước:

- Chuẩn hóa dữ liệu âm thanh.
- Tiền xử lý tín hiệu.
- Chuẩn bị dữ liệu cho mô hình học sâu.
- Chia dữ liệu phục vụ huấn luyện và đánh giá.

Các bước trên không làm thay đổi nội dung hoặc nhãn của dữ liệu gốc mà chỉ chuyển đổi dữ liệu sang định dạng phù hợp cho quá trình thực nghiệm.

Chi tiết của các bước này sẽ được trình bày trong mục **Data Preprocessing**.

---

## Khả năng tái lập

Toàn bộ quy trình sử dụng dữ liệu trong dự án có thể được tái lập bằng cách:

1. Tải bộ dữ liệu VSASV từ nguồn chính thức.
2. Thực hiện các bước tiền xử lý được mô tả trong mục **Data Preprocessing**.
3. Chia dữ liệu theo cấu hình được trình bày trong mục **Data Splits**.
4. Sử dụng dữ liệu đã xử lý để huấn luyện và đánh giá mô hình Speech Anti-Spoofing.

Quy trình này giúp đảm bảo các thí nghiệm có thể được lặp lại và kiểm chứng bởi các nhà nghiên cứu khác.

# Data Preprocessing

## Mục tiêu

Mặc dù được xây dựng từ cùng một nguồn dữ liệu, các tệp âm thanh trong VSASV vẫn tồn tại những khác biệt về đặc tính tín hiệu như tần số lấy mẫu, độ dài, mức biên độ và khoảng lặng ở đầu hoặc cuối tín hiệu. Những khác biệt này có thể làm giảm tính nhất quán của dữ liệu đầu vào và ảnh hưởng đến khả năng học đặc trưng của mô hình.

Do đó, trước khi trích xuất đặc trưng và huấn luyện mô hình, toàn bộ dữ liệu được chuẩn hóa thông qua một quy trình tiền xử lý thống nhất nhằm:

- Đồng nhất định dạng của tất cả tệp âm thanh.
- Loại bỏ các thành phần tín hiệu không cần thiết.
- Chuẩn hóa đầu vào cho mô hình học sâu.
- Tăng tính ổn định và khả năng tái lập của quá trình huấn luyện.

Quy trình tiền xử lý chỉ thay đổi biểu diễn của tín hiệu âm thanh, không làm thay đổi nội dung lời nói hoặc nhãn của dữ liệu.

---

## Dữ liệu đầu vào

Dữ liệu đầu vào là các tệp âm thanh WAV được lựa chọn từ bộ dữ liệu gốc VSASV.

```
datasets/raw/VSASV_PAPER_50000/
├── train/
├── val/
└── test/
```

Bộ dữ liệu sử dụng trong dự án bao gồm **50.000 tệp âm thanh**, được tổ chức theo các tập huấn luyện, xác thực và kiểm thử.

---

## Dữ liệu đầu ra

Sau khi hoàn thành tiền xử lý, toàn bộ dữ liệu được lưu tại:

```
datasets/processed/VSASV_PAPER_50000/
```

Cấu trúc thư mục và tên tệp được giữ nguyên nhằm đảm bảo khả năng truy xuất dữ liệu và tính tương thích với các bước tiếp theo trong pipeline.

---

## Cấu hình tiền xử lý

Toàn bộ dữ liệu được chuẩn hóa theo cùng một cấu hình.

| Tham số | Giá trị |
|---------|---------|
| Sampling Rate | 16.000 Hz |
| Audio Channel | Mono |
| Duration | 4 giây |
| Target Length | 64.000 samples |
| Silence Trimming | Có |
| Trim Threshold | 30 dB |
| Amplitude Normalization | [-1, 1] |

---

## Quy trình tiền xử lý

### 1. Chuẩn hóa tần số lấy mẫu và số kênh

Tất cả các tệp âm thanh được chuyển về:

- Sampling Rate: **16 kHz**
- Mono channel

Việc chuẩn hóa này giúp giảm sự khác biệt giữa các nguồn dữ liệu và đảm bảo tính nhất quán của đầu vào. Đồng thời, tần số lấy mẫu 16 kHz vẫn bảo toàn đầy đủ dải tần quan trọng của tín hiệu tiếng nói trong khi giảm chi phí lưu trữ và tính toán.

---

### 2. Loại bỏ DC Offset

Sau khi đọc dữ liệu, giá trị trung bình của tín hiệu được đưa về 0 nhằm loại bỏ thành phần DC Offset có thể xuất hiện trong quá trình thu âm hoặc xử lý dữ liệu.

Bước này giúp phổ tín hiệu phản ánh chính xác hơn đặc điểm của giọng nói và cải thiện tính ổn định của quá trình trích xuất đặc trưng.

---

### 3. Loại bỏ khoảng lặng

Khoảng lặng ở đầu và cuối mỗi đoạn ghi âm được loại bỏ bằng phương pháp Voice Activity Detection dựa trên ngưỡng năng lượng (30 dB).

Việc loại bỏ khoảng lặng giúp:

- Giảm thông tin dư thừa.
- Tăng tỷ lệ tín hiệu chứa nội dung lời nói.
- Giúp mô hình tập trung vào các đặc trưng mang tính phân biệt.

---

### 4. Chuẩn hóa biên độ

Sau khi loại bỏ khoảng lặng, biên độ của mỗi tín hiệu được chuẩn hóa về khoảng:

```
[-1, 1]
```

Quá trình này làm giảm sự khác biệt về mức âm lượng giữa các bản ghi, đồng thời giúp quá trình tối ưu mô hình diễn ra ổn định hơn.

---

### 5. Chuẩn hóa độ dài

Để tạo đầu vào có kích thước cố định cho mô hình CNN, tất cả các đoạn âm thanh được đưa về thời lượng **4 giây** (64.000 mẫu).

- Các đoạn ngắn hơn được bổ sung bằng **Zero Padding** ở cuối tín hiệu.
- Các đoạn dài hơn được cắt bớt phần vượt quá 4 giây.

Việc chuẩn hóa độ dài giúp tất cả mẫu dữ liệu có cùng kích thước đầu vào, thuận lợi cho quá trình huấn luyện theo mini-batch.

---

### 6. Lưu dữ liệu

Các tệp âm thanh sau khi xử lý được lưu dưới định dạng WAV với tần số lấy mẫu 16 kHz.

Tên tệp và cấu trúc thư mục được giữ nguyên nhằm đảm bảo khả năng ánh xạ giữa dữ liệu gốc và dữ liệu đã xử lý.

---

## Khả năng tái lập

Pipeline tiền xử lý được thiết kế theo hướng xác định (deterministic), nghĩa là cùng một dữ liệu đầu vào sẽ luôn tạo ra cùng một kết quả đầu ra.

Ngoài ra, chương trình kiểm tra sự tồn tại của tệp đầu ra trước khi xử lý nhằm:

- Tránh ghi đè dữ liệu đã xử lý.
- Cho phép tiếp tục xử lý nếu pipeline bị gián đoạn.
- Giảm thời gian thực thi khi chạy lại toàn bộ quy trình.

---

## Kết quả tiền xử lý

Sau khi hoàn thành pipeline:

- Toàn bộ **50.000** tệp âm thanh được xử lý thành công.
- Không phát sinh lỗi trong quá trình xử lý.
- Tất cả dữ liệu được chuẩn hóa về **16 kHz**, **Mono**.
- Khoảng lặng ở đầu và cuối tín hiệu được loại bỏ.
- Thành phần DC Offset được loại bỏ.
- Biên độ tín hiệu được chuẩn hóa về khoảng **[-1, 1]**.
- Độ dài của tất cả các mẫu được chuẩn hóa thành **4 giây (64.000 samples)**.

Dữ liệu sau tiền xử lý được sử dụng trực tiếp cho bước **Feature Extraction**, nơi các tín hiệu âm thanh được chuyển đổi thành biểu diễn **Log-Mel Spectrogram** để phục vụ quá trình huấn luyện và đánh giá mô hình Speech Anti-Spoofing.

# Quality Assurance

## Mục tiêu

Quá trình đảm bảo chất lượng dữ liệu được thực hiện nhằm đảm bảo rằng toàn bộ dữ liệu đầu vào đáp ứng các yêu cầu cần thiết trước khi được sử dụng cho quá trình trích xuất đặc trưng và huấn luyện mô hình.

Các bước kiểm tra tập trung vào tính toàn vẹn của dữ liệu, sự nhất quán về định dạng, tính chính xác của nhãn và khả năng sử dụng của các tệp âm thanh.

---

## Đảm bảo chất lượng của bộ dữ liệu gốc

Bộ dữ liệu sử dụng trong dự án được kế thừa từ **VSASV Dataset**, một bộ dữ liệu nghiên cứu được công bố nhằm phục vụ bài toán Spoofing-Aware Speaker Verification (SASV).

Các mẫu dữ liệu trong VSASV đã được tổ chức theo cấu trúc rõ ràng, trong đó mỗi tệp âm thanh được gắn với một nhãn duy nhất và các thông tin siêu dữ liệu (metadata) tương ứng. Điều này giúp đảm bảo tính nhất quán giữa dữ liệu âm thanh và nhãn phân loại.

Dự án không thay đổi nội dung hoặc nhãn của bộ dữ liệu gốc mà chỉ thực hiện các bước tiền xử lý để chuẩn hóa dữ liệu trước khi huấn luyện mô hình.

---

## Kiểm tra dữ liệu đầu vào

Trước khi bắt đầu tiền xử lý, hệ thống kiểm tra sự tồn tại của thư mục chứa dữ liệu đầu vào.

Nếu không tìm thấy dữ liệu hoặc đường dẫn không hợp lệ, chương trình sẽ dừng thực thi và thông báo lỗi nhằm tránh việc xử lý trên bộ dữ liệu không đầy đủ hoặc sai cấu trúc.

Ngoài ra, chương trình chỉ xử lý các tệp âm thanh có định dạng phù hợp, giúp giảm thiểu nguy cơ phát sinh lỗi trong quá trình đọc dữ liệu.

---

## Kiểm tra quá trình tiền xử lý

Sau mỗi bước tiền xử lý, dữ liệu tiếp tục được kiểm tra để đảm bảo các yêu cầu sau:

- Tất cả tệp âm thanh được chuyển về tần số lấy mẫu **16 kHz**.
- Mỗi tệp chỉ còn một kênh âm thanh (Mono).
- Biên độ tín hiệu được chuẩn hóa về cùng một khoảng giá trị.
- Độ dài của mọi tệp âm thanh được chuẩn hóa thành **4 giây (64.000 mẫu)**.
- Khoảng lặng ở đầu và cuối tín hiệu được loại bỏ theo cùng một tiêu chí.
- Không thay đổi nội dung lời nói hoặc nhãn của dữ liệu.

Việc chuẩn hóa này giúp tất cả các mẫu dữ liệu có cùng định dạng đầu vào trước khi trích xuất đặc trưng.

---

## Kiểm tra dữ liệu đầu ra

Sau khi hoàn thành tiền xử lý, các tệp âm thanh được ghi ra thư mục đích và giữ nguyên cấu trúc thư mục cũng như tên tệp so với dữ liệu gốc.

Quá trình này giúp đảm bảo:

- Không làm mất dữ liệu.
- Không thay đổi cấu trúc thư mục.
- Dễ dàng đối chiếu giữa dữ liệu gốc và dữ liệu đã xử lý.
- Hỗ trợ truy vết nếu phát sinh lỗi trong các bước tiếp theo.

---

## Kiểm tra tính toàn vẹn của dữ liệu

Sau khi hoàn thành pipeline, dữ liệu được kiểm tra để xác nhận:

- Không xuất hiện tệp âm thanh bị thiếu.
- Không xuất hiện tệp âm thanh rỗng.
- Mỗi tệp âm thanh đều có nhãn tương ứng.
- Không tồn tại mẫu dữ liệu bị gán nhiều nhãn.
- Không thay đổi số lượng mẫu dữ liệu trong quá trình tiền xử lý.

Việc kiểm tra này giúp đảm bảo rằng dữ liệu sau xử lý vẫn phản ánh đầy đủ bộ dữ liệu ban đầu.

---

## Kiểm tra dữ liệu trùng lặp

Trong quá trình kiểm tra chất lượng dữ liệu, các tệp âm thanh được so sánh thông qua giá trị băm MD5 nhằm phát hiện các mẫu có nội dung hoàn toàn giống nhau.

Kết quả cho thấy tồn tại **09 cặp tệp âm thanh trùng khớp hoàn toàn** nhưng được gán cho **các Speaker ID khác nhau** trong bộ dữ liệu VSASV. Các cặp dữ liệu này xuất hiện giữa các tập **train**, **validation** và **test**.

Phát hiện này cho thấy bộ dữ liệu gốc có tồn tại một lượng nhỏ dữ liệu trùng lặp hoặc sai lệch trong quá trình gán định danh người nói. Tuy nhiên, với tổng quy mô khoảng **50.000 mẫu**, số lượng này chiếm tỷ lệ rất nhỏ và được đánh giá là không ảnh hưởng đáng kể đến quá trình huấn luyện cũng như đánh giá mô hình Speech Anti-Spoofing.

Các mẫu dữ liệu này được giữ nguyên nhằm đảm bảo tính nhất quán với bộ dữ liệu VSASV gốc.

---

## Khả năng tái lập

Pipeline tiền xử lý được xây dựng theo quy trình xác định (deterministic), nghĩa là cùng một dữ liệu đầu vào sẽ luôn tạo ra cùng một kết quả đầu ra.

Bên cạnh đó, chương trình sử dụng cơ chế bỏ qua các tệp đã được xử lý trước đó, giúp:

- Hạn chế việc ghi đè dữ liệu.
- Tiết kiệm thời gian khi chạy lại pipeline.
- Cho phép tiếp tục xử lý nếu quá trình thực thi bị gián đoạn.

Điều này góp phần nâng cao tính ổn định và khả năng tái lập của toàn bộ quy trình xử lý dữ liệu.

---

## Kiểm tra Data Leakage

Bộ dữ liệu được xây dựng theo **speaker-independent protocol**, trong đó mỗi người nói chỉ xuất hiện trong duy nhất một tập dữ liệu.

Quá trình kiểm tra xác nhận:

- Không tồn tại **speaker overlap** giữa các tập huấn luyện, xác thực và kiểm thử.
- Không phát hiện hiện tượng **data leakage** phát sinh từ quá trình tiền xử lý dữ liệu.
- Không phát hiện data leakage trong quá trình trích xuất đặc trưng hoặc huấn luyện mô hình.

Mặc dù phát hiện một số cặp âm thanh trùng lặp giữa các tập dữ liệu, đây là đặc điểm vốn có của bộ dữ liệu VSASV và không phải do quy trình xây dựng bộ dữ liệu của dự án.

Do bài toán chính của dự án là **Speech Anti-Spoofing**, các mẫu trùng lặp này được đánh giá không tạo ra ảnh hưởng đáng kể đến kết quả thực nghiệm.

---

## Kết quả kiểm tra

Sau khi hoàn thành quá trình tiền xử lý:

- Toàn bộ **50.000** tệp âm thanh được xử lý thành công.
- Không phát sinh lỗi trong quá trình xử lý.
- Không ghi nhận trường hợp mất dữ liệu hoặc sai lệch số lượng tệp.
- Tất cả các tệp đều đáp ứng cấu hình chuẩn hóa đã thiết lập.
- Dữ liệu đầu ra sẵn sàng cho bước trích xuất đặc trưng và huấn luyện mô hình.

---

## Đánh giá trực quan

Ngoài các kiểm tra tự động, dữ liệu còn được đánh giá thông qua việc quan sát trực quan các biểu diễn phổ (spectrogram) của các mẫu Bonafide và Spoof.

Kết quả quan sát cho thấy các mẫu thuộc hai lớp có sự khác biệt tương đối rõ ràng về phân bố năng lượng theo thời gian và tần số. Một số mẫu Spoof xuất hiện các đặc điểm phổ ít tự nhiên hơn so với các mẫu Bonafide, phản ánh sự khác biệt do quá trình tổng hợp hoặc biến đổi giọng nói tạo ra.

Những khác biệt này phù hợp với mục tiêu thiết kế của VSASV là hỗ trợ nghiên cứu về **Spoofing-Aware Speaker Verification** và **Speech Anti-Spoofing**, đồng thời tạo điều kiện thuận lợi cho việc học các đặc trưng phân biệt của mô hình học sâu.

Tuy nhiên, đánh giá này chỉ mang tính chất quan sát trực quan và không thay thế cho các phân tích định lượng về khả năng phân biệt giữa các lớp dữ liệu.

---

## Hạn chế

Quy trình đảm bảo chất lượng trong dự án chủ yếu tập trung vào chất lượng kỹ thuật của dữ liệu, bao gồm định dạng tệp, tính toàn vẹn và sự nhất quán của tín hiệu âm thanh.

Dự án không thực hiện đánh giá lại chất lượng ngữ âm, xác minh thủ công nội dung phát ngôn hoặc gán nhãn lại dữ liệu. Do đó, chất lượng của nhãn và nội dung lời nói vẫn phụ thuộc vào bộ dữ liệu VSASV gốc.

---

## Data Validation

Sau khi hoàn thành tiền xử lý, toàn bộ dữ liệu được kiểm tra lại nhằm xác nhận:

- Không tồn tại tệp âm thanh bị hỏng hoặc không thể đọc.
- Tất cả các tệp đều có cùng tần số lấy mẫu (16 kHz).
- Tất cả các tệp đều ở định dạng Mono.
- Mỗi tệp đều có thời lượng đúng 4 giây sau khi chuẩn hóa.
- Không phát hiện giá trị NaN hoặc Inf trong tín hiệu âm thanh.
- Mỗi tệp đều có nhãn hợp lệ (Bonafide hoặc Spoof).

Các bước kiểm tra này được thực hiện tự động trước khi dữ liệu được sử dụng cho bước trích xuất đặc trưng.

# Biases

## Tổng quan

Mặc dù VSASV là một trong những bộ dữ liệu tiếng Việt có quy mô lớn dành cho bài toán **Spoofing-Aware Speaker Verification (SASV)**, bộ dữ liệu vẫn tồn tại một số thiên lệch (bias) vốn có do quá trình thu thập, xây dựng và lựa chọn dữ liệu. Các thiên lệch này cần được xem xét khi sử dụng bộ dữ liệu để huấn luyện hoặc đánh giá mô hình.

---

## Thiên lệch về ngôn ngữ

Toàn bộ dữ liệu trong bộ dữ liệu đều là **tiếng Việt**.

Do đó, các mô hình được huấn luyện trên bộ dữ liệu này chủ yếu học các đặc trưng của tiếng nói tiếng Việt và có thể không đạt hiệu quả tương đương khi áp dụng cho các ngôn ngữ khác.

---

## Thiên lệch về loại tấn công

Các mẫu Spoof được tạo ra bằng một số kỹ thuật giả mạo cụ thể do nhóm phát triển VSASV lựa chọn.

Vì vậy, mô hình có xu hướng học các đặc trưng của những phương pháp giả mạo này và chưa chắc có khả năng tổng quát tốt đối với các kỹ thuật tổng hợp giọng nói mới hoặc chưa xuất hiện trong bộ dữ liệu.

---

## Thiên lệch về điều kiện thu thập

Dữ liệu được xây dựng trong những điều kiện thu thập xác định và không phản ánh đầy đủ mọi môi trường ghi âm trong thực tế.

Các yếu tố như:

- môi trường nhiều tạp âm,
- thiết bị ghi âm chất lượng thấp,
- truyền tải qua mạng,
- hoặc các điều kiện thu âm đặc biệt,

có thể chưa được đại diện đầy đủ trong bộ dữ liệu.

---

## Thiên lệch do tiền xử lý

Trong dự án này, toàn bộ dữ liệu được chuẩn hóa về:

- 16 kHz,
- Mono,
- thời lượng 4 giây.

Việc chuẩn hóa này giúp tăng tính nhất quán của dữ liệu đầu vào nhưng đồng thời làm giảm sự đa dạng về định dạng tín hiệu. Vì vậy, mô hình được huấn luyện có thể cần được đánh giá lại khi làm việc với các đoạn âm thanh có thời lượng hoặc tần số lấy mẫu khác.

---

## Thiên lệch do lựa chọn dữ liệu

Phiên bản sử dụng trong dự án chỉ bao gồm **50.000 mẫu âm thanh** được trích xuất từ bộ dữ liệu VSASV gốc.

Mặc dù số lượng này đủ lớn để phục vụ quá trình thực nghiệm, nó vẫn chỉ đại diện cho một phần của toàn bộ dữ liệu gốc và có thể chưa phản ánh đầy đủ tất cả các đặc điểm của VSASV.

---

# Limitations

## Giới hạn của bộ dữ liệu

Bộ dữ liệu được xây dựng nhằm phục vụ nghiên cứu về **Speech Anti-Spoofing** và không được thiết kế để đại diện cho toàn bộ các tình huống sử dụng trong thực tế.

Do đó, khi triển khai mô hình trong các hệ thống thực tế, cần cân nhắc những hạn chế dưới đây.

---

## Không đại diện cho mọi hình thức giả mạo

Các mẫu Spoof chỉ bao gồm những phương pháp giả mạo được sử dụng trong bộ dữ liệu VSASV.

Bộ dữ liệu không thể bao phủ toàn bộ các kỹ thuật tổng hợp và chuyển đổi giọng nói đang phát triển nhanh chóng, đặc biệt là các mô hình sinh mới xuất hiện sau thời điểm VSASV được công bố.

---

## Không đánh giá đầy đủ môi trường thực tế

Phần lớn dữ liệu được xây dựng trong điều kiện nghiên cứu có kiểm soát.

Các tình huống như:

- ghi âm qua điện thoại,
- truyền qua nền tảng trực tuyến,
- môi trường nhiều nhiễu,
- thiết bị ghi âm khác nhau,

có thể tạo ra đặc điểm tín hiệu khác với dữ liệu trong bộ dữ liệu này.

---

## Phát hiện dữ liệu trùng lặp

Trong quá trình kiểm tra chất lượng dữ liệu, phát hiện **09 cặp tệp âm thanh trùng khớp hoàn toàn (MD5 giống nhau)** nhưng được gán cho các **Speaker ID khác nhau**.

Đây là đặc điểm tồn tại trong bộ dữ liệu VSASV gốc và không phát sinh từ quá trình xây dựng bộ dữ liệu của dự án.

Do số lượng rất nhỏ so với tổng số mẫu, hiện tượng này được đánh giá không ảnh hưởng đáng kể đến kết quả thực nghiệm. Tuy nhiên, người sử dụng cần lưu ý nếu khai thác bộ dữ liệu cho các bài toán liên quan trực tiếp đến định danh người nói.

---

## Không sử dụng thông tin người nói

Mặc dù VSASV được thiết kế cho bài toán **Spoofing-Aware Speaker Verification**, dự án này chỉ sử dụng nhãn **Bonafide** và **Spoof**.

Các thông tin về định danh người nói không được sử dụng trong quá trình huấn luyện mô hình. Do đó, bộ dữ liệu đã tiền xử lý không phù hợp cho các nghiên cứu yêu cầu khai thác đặc trưng định danh người nói.

---

## Không thay đổi nhãn gốc

Dự án giữ nguyên toàn bộ nhãn của bộ dữ liệu VSASV.

Nếu tồn tại sai sót trong nhãn hoặc metadata của bộ dữ liệu gốc, những sai sót này cũng sẽ được giữ nguyên trong phiên bản đã tiền xử lý.

---

## Khuyến nghị sử dụng

Người sử dụng nên xem bộ dữ liệu này là một tập dữ liệu nghiên cứu phục vụ phát triển và đánh giá các mô hình **Speech Anti-Spoofing**.

Khi triển khai trong các hệ thống thực tế hoặc mở rộng sang các miền dữ liệu khác, nên đánh giá bổ sung trên các bộ dữ liệu độc lập để kiểm tra khả năng tổng quát hóa của mô hình.

