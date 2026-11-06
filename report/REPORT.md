# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (Device: cuda)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
  Cả bức ảnh. Vì JSON chỉ có một lớp cho mỗi ảnh và không có tọa độ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
  Đi kèm taxonomy/checkpoint (ImageNet-1K), model không tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
  Cùng một ID có thể có nghĩa khác ở taxonomy khác, nên cần ghi cả ba để không mơ hồ.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
  Guideline cần nói tiêu chí chọn một lớp, ví dụ chọn vật chiếm diện tích lớn nhất hoặc chủ thể chính.
6. Vì sao không được lấy `score` của model làm ground truth?
  score chỉ là độ tự tin của model, model có thể tự tin mà sai. Ground truth do người gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=oven ; `score`=0.686791 ; `bbox_xyxy`=[0.11, 187.54, 195.96, 292.81] ; `bbox_width`=195.85 ; `bbox_height`=105.26
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
  Góc trên-trái ở (0.11, 187.54) và góc dưới-phải ở (195.96, 292.81). Hộp rộng 195.85 pixel và cao 105.26 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
  Ngưỡng 0.20 cho 17 prediction, ngưỡng 0.60 cho 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
  Khi hạ ngưỡng thì giữ lại nhiều hộp hơn (từ 6 lên 17), trong đó có nhiều hộp điểm thấp như `spoon`, `bottle`, `potted plant`, `dining table`. Reviewer phải xem thêm từng hộp để quyết định giữ hay bỏ, kiểm tra sai lớp, hộp trùng và hộp lệch, nên khối lượng kiểm tra tăng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
  Quy tắc box: "Mỗi vật thể thuộc taxonomy có một hộp riêng, ôm sát phần nhìn thấy của vật, không chừa viền thừa và không gộp nhiều vật vào một hộp."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
  Trên ảnh `kitchen`, hộp `person 0.61` ở mép trái chỉ thấy một phần tay, phần còn lại bị cắt khỏi khung ảnh. Tôi chọn **escalation** hỏi Lab Coach: guideline chung có thể quy định "hộp ôm phần nhìn thấy", nhưng chưa nói phần nhìn thấy nhỏ đến mức nào thì vẫn gán nhãn `person`. Đây là ca ranh giới nên không tự quyết, để cả nhóm annotator gán nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-011 ; `class_name`=spoon ; `score`=0.359147 ; số điểm polygon=`polygon_point_count`=37 ; 3–5 điểm đầu của `polygon_xy`=[529.0,66.0] [529.0,72.0] [528.0,73.0] [528.0,74.0] [527.0,75.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
  Box chỉ cho biết vị trí và kích thước một hình chữ nhật (bbox của kitchen-011 rộng khoảng 14 px, cao khoảng 71 px). Polygon là chuỗi điểm [x, y] đi dọc theo mép vật nên cho biết hình dạng thật: vật dài, hẹp, thẳng đứng (một dụng cụ bếp treo trên tường). Từ polygon cũng tính được diện tích thực của vật và biết pixel nào thuộc vật, pixel nào là nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
  `instance_id` chỉ là số thứ tự để phân biệt từng đối tượng trong output của bài lab này, nên hai vật cùng lớp vẫn có ID riêng. Nó không phải class ID, vì class ID là mã chỉ loại vật và giống nhau cho mọi vật cùng loại. Nó cũng không phải tracking ID, vì tracking ID dùng để theo dõi cùng một vật qua nhiều khung hình video, còn đây chỉ là một ảnh tĩnh, và chạy lại với ngưỡng khác thì số thứ tự có thể đổi.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
  Quy tắc vẽ biên: "Vẽ polygon bám theo mép ngoài của phần nhìn thấy của vật, đủ điểm để ôm các chỗ cong, không lấn sang nền hoặc vật khác, và không đưa phần vật khác che lên trên vào mask."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
  Trên ảnh `kitchen`, chỗ khó là cái bàn (`kitchen-005`, `dining table`, 598 điểm). Mask của bàn phủ cả mặt bàn lẫn gầm bàn, đồng thời chồng lên các `bowl`, `person 0.43` và `oven 0.49` nằm trên/sau mép bàn, nên ranh giới giữa bàn và các vật này không rõ. Tôi chọn dùng **guideline**: đây là tình huống lặp lại ở nhiều ảnh (đồ đặt trên bàn), nên có thể viết thành quy tắc rõ, ví dụ "vật nằm trên bàn thì loại khỏi mask của bàn, mỗi vật có mask riêng". Nếu gặp ca guideline không quyết được (ví dụ dụng cụ treo trên tường vừa giống `spoon` vừa giống muôi, mà COCO-80 không có lớp muôi) thì mới escalation hỏi Lab Coach.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 lớp cho cả bức ảnh (ví dụ ảnh `traffic` → 1 nhãn), kèm `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có nhiều xe buýt và ô tô nhưng top-1 là `cab` (score 0.510915), top-2 là `minibus` (0.164284); ImageNet-1K không có lớp "cảnh giao thông" nên không lớp nào mô tả đúng cả cảnh | Chọn một lớp theo quy tắc chủ thể chính trong guideline; ca mơ hồ thì hỏi Lab Coach | Kiểm tra lớp có đúng quy tắc chọn chủ thể và nhất quán giữa các ảnh tương tự; không dựa vào `score` để kết luận đúng/sai |
| Phát hiện vật thể | 1 hộp `[x_min, y_min, x_max, y_max]` (pixel) + 1 lớp COCO-80 cho mỗi vật | Ảnh `kitchen`: `person 0.61` ở mép trái chỉ thấy một phần tay, bị cắt khỏi khung ảnh; hộp `oven` [0.11, 187.54, 195.96, 292.81] chạm sát mép trái và bao cả vùng bếp lò; nhiều hộp `bowl`/`cup` chồng lên nhau ở góc dưới trái; nồi chảo treo tường không được phát hiện (COCO-80 không có lớp phù hợp); hạ ngưỡng từ 0.60 xuống 0.20 thì từ 6 lên 17 hộp | Vẽ hộp ôm sát phần nhìn thấy của từng vật, chọn lớp COCO-80; hỏi Lab Coach với ca vật bị cắt mép hoặc quá nhỏ | Kiểm tra hộp thiếu, thừa, trùng, lệch, sai lớp; rà lại các hộp điểm thấp khi ngưỡng thấp |
| Instance segmentation | 1 polygon `polygon_xy` + 1 lớp COCO-80 cho mỗi instance, mỗi instance có `instance_id` riêng | Ảnh `kitchen`: `kitchen-005` (dining table) có 598 điểm, biên bị bát, chảo, bột trên mặt bàn che nên polygon rời rạc; `kitchen-011` (spoon, score 0.359147) là vật mảnh, polygon có điểm nhảy bất thường; `kitchen-009` (person, 0.431438) bị cắt ở mép trái ảnh | Vẽ polygon bám mép phần nhìn thấy, loại vật che phía trên khỏi mask, mỗi vật một mask; ca khó thì escalation | Kiểm tra biên có sát không, các instance có tách đúng không, mask có lẫn vật khác hoặc sai lớp không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
  Chỉ dùng 3 ảnh COCO công khai có sẵn của bài lab, không tải ảnh cá nhân hay dữ liệu nội bộ, và không ghi họ tên, MSSV, email, số điện thoại hay thông tin cá nhân vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
  Tôi sẽ dừng lại ngay, không tiếp tục xử lý hay lưu dữ liệu đó, và báo cho Lab Coach/Giảng viên.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
