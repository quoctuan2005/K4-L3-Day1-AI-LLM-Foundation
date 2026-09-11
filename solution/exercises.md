# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi thử chạy với 4 mức temperature, em thấy rõ sự khác biệt: ở mức 0.0, câu trả lời rất an toàn và ổn định, lần nào chạy cũng ra cùng một sự thật quen thuộc (như Việt Nam xuất khẩu cà phê); ở mức 0.5 đến 1.0, câu trả lời đa dạng và diễn đạt tự nhiên hơn hẳn; còn khi đẩy lên 1.5 thì câu cú bắt đầu bị lủng củng, dùng từ ngữ lạ và có cảm giác mô hình dễ bịa ra thông tin không chuẩn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em sẽ đặt khoảng 0.1 đến 0.2 (tối đa 0.3). Chatbot chăm sóc khách hàng cần nhất là sự chính xác, câu trả lời phải nhất quán và bám sát tài liệu quy định của công ty, nếu để temperature cao thì bot rất dễ chém gió hoặc trả lời lung tung làm khách hiểu sai thông tin.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Nhìn vào bảng giá output ($0.010 so với $0.0006 cho 1k token), GPT-4o đắt gấp khoảng 16.7 lần so với bản Mini. Với bài toán trên (10.5 triệu token/ngày), dùng GPT-4o tốn khoảng $105/ngày còn dùng Mini chỉ tốn tầm hơn $6/ngày.
> - Đáng tiền dùng GPT-4o: Khi cần giải quyết các bài toán logic khó, viết code phức tạp hoặc đọc hiểu, phân tích tài liệu hợp đồng dài cần độ chính xác tuyệt đối.
> - Nên dùng Mini: Cho các tác vụ đơn giản, lặp đi lặp lại nhiều như chào hỏi, phân loại ý định người dùng (intent), trích xuất thông tin cơ bản hoặc tóm tắt ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác nhau một trời một vực về giọng điệu và từ ngữ: bản giáo viên tiểu học dùng từ rất trong sáng, ví von blockchain giống như cuốn sổ nhật ký chung của cả lớp mà bạn nào cũng được nhìn và không ai tự ý tẩy xóa được; còn bản chuyên gia tài chính dùng toàn từ ngữ chuyên ngành như sổ cái phân tán, cơ chế đồng thuận PoW/PoS, tính toàn vẹn dữ liệu. Qua đó em thấy system prompt giúp định hình phong cách, vai trò và đối tượng nghe cực kỳ hiệu quả mà mình không cần phải train lại mô hình.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn 100 từ tiếng Việt, nếu ước tính thô (100 / 0.75) thì ra tầm 133 token, nhưng đếm bằng tiktoken thì lên tới khoảng 170-190 token, chênh lệch cỡ 30% đến 40%. Tiếng Việt tốn nhiều token hơn tiếng Anh vì bộ tokenizer của OpenAI được huấn luyện chủ yếu bằng tiếng Anh; các từ tiếng Việt có dấu (như á, à, ỏ, ã, ạ, ư, ơ...) thường bị bẻ nhỏ thành nhiều mảnh ký tự (subwords) hoặc từng byte UTF-8 riêng biệt, khiến 1 từ tiếng Việt thường bị tính thành 1.5 đến 2 token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi làm giao diện chat trực tiếp với người dùng, vì từng chữ chạy ra màn hình ngay lập tức giúp người dùng có cái đọc liền mà không phải sốt ruột ngồi chờ vài giây màn hình đứng im. Còn non-streaming sẽ hợp hơn khi mình chạy các tác vụ nền (background job), xử lý dữ liệu hàng loạt không cần người xem, hoặc khi cần bot trả về đúng cấu trúc JSON chuẩn để code phía sau xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp tăng thời gian chờ sau mỗi lần thử lại (0.1s -> 0.2s -> 0.4s...), giúp giãn các request ra để server có thời gian giải tỏa bớt hàng đợi đang nghẽn. Nếu để delay cố định ví dụ 1 giây, khi server vừa chập chờn thì hàng nghìn máy cùng lúc gửi lại request đúng 1 giây sau đó, tạo thành những đợt sóng dồn dập khiến server càng nghẽn nặng hơn và sập hẳn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona là: "Bạn là trợ giảng thân thiện của khóa học AI, giải thích các khái niệm kỹ thuật một cách dễ hiểu bằng tiếng Việt và luôn đưa ra câu trả lời súc tích, ngắn gọn." Em dùng từ "thân thiện, dễ hiểu" để bot trả lời gần gũi, không mang tính phán xét khi học viên hỏi câu đơn giản; cụm "súc tích, ngắn gọn" để câu trả lời đi thẳng vào trọng tâm, không bị dài dòng lan man trên terminal và tiết kiệm token; còn "bằng tiếng Việt" để bot luôn giữ tiếng Việt làm ngôn ngữ chính, không tự ý chuyển sang tiếng Anh khi gặp từ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là bot chỉ nhớ được đúng 3 lượt chat gần nhất, các câu hỏi trước đó bị cắt bỏ hoàn toàn và tắt terminal là quên hết sạch. Để cải thiện, em có thể lưu lịch sử chat vào file JSON hoặc SQLite để mở lại vẫn còn; đồng thời khi đoạn chat quá 3 lượt, mình có thể cho một model nhỏ tóm tắt ngắn gọn các ý quan trọng ở đoạn trước rồi đưa vào prompt, vừa giúp bot nhớ được nội dung xuyên suốt vừa không sợ tốn quá nhiều token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
