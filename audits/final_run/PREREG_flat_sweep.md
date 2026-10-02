# ĐĂNG KÝ TRƯỚC — quét thuế phẳng + nhánh rl_mean_fixed (chưa chạy nhánh nào)

Ghi lúc: 2026-10-02, trước khi viết mã và trước khi chạy bất kỳ nhánh mới nào. Mô hình đóng băng tại commit `14406fd`
(`be/env.py`, `be/rule_engine.py` không đổi kể từ `ed57a19`). Checkpoint: `be/checkpoint/final_v037_seed{42,202,303}/iter_100`.
Thiết kế: 3 seed × 2 chế độ policy (tất định, lấy mẫu) × 10 episode ghép cặp (reset seed 42+i). Các nhánh mới: `rl_mean_fixed`,
`flat_pit{0.10,0.20,0.30,0.50}_cit0`, `flat_pit0_cit{0.10,0.20,0.30,0.50}`; 3 công cụ phụ (ρ, bơm cầu, trợ cấp) của các nhánh
flat = trung bình hành động RL trong cùng lần benchmark; `rl_mean_fixed` = trung bình cả 5 công cụ của RL, cố định theo thời gian.
Điểm (0, 0) = `free_market+rl_aux`. Nội dung đăng ký (giữ nguyên chữ theo yêu cầu):

- Câu hỏi: planner RL có phản ứng theo trạng thái không, và có tìm được cặp thuế phẳng tốt nhất trong thế giới nó học không?
- H1: hiệu ghép cặp (rl_learned − rl_mean_fixed) có khoảng tin cậy chứa 0 ở cả hai chế độ. Nếu đúng: không có bằng chứng giá trị phản ứng theo trạng thái. Nếu RL cao hơn rõ: tính phản ứng theo trạng thái có giá trị.
- H2: Eq×Prod giảm khi thuế TNCN tăng ở lát cắt thuế DN = 0.
- H3: ở lát cắt thuế TNCN = 0, Eq×Prod tại thuế DN ≤ 0,30 không thấp hơn điểm (0, 0) (free_market+rl_aux), và thấp hơn đỉnh của lát cắt tại 0,50.
- H4 (thăm dò): ở lát cắt thuế DN, 1−Gini thay đổi dưới 0,02 so với điểm (0, 0), và biến thiên Eq×Prod chủ yếu đến từ GDP. (Ngưỡng 0,02 đặt bây giờ, trước khi thấy dữ liệu.)
- Quy tắc: báo cáo mọi nhánh, không bỏ nhánh nào, không chạy lại trừ khi bị lỗi chương trình; nếu kết quả làm yếu câu chuyện, vẫn báo nguyên văn.
