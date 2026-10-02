# Ghi chú vận hành lần chạy quét thuế phẳng (2026-10-02 18:08–20:01) — báo nguyên văn

1. **Chạy trùng 4 tổ hợp (lỗi vận hành của Claude Code).** Lúc 18:38 công cụ báo lệnh nền `run_flat_sweep.sh` "killed" do giới hạn thời
   gian; thực tế chỉ phần theo dõi bị dừng, vòng lặp vẫn chạy. Claude Code hiểu sai và khởi chạy thêm 4 lệnh riêng cho seed 202/303 × 2 chế
   độ → mỗi tổ hợp này chạy HAI LẦN song song, cùng ghi vào một file JSON/log (bản kết thúc sau ghi đè). Không phải chạy lại có chọn lọc.
   Bằng chứng không ảnh hưởng dữ liệu: mỗi tiến trình được gieo seed cố định (`build_ppo_config(seed=42)`), nên với cùng thứ tự nhánh kết quả
   tất định; `rl_learned` (cả 2 chế độ) và `free_market+rl_aux` (tất định) trùng TUYỆT ĐỐI (lệch 0.00) với benchmark cũ chạy riêng lẻ; cả 6
   file đủ 11 nhánh × 10 episode. Thời gian thực tổng 1 giờ 53 phút (< 3 giờ), nhưng 18:42–20:01 máy chạy 2 tiến trình benchmark đồng thời.
2. **Traceback trong `sweep_seed42_exp.log`**: "Exception ignored in atexit callback: shutdown" — lỗi Ray khi tắt tiến trình lúc thoát (do shell
   cha bị dừng), xảy ra SAU khi JSON đã ghi; không phải lỗi chương trình ảnh hưởng kết quả.
3. **`provenance.git_head` rỗng ở `sweep_seed42_exp.json`** (lệnh git con thất bại khi shell cha bị dừng). File chạy 18:24–18:42 trên cây mã
   commit `0d964d8` (18:09), không có file chưa commit (`git_dirty` rỗng); sha256 env.py/rule_engine.py/checkpoint trước = sau.
4. **Chế độ lấy mẫu KHÔNG ghép cặp được nhiễu hành động giữa các nhánh**: torch RNG không gieo lại theo nhánh/episode → kết quả một nhánh phụ thuộc
   thứ tự chạy. `free_market+rl_aux` (chạy thứ 2 lần này, thứ 5 ở benchmark cũ) lệch tới 63–90 Eq×Prod mỗi episode so với benchmark cũ. Ghép
   cặp ở chế độ lấy mẫu chỉ là ghép điều kiện ban đầu; CI của hiệu ghép cặp vẫn hợp lệ (đã chứa nhiễu này) nhưng không được gọi là "ghép cặp đầy
   đủ". Chế độ tất định không bị ảnh hưởng (lệch 0.00).
