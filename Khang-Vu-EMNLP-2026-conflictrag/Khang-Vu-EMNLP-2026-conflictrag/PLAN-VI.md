# Kế Hoạch Kỹ Thuật — ConflictRAG

Project: `EMNLP-2026-conflictrag`
Tiêu đề: *ConflictRAG: Policy Evaluation for Retrieval-Augmented Generation with Conflicting Evidence*

> **Thông báo trạng thái:** Mọi kết quả định lượng trong gói tài liệu này đều là **tổng hợp nhân tạo (synthetic)**. Các bảng trong `sections/*.tex` và `appendix/*.tex` là các con số nội tuyến được mã cứng (hardcoded); bốn file hình PDF trong `figures/` được render bởi [`scripts/generate_figures.py`](scripts/generate_figures.py) từ các số **hardcoded bên trong script đó**; và các file JSON/CSV trong `results/` được viết bởi [`scripts/make_artifacts.py`](scripts/make_artifacts.py) từ một dict `DATA` cố định. Chưa có thí nghiệm thực sự nào được chạy. Tệ hơn nữa, ba lớp artifact (bảng trong bài báo, script hình, JSON kết quả) **thậm chí không nhất quán với nhau** — xem [§4](#4-data--artifact-dependency-map) và cảnh báo sai lệch. Kế hoạch này là quy trình để thay thế tất cả chúng bằng các phép đo thực.

---

## 1. Tóm tắt bài báo & trạng thái hiện tại

ConflictRAG đề xuất một **lớp policy (chính sách) không cần huấn luyện (training-free), chỉ dùng CPU** nằm giữa bộ retriever (truy xuất) và generator (sinh văn bản) của RAG, quyết định theo từng truy vấn trong số bốn hành động có cấu trúc — `Answer`, `Disclaimer`, `Disclose` (tiết lộ xung đột), `Abstain` — dựa trên bốn tín hiệu phía truy xuất: **dispersion** (phân tán) cặp đoạn văn (sentence-transformer embeddings, Eq. `eq:dispersion`), **NLI contradiction** (mâu thuẫn suy luận ngôn ngữ tự nhiên) (cross-encoder), **semantic claim support** (hỗ trợ tuyên bố ngữ nghĩa), và **language/domain mismatch** (bất khớp ngôn ngữ/lĩnh vực), được đưa vào lớp quyết định logistic regression 6 chiều với ngưỡng abstention (bỏ qua) θ có thể tinh chỉnh. Đánh giá trên tập kiểm tra kết hợp 2.394 ví dụ bao gồm WhoQA, NoMIRACL-nonrelevant, ConflictQA, FaithDial so với năm baseline trở lên (No-filter RAG, Naive Abstain, Confidence Threshold, NLI Conflict Detector, GPT-4 Verifier, cùng GPT-4-CoT / GPT-4o-Direct xuất hiện trong bảng nhưng không được mô tả trong phần baseline), báo cáo Answer F1, hallucination rate (tỷ lệ ảo giác) (theo kiểu FActScore), abstention precision/recall, conflict-disclosure F1, và Brier score, với chuyển giao held-out sang PubMedQA và ASQA.

Các tuyên bố chính (giảm 55,7% hallucination tương đối so với No-filter RAG, giữ lại 96,8% chất lượng trả lời, Brier 0,214 so với GPT-4 là 0,198 với chi phí API bằng 0) đều là **do simulator tạo ra** và phải được thay thế trước khi nộp. Ba nguy cơ ngay lập tức ngoài "con số là giả": (a) **không có thư mục `configs/`** (giả định của prompt không đúng ở đây — toàn bộ cấu hình thí nghiệm chỉ có dạng văn xuôi trong `sections/04_experimental_setup.tex` và phải được soạn từ đầu); (b) schema `results/*.json` (`Control` / `Cheap baseline` / `Strong API` / `ConflictRAG` với `quality`/`risk`/`cost`) là một **template chung không liên quan gì đến các bảng 7-hệ-thống × 6-metric của bài báo**; (c) các hình được tạo bởi một script hardcoded *thứ hai*, độc lập (`generate_figures.py`) mà không bao giờ đọc `results/`. Con đường trung thực là chạy thí nghiệm thực (rẻ, khả thi trên laptop) được mô tả trong §5, định nghĩa schema kết quả đúng, kết nối cả hai script với nó, và tự động tạo bảng.

---

## 2. Danh sách đầy đủ các BẢNG

Tất cả các bảng đều là **LaTeX nội tuyến hardcoded** (không có `\input` từ kết quả, không đọc `pgfplotstable`). "Nguồn thực" = thí nghiệm + file kết quả phải tạo ra từng ô sau khi thay thế.

| # | `\label` | File nguồn:dòng | Cột / metric được báo cáo | Thí nghiệm thực + file kết quả cần có |
|---|----------|-----------------|---------------------------|----------------------------------------|
| T1 | `tab:main` | [`sections/05_results.tex`:10–30](sections/05_results.tex) | Hệ thống × {Ans. F1, Halluc. %, Abst. P, Abst. R, Conf. Disc. F1, Brier}; 7 hệ thống (No-filter RAG, Naive Abstain, Confidence Threshold, NLI Conflict Detector, GPT-4 Verifier, GPT-4-CoT, GPT-4o-Direct) + ConflictRAG; `†` = bootstrap sig. | **Exp-A chạy chính** trên tập kiểm tra kết hợp 2.394 ví dụ, tất cả hệ thống → `results/main_results.json` (schema mới theo hệ-thống×metric với seeds + bootstrap CIs) |
| T2 | `tab:dataset-matrix` | [`sections/05_results.tex`:32–48](sections/05_results.tex) | Lát cắt dataset × {Items, Validation, Role}: WhoQA 428, NoMIRACL 438, ConflictQA 408, FaithDial 533, Dev 587 | **Bước xây dựng/kê khai dataset** → `results/dataset_matrix.csv` (số lượng thực sau dedup + κ) |
| T3 | `tab:per-dataset` | [`sections/05_results.tex`:54–70](sections/05_results.tex) | Dataset × {No-filter RAG, NLI Detector, ConflictRAG} dạng "Answer F1 / Halluc. %" | **Phân tích theo dataset của Exp-A** → `results/per_dataset.json` mới |
| T4 | `tab:error-taxonomy` | [`sections/05_results.tex`:72–88](sections/05_results.tex) | Loại lỗi × Tỷ lệ %: retrieval miss 12,5; unsupported gen 9,7; calibration 6,1; semantic drift 5,7; over-abstention 4,3 | **Chú thích lỗi thủ công** trên 200 lỗi ConflictRAG được lấy mẫu → `results/error_taxonomy.csv` |
| T5 | `tab:generalization` | [`sections/05_results.tex`:134–148](sections/05_results.tex) | Dataset × {GPT-4 Verifier, ConflictRAG} Answer F1: PubMedQA, ASQA, Original avg | **Exp-D chuyển giao OOD (Answer F1)** → `results/generalization.json` mới |
| T6 | `tab:ablation` | [`sections/06_analysis.tex`:7–27](sections/06_analysis.tex) | Biến thể × {Overall F1, Δ}; 5 lần loại bỏ đặc trưng đơn + 2 lần loại bỏ kết hợp | **Exp-B ablation** (khớp lại logistic với tập con đặc trưng) → `results/ablations.json` |
| T7 | `tab:hparam` | [`sections/06_analysis.tex`:34–50](sections/06_analysis.tex) | τ_s ∈ {0,40; 0,50; 0,60; 0,70; 0,80} × Overall F1 | **Exp-C quét độ nhạy trên τ_s** → `results/sensitivity.json` mới |
| T8 | `tab:efficiency` | [`sections/06_analysis.tex`:125–141](sections/06_analysis.tex) | Phương pháp × {Thời gian/item giây, RAM đỉnh GB, Items/giờ}; 5 hệ thống | **Đo thời gian/bộ nhớ** trong Exp-A → `results/efficiency.json` mới |
| T9 | `tab:iaa-ablation` | [`sections/06_analysis.tex`:174–189](sections/06_analysis.tex) | Tập con đánh giá × {F1, Prec, Rec}: Full vs IAA>0.8; hàng Δ với bootstrap p | **Đánh giá tập con nhất quán cao** → `results/iaa_ablation.json` mới |
| T10 | `tab:ood-transfer` | [`sections/06_analysis.tex`:194–207](sections/06_analysis.tex) | Dataset × {ConflictRAG F1, Baseline F1, Δ}: PubMedQA, ASQA (so với NLI Detector) | **Exp-D chuyển giao OOD (Overall F1 so với baseline NLI)** → `results/generalization.json` |
| T11 | (không nhãn) Ablation mở rộng | [`appendix/d_extended_tables.tex`:4–18](appendix/d_extended_tables.tex) | Component × Score: Full 63,6 / No protocol control 61,1 / No uncertainty 60,7 / No validation filter 61,3 | **Phần thừa từ template chung** — KHÔNG khớp tên thành phần T6. Xóa hoặc ánh xạ lại vào ablation thực → `results/ablations.json` |
| T12 | (không nhãn) Robustness mở rộng | [`appendix/d_extended_tables.tex`:20–34](appendix/d_extended_tables.tex) | Lát cắt × Score: Seen-style/Hard-lang/Noisy/Fresh | **Phần thừa từ template chung** — KHÔNG khớp điều kiện Fig 3. Ánh xạ lại vào các lát cắt robustness thực → `results/robustness.json` |
| T13 | `tab:qual-success` | [`appendix/e_qualitative_examples.tex`:10–24](appendix/e_qualitative_examples.tex) | Đầu vào định tính / đầu ra ConflictRAG / đầu ra baseline (3 trường hợp thành công) | **Bản ghi thực** lấy từ replay traces (phải hiển thị điểm dispersion/NLI thực) |
| T14 | `tab:qual-comparison` | [`appendix/e_qualitative_examples.tex`:38–50](appendix/e_qualitative_examples.tex) | Đầu vào định tính / ConflictRAG / Baseline tốt nhất (ví dụ CEO) | **Bản ghi thực** từ replay trace |

**Lưu ý về T11/T12:** các bảng phụ lục này mang văn bản `\syntheticlabel{}` theo nghĩa đen trong phần caption và dùng tên thành phần/lát cắt chung của simulator — chúng là các artifact giả tạo rõ ràng nhất và phải được xóa hoặc xây dựng lại để khớp với từ vựng T6/Fig-3 thực.

**Các tuyên bố số nội tuyến (không có trong bảng) cũng phải được tái suy ra:** abstract (`main.tex`:37: 2.394, 55,7%, 96,8%, 0,214, 0,198); phương pháp (`03_method.tex`:37 "579 examples", :49 "47 seconds / 100 examples"); thiết lập (`04_experimental_setup.tex`: kích thước thô 612/438/583/761, κ=0,71, θ*=0,55, θ_c=0,42, C=1,0); văn xuôi kết quả (`05_results.tex`:6–8, :50 Cohen's d>0,5); văn xuôi phân tích (`06_analysis.tex`: delta ablation 4,1/2,9/2,2/1,1, kết hợp 9,5, learning curve 51,2→81,2, single-passage 62,7→64,5, C-sweep 3,2/1,4, 2.000 items/hr, IAA 83,4 vs 81,2, OOD +10,8/+11,7).

---

## 3. Danh sách đầy đủ các HÌNH

Hai trong số bốn hình PDF (`fig1`, `fig4`) là các render matplotlib thực từ [`scripts/generate_figures.py`](scripts/generate_figures.py); `fig2` và `fig3` cũng được render ở đó. **Cả bốn đều nhúng số hardcoded trong Python, không đọc từ `results/`.** Ngoài ra còn có **bốn hình TikZ/pgfplots nội tuyến** được vẽ trực tiếp trong `.tex` với tọa độ hardcoded.

| `\label` | Hàm tạo / vị trí | Nội dung hình | Nguồn dữ liệu thực cần có |
|----------|-------------------|---------------|---------------------------|
| `fig:tradeoff` (`fig1_tradeoff.pdf`) | `fig1_tradeoff()` — [`generate_figures.py`:56–139](scripts/generate_figures.py) | (a) Đường θ-sweep Abstention-rate vs Answer F1 + các điểm baseline; (b) Abstention-rate vs Halluc-rate. θ-sweep là công thức tham số (`abstain_rates = 5+80*…`), không phải dữ liệu. | **θ-sweep của ConflictRAG** (θ=0,1→0,9 bước 0,05) ghi lại (abstain rate, Answer F1, halluc %) theo từng θ + các điểm baseline cố định → `results/threshold_sweep.json` mới |
| `fig:ablation` (`fig2_ablation.pdf`) | `fig2_ablation()` — [`generate_figures.py`:146–196](scripts/generate_figures.py) | Thanh nhóm: Answer F1 / Abstention F1 / Overall F1 cho full + 5 biến thể loại bỏ đặc trưng | **Exp-B ablation** (cùng lần chạy với T6) → `results/ablations.json` (mở rộng schema sang per-metric, không chỉ Overall F1) |
| `fig:robustness` (`fig3_robustness.pdf`) | `fig3_robustness()` — [`generate_figures.py`:203–257](scripts/generate_figures.py) | Overall F1 của 5 hệ thống trên các điều kiện In-dist / OOD / High-noise / Low-evidence | **Đánh giá theo điều kiện lát cắt** của tất cả hệ thống → `results/robustness.json` (mở rộng sang per-system per-condition) |
| `fig:calibration` (`fig4_calibration.pdf`) | `fig4_calibration()` — [`generate_figures.py`:264–338](scripts/generate_figures.py) | Sơ đồ độ tin cậy (predicted P(Answer) vs độ chính xác thực nghiệm, 5 bin) + Brier per system | **Calibration (hiệu chỉnh) trên tập held-out 266 ví dụ**: per-system per-bin (pred, empirical) + Brier → `results/calibration.json` (mở rộng sang per-system bins, thay schema 4-bucket) |
| `fig:main-bar` | TikZ nội tuyến — [`sections/05_results.tex`:97–129](sections/05_results.tex) | Thanh Answer F1 theo dataset cho No-filter / NLI Detector / ConflictRAG | Cùng dữ liệu với T3 → `results/per_dataset.json` (khuyến nghị chuyển tọa độ TikZ sang `\pgfplotstableread`) |
| `fig:scaling` | TikZ nội tuyến — [`sections/06_analysis.tex`:52–79](scripts/) | Overall F1 vs phần trăm tập huấn luyện (10/25/50/75/100%) cho ConflictRAG vs NLI Detector | **Thí nghiệm learning curve (đường cong học tập) với sub-sampling** → `results/learning_curve.json` mới |
| `fig:tradeoff-scatter` | TikZ nội tuyến — [`sections/06_analysis.tex`:81–114](sections/06_analysis.tex) | Scatter Overall F1 vs độ trễ cho 5 hệ thống | Cùng dữ liệu với T8 efficiency → `results/efficiency.json` |
| `fig:learning-curve` | TikZ nội tuyến — [`sections/06_analysis.tex`:146–169](sections/06_analysis.tex) | Overall F1 vs phần trăm huấn luyện (1/5/10/25/50/100%) — **lưu ý x-points khác và y-values khác (81,2 ở 100%) so với `fig:scaling` (68,8 ở 100%): mâu thuẫn nội bộ** | `results/learning_curve.json` — phải đối chiếu với `fig:scaling`; hai hình không đồng ý về F1 full-data (68,8 vs 81,2) |

**Cờ mâu thuẫn:** `fig:scaling` báo cáo Overall F1 full-data = 68,8 (khớp T6), nhưng `fig:learning-curve` báo cáo 81,2 (khớp với các số "F1 trong khoảng 80s" của `tab:iaa-ablation`/`tab:ood-transfer`). Bài báo dùng ngầm hai thang đo F1 khác nhau (thang ~68 trong T1/T6 và thang ~81 trong T9/T10/`fig:learning-curve`). Phải thống nhất về một metric được định nghĩa duy nhất trước khi nộp.

---

## 4. Sơ đồ phụ thuộc dữ liệu → artifact

```
RAW DATASETS (download)                 ConflictRAG PIPELINE                 ARTIFACTS
  WhoQA ───────────┐                    feature extraction:                 results/main_results.json ─► T1, abstract
  NoMIRACL-nonrel ─┤                      - dispersion (MiniLM emb)          results/per_dataset.json ──► T3, fig:main-bar
  ConflictQA ──────┼─► combined eval ──►  - NLI contradiction        ──┬──► results/dataset_matrix.csv ► T2
  FaithDial ───────┘   3,073 ex           - claim support              │    results/ablations.json ─────► T6, fig:ablation
  PubMedQA ─┐                             - lang/domain mismatch       ├──► results/robustness.json ────► T12, fig:robustness
  ASQA ─────┴─► OOD eval ──────────►     logistic policy (sklearn)     │    results/sensitivity.json ───► T7
                                          + θ threshold                ├──► results/threshold_sweep.json ► fig:tradeoff
  baselines: No-filter RAG, Naive,        + Llama-3.1-8B generator     │    results/calibration.json ────► fig:calibration
   Confidence, NLI Detector,              + GPT-4 Verifier (API)       ├──► results/efficiency.json ─────► T8, fig:tradeoff-scatter
   GPT-4-CoT, GPT-4o-Direct               + replay-trace logging       ├──► results/learning_curve.json ► fig:scaling, fig:learning-curve
                                                                       ├──► results/generalization.json ► T5, T10
                                          manual annotation (200 err)  ├──► results/error_taxonomy.csv ─► T4
                                          replay-trace sampling        └──► results/qualitative.json ───► T13, T14
                                          high-IAA subset eval ──────────►  results/iaa_ablation.json ──► T9
                                          claim ledger writer ───────────►  results/claim_ledger.csv ───► §7 audit
```

**Kết nối bị hỏng hiện tại (phải sửa):**
- `make_artifacts.py` ghi `results/*.json` từ `DATA` chung (Control/Cheap baseline/Strong API). **Không có bảng hay hình nào thực sự đọc các file JSON này.**
- `generate_figures.py` hardcode số của riêng nó; các hình bị tách rời khỏi `results/`.
- Tất cả bảng `.tex` hardcode số tách rời khỏi cả hai. → Ba nguồn sự thật độc lập, do đó có các mâu thuẫn được gắn cờ ở §3.

---

## 5. Các thí nghiệm cần chạy

> **Không có thư mục `configs/`.** Sản phẩm đầu tiên là *tạo* các YAML trong `configs/` mã hóa mọi thứ hiện đang ở dạng văn xuôi trong [`sections/04_experimental_setup.tex`](sections/04_experimental_setup.tex) và [`appendix/a_reproducibility.tex`](appendix/a_reproducibility.tex). Các file được đề xuất: `configs/datasets.yaml`, `configs/models.yaml`, `configs/policy.yaml`, `configs/eval.yaml`.

### Dataset cần thu thập (URL từ `appendix/a_reproducibility.tex`:10–11)
- **WhoQA** — `https://github.com/whichxkj/WhoQA` (toàn bộ eval split, ~612 thô; ~41% có xung đột). **Xác minh repo/URL này có tồn tại không — chưa được kiểm chứng và có thể là giả mạo.**
- **NoMIRACL nonrelevant** — HF `miracl/miracl`, tập con nonrelevant tiếng Anh (~438). Gold action = Abstain/Disclaimer.
- **ConflictQA** — `https://github.com/xiye17/conflictqa` (~583, các đoạn văn mâu thuẫn tổng hợp).
- **FaithDial** — `https://github.com/McGill-NLP/FaithDial` (~761 test).
- **PubMedQA** (`pubmedqa2019`) và **ASQA** (`asqa2022`) — chỉ cho chuyển giao OOD; tải xuống từ nguồn chính thức.
- Ghi lại giấy phép, phiên bản/commit, checksum SHA-256, và số lượng sau dedup của từng dataset vào `configs/datasets.yaml` + `results/dataset_matrix.csv` (T2).

### Mô hình (từ `03_method.tex` + `appendix/a_reproducibility.tex`)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- NLI: `cross-encoder/nli-MiniLM2-L6-H768`
- Extractive QA: `deepset/minilm-uncased-squad2`
- Generator: `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` qua `llama-cpp-python`, nhiệt độ 0
- Nhận dạng ngôn ngữ: `langdetect`
- Policy: `LogisticRegression` scikit-learn, one-vs-rest, L2, C được tinh chỉnh trên {0,01; 0,1; 1,0; 10,0}

### Các baseline cần triển khai (tên chính xác từ bài báo)
1. **No-filter RAG** — luôn `Answer` với Llama-3.1-8B.
2. **Naive Abstain** — luôn `Abstain`.
3. **Confidence Threshold** — log-prob chuẩn hóa của generator, abstain khi dưới θ_c (tinh chỉnh; bài báo nói 0,42).
4. **NLI Conflict Detector** — quy tắc nhị phân đơn trên cùng mô hình NLI (contradiction>0,6 ⇒ Disclose, ngược lại Answer).
5. **GPT-4 Verifier** — `gpt-4-turbo-2024-04-09`, prompt có cấu trúc (`appendix/b`:27–33). Oracle/chi phí cao nhất.
6. **GPT-4-CoT** và **GPT-4o-Direct** — xuất hiện trong T1/T8/hình nhưng **không được mô tả trong phần văn xuôi baseline**; hoặc triển khai và tài liệu hóa chúng, hoặc xóa chúng khỏi tất cả bảng/hình. (Quyết định trước khi chạy.)

### Giao thức huấn luyện / đánh giá
- Chia 30% dev / 70% test theo từng dataset; NoMIRACL chỉ eval (OOD). Không dùng dữ liệu test trong việc chọn ngưỡng (`04`:46–48).
- Khớp logistic policy trên dev kết hợp (~587 ví dụ); θ được quét 0,1–0,9 bước 0,05; điểm hoạt động θ* theo F1 tốt nhất trên dev (bài báo dùng 0,55).
- Chú thích gold action cho WhoQA/FaithDial qua rubric 4 chiều (`appendix/b`:8–14), 2 người chú thích trên mẫu 200 ví dụ chung → báo cáo Cohen's κ thực (bài báo tuyên bố 0,71); phần còn lại do một người chú thích với kiểm tra chéo 10%.

### Seeds, metric, ý nghĩa thống kê
- **Seeds:** ≥5 seeds cho tất cả các bước ngẫu nhiên (sklearn `random_state`, bất kỳ lấy mẫu nào, phân chia dev/test). Báo cáo mean ± 95% CI.
- **Metric (định nghĩa rõ ràng từng cái và ghim một thang "F1"):** Answer F1 (token-overlap trên các ví dụ gold-`Answer`); Hallucination rate (tỷ lệ phần phân tử không được hỗ trợ theo kiểu FActScore, tham chiếu Wikipedia); Abstention precision/recall (non-`Answer` là positive); Conflict-Disclosure F1 (các ví dụ gold-conflict); Brier score (posterior 4-lớp, tập calibration 266 ví dụ); Overall F1 (trung bình điều hòa của Answer F1 + Abstention F1 — **metric của phần phân tích hiện có hai khoảng giá trị mâu thuẫn**).
- **Ý nghĩa thống kê:** paired bootstrap, n=1000, p<0,05 (các dấu `†` trong T1); báo cáo Cohen's d (tuyên bố d>0,5). Tính toán và lưu trữ CI cho mọi ô được báo cáo.

### Lý thuyết / tuyên bố hình thức (`appendix/c_protocol_proofs.tex`)
Phụ lục đưa ra các mệnh đề ở cấp độ giao thức, **không phải** thực nghiệm:
- **Prop 1 (ledger completeness — tính đầy đủ của sổ cái):** mọi tuyên bố tổng hợp ánh xạ đến một table/figure id, mọi ô ánh xạ đến một tập replay-tuple. → **Nghĩa vụ thực nghiệm:** thực sự xây dựng kho replay-tuple `(x_i, y_is, z_is, c_is)` và `results/claim_ledger.csv` để ánh xạ này là thực, không chỉ được khẳng định.
- **Prop 2 (selective escalation cost bound — giới hạn chi phí leo thang có chọn lọc):** chi phí mỗi ví dụ = c_l + q·c_h. → **Nghĩa vụ thực nghiệm:** nếu có biến thể cascade/escalation nào được báo cáo, ghi lại chi phí rẻ/đắt per-example và tỷ lệ leo thang q; xác minh giới hạn cộng tính có đúng trong `results/efficiency.json`.
- "Nghĩa vụ đặc thù của ConflictRAG" yêu cầu mọi ví dụ phải lưu trữ điểm uncertainty (độ không chắc chắn) vô hướng + ngưỡng được chọn trước khi chấm điểm kết quả, và một bảng utility (hữu ích) rõ ràng per-action. → Triển khai replay logging đáp ứng điều này trước khi trích dẫn bất kỳ số expected-utility nào.

### Khả thi về tài nguyên (laptop / chỉ API)
Theo `03_method.tex`:49 và `appendix/a`:4: MacBook Pro M2, 16 GB, chỉ CPU, ~47 giây / 100 ví dụ. Toàn bộ 2.394 + OOD + ablation + θ-sweep × seeds vẫn dưới một ngày CPU trên laptop. Chỉ các baseline GPT-4* mới cần ngân sách API (xem §8).

---

## 6. Quy trình thay thế (từng bước)

### (a) Chạy thí nghiệm, dump metric vào `results/*.json` theo schema thực
Xây dựng một runner (ví dụ `scripts/run_experiments.py`) thực thi Exp-A…D + ablation + sweep và ghi **một JSON cho mỗi nhóm artifact** với schema được tài liệu hóa. Thay thế hoàn toàn schema `DATA` chung. Hình dạng `main_results.json` được đề xuất (lưu ý: file hiện tại là `rows:[{method,quality,risk,cost}]` — bỏ đi):
```json
{
  "eval_set": {"n": 2394, "datasets": ["WhoQA","NoMIRACL","ConflictQA","FaithDial"]},
  "seeds": [13,21,42,87,100],
  "systems": {
    "ConflictRAG": {"answer_f1": {"mean":..,"ci":[..,..]},
                    "halluc_pct": {...}, "abst_p": {...}, "abst_r": {...},
                    "conf_disc_f1": {...}, "brier": {...},
                    "sig_vs_baselines": {"answer_f1": {"p":..,"cohens_d":..}}},
    "No-filter RAG": {...}, "...": {}
  }
}
```
Thêm `results/result_schema.json` (hoặc README) ghi lại mọi key (prompt đề cập đến quy ước file này; nó chưa tồn tại ở đây — tạo mới).

### (b) Tái cấu trúc các script artifact để ĐỌC `results/*.json`
- **[`scripts/make_artifacts.py`](scripts/make_artifacts.py):** xóa dict `DATA` hardcoded (dòng 13). Script này chỉ nên chạy sau runner thực; chuyển nó thành bộ phát CSV/ledger **đọc** `results/main_results.json` v.v. Cụ thể, thay thế các lệnh ghi `main()` (dòng 58–78) để `claim_ledger.csv`, `dataset_matrix.csv`, `error_taxonomy.csv` được suy ra từ các file kết quả thực, và xóa bộ viết PDF giả `write_pdf`/`pdf_escape` (dòng 15–55, 79–102) — các hình đến từ `generate_figures.py`, không phải từ đường dẫn fake-PDF này.
- **[`scripts/generate_figures.py`](scripts/generate_figures.py):** thay thế mọi mảng hardcoded bằng `json.load` từ `results/`:
  - `fig1_tradeoff()` dòng 63–73 (`thetas`, `abstain_rates`, `answer_f1`, `halluc_rates` công thức tham số) và các tuple baseline dòng 89–94, 119–124 ← `results/threshold_sweep.json`.
  - `fig2_ablation()` dict `data` dòng 160–164 ← `results/ablations.json`.
  - `fig3_robustness()` `systems_data` dòng 213–219 ← `results/robustness.json`.
  - `fig4_calibration()` `calib_data` dòng 274–291 và `brier_scores` dòng 293–298 ← `results/calibration.json`.
  - Thêm thanh sai số (error bars) 95% CI vào fig2/fig3 (các caption hứa hẹn CI và ý nghĩa thống kê).

### (c) Tái tạo hình
Sau (a)/(b): `python scripts/make_artifacts.py && python scripts/generate_figures.py`. Xác nhận tất cả bốn PDF được tái tạo từ JSON và các hình TikZ nội tuyến (`fig:main-bar`, `fig:scaling`, `fig:tradeoff-scatter`, `fig:learning-curve`) được cập nhật từ JSON tương ứng (khuyến nghị chuyển đổi tọa độ hardcoded sang `\pgfplotstableread{results/...}` để chúng không thể bị sai lệch).

### (d) Thay thế các số nội tuyến hardcoded trong mọi bảng `.tex`
Cập nhật từng bảng, lý tưởng nhất là qua tự động tạo (xem khuyến nghị bên dưới):
- T1 `tab:main` ([`05_results.tex`:17–25](sections/05_results.tex)) — tất cả 8 hàng × 6 cột + dấu `†`.
- T2 `tab:dataset-matrix` ([`05_results.tex`:39–43](sections/05_results.tex)) — số lượng item, validation, κ.
- T3 `tab:per-dataset` ([`05_results.tex`:62–65](sections/05_results.tex)).
- T4 `tab:error-taxonomy` ([`05_results.tex`:79–83](sections/05_results.tex)).
- T5 `tab:generalization` ([`05_results.tex`:141–143](sections/05_results.tex)).
- T6 `tab:ablation` ([`06_analysis.tex`:14–22](sections/06_analysis.tex)).
- T7 `tab:hparam` ([`06_analysis.tex`:41–45](sections/06_analysis.tex)).
- T8 `tab:efficiency` ([`06_analysis.tex`:132–136](sections/06_analysis.tex)).
- T9 `tab:iaa-ablation` ([`06_analysis.tex`:181–184](sections/06_analysis.tex)).
- T10 `tab:ood-transfer` ([`06_analysis.tex`:201–202](sections/06_analysis.tex)).
- T11/T12 bảng phụ lục mở rộng ([`appendix/d_extended_tables.tex`:11–14, 27–30](appendix/d_extended_tables.tex)) — **xây dựng lại để khớp với từ vựng T6/Fig-3 thực hoặc xóa; xóa `\syntheticlabel{}` khỏi caption.**
- T13/T14 định tính ([`appendix/e_qualitative_examples.tex`:17–19, 45](appendix/e_qualitative_examples.tex)) — thay các bản ghi được bịa đặt bằng các ví dụ replay-trace thực bao gồm điểm dispersion/NLI thực.
- Cộng với tất cả các số văn xuôi nội tuyến được liệt kê ở cuối §2.

**Khuyến nghị (chống sai lệch):** tạo *nội dung* bảng từ `results/*.json` qua một script nhỏ phát ra các đoạn `tables/*.tex` được `\input{}` bởi các section (hoặc dùng `pgfplotstable`). Điều này đảm bảo bài báo == kết quả và loại bỏ loại lỗi chỉnh sửa thủ công mà Prop 1 trong `c_protocol_proofs.tex` cảnh báo.

### (e) Cập nhật `claim_ledger.csv` và tái xác minh mọi tuyên bố số
- Viết lại [`results/claim_ledger.csv`](results/claim_ledger.csv): một hàng mỗi tuyên bố số trong *bản thảo* với các cột `claim_id, value, table/figure_label, replay_tuple_set, source_script, status=verified`. Bỏ tất cả các hàng `fixed-seed simulator` / `Synthetic…`.
- Tái xác minh mọi số trong văn xuôi **và** trong các ghi chú phản hồi đánh giá trích dẫn kết quả: [`reviews/author_response.md`](reviews/author_response.md), [`reviews/round2_author_response.md`](reviews/round2_author_response.md), và `notes/review_round_0{1,2}.md`, `notes/revision_execution_report.md` — những tài liệu này trích dẫn các số tổng hợp và sẽ mâu thuẫn với kết quả thực.

### (f) Biên dịch lại
`latexmk -pdf main.tex` (có acl.sty). Xác nhận không còn dấu `\todoexp{}` / TODO đỏ và không còn chuỗi `Synthetic example result` nào trong PDF đã biên dịch.

---

## 7. Kiểm tra tính nhất quán & độ chặt chẽ

- **bài báo == kết quả == hình:** sau khi tự động tạo, so sánh từng ô bảng với nguồn `results/*.json` của nó; khẳng định đầu vào hình được tải từ cùng JSON. Thêm kiểm tra CI thất bại nếu bất kỳ token số `.tex` nào thiếu hàng ledger tương ứng.
- **Giải quyết hai thang F1:** thống nhất `fig:scaling` (68,8) vs `fig:learning-curve` (81,2) và "F1 trong khoảng 80s" của T9/T10 vs "F1 trong khoảng 60s" của T1/T6. Định nghĩa chính xác một metric Overall-F1 và tính toán lại.
- **CI + nhiều seeds:** ≥5 seeds, 95% CI trên mọi metric được báo cáo; các tỷ lệ phần trăm abstract/kết luận (55,7%, 96,8%) cũng cần CI.
- **Ý nghĩa thống kê:** paired bootstrap thực (n=1000) cho mọi ô `†`; Cohen's d thực cho tuyên bố "d>0,5"; p=0,008 thực cho IAA ablation.
- **Giấy phép dataset / checksum / kê khai:** xác minh URL WhoQA thực sự phân giải (đã được gắn cờ chưa xác minh); ghi lại giấy phép + SHA-256 + phiên bản mỗi dataset trong `configs/datasets.yaml`; đối chiếu số lượng T2 (428/438/408/533/587) với số lượng thực sau dedup và với tổng 2.394 của abstract (các phần hiện tại cộng lại 1.807 test + 587 dev = 2.394 — xác nhận phép tính này tồn tại qua các phân chia thực).
- **GPT-4-CoT / GPT-4o-Direct:** tài liệu hóa hoặc xóa — hiện tại chúng xuất hiện trong bảng/hình mà không có mô tả thiết lập.
- **Xóa mọi dấu tổng hợp:** xóa các lần dùng `\syntheticlabel{}` (`appendix/d`, `appendix/c`:2), các key `synthetic_label` trong tất cả `results/*.json` và `make_artifacts.py`, trạng thái `Synthetic…` trong `claim_ledger.csv`, và phần "Synthetic Status" của `notes/source_note.md` / `notes/completion_note.md` khi đã có dữ liệu thực.
- **Kiểm toán replay-trace (Prop 1):** xác nhận chuỗi claim ledger → table/figure → replay-tuple là thực và đầy đủ, thỏa mãn `appendix/c_protocol_proofs.tex`.

---

## 8. Ước tính tài nguyên

Được thiết kế là bài báo **rẻ, chỉ dùng CPU** (đề xuất: "vừa phải nhưng có thể quản lý được trên laptop," thời gian 12–15 tuần).

| Thành phần | Ước tính |
|------------|----------|
| Tính toán CPU | ConflictRAG + baseline CPU (No-filter, Naive, Confidence, NLI Detector) trên ~2.394 + OOD + ablation + θ-sweep × 5 seeds: ở ~0,47 giây/ví dụ (47 giây/100 của bài báo), khoảng **5–15 giờ CPU** trên M2. **Không cần GPU.** |
| Sinh văn bản Llama-3.1-8B (Q4) | CPU qua llama.cpp; bước chậm nhưng vẫn khả thi qua đêm trên laptop. |
| **Chi phí API (GPT-4 Verifier / GPT-4-CoT / GPT-4o-Direct)** | Mục duy nhất phải trả tiền. ~2.394 + các ví dụ OOD × tối đa 3 biến thể GPT-4 × prompt có cấu trúc. Khoảng **$50–200** tùy thuộc vào biến thể GPT nào được giữ lại và độ dài ngữ cảnh. Chỉ giữ GPT-4 Verifier để tối thiểu hóa chi phí. |
| Công sức chú thích | Nhãn policy 4 chiều cho WhoQA + FaithDial dev (~vài trăm ví dụ) bởi 2 tác giả-người chú thích cho mẫu IAA 200 ví dụ + phần còn lại; cộng thêm phân loại lỗi thủ công 200 lỗi (T4) và chọn lọc định tính (T13/T14). Ước tính **2–4 người-ngày**. |
| Tổng thể | **Thấp** (chỉ CPU + API khiêm tốn + chú thích của tác giả). |

---

## 9. Rủi ro & giải pháp (theo thứ tự mức độ nghiêm trọng)

1. **Kết quả tổng hợp là giả tạo và hiện không nhất quán trên ba lớp.** *Giải pháp:* toàn bộ pipeline §6; nguồn sự thật duy nhất (`results/*.json`) cung cấp cho các bảng/hình được tự động tạo.
2. **Các mâu thuẫn số nội bộ đã có trong bản thảo** (hai thang F1; bất khớp từ vựng T11/T12; GPT-4-CoT/GPT-4o-Direct chưa được tài liệu hóa). *Giải pháp:* thống nhất §7 trước khi chạy lại để schema mục tiêu không còn mơ hồ.
3. **Xuất xứ dataset/trích dẫn chưa được xác minh** — URL GitHub WhoQA và một số trích dẫn `whoqa2025`/`conflictqa2023` có thể là giả mạo. *Giải pháp:* giải quyết từng URL; chạy kiểm toán tính toàn vẹn trích dẫn; bỏ hoặc thay thế bất cứ thứ gì không phân giải được.
4. **Các tuyên bố chính có thể không đứng vững qua các lần chạy thực** (giảm 55,7%, đánh bại GPT-4o-Direct, OOD +10–12 điểm là đáng ngờ về mức độ sạch sẽ). *Giải pháp:* chạy trước, viết tuyên bố sau; để CI thực quyết định văn xuôi; làm yếu tuyên bố nếu cần.
5. **Không có `configs/` và các script bị ngắt kết nối** làm cho việc tái tạo không thể thực hiện được theo cách hiện tại. *Giải pháp:* soạn `configs/*.yaml`, runner `run_experiments.py`, và `result_schema.json`; kết nối script với JSON (§6b).
6. **Tính chủ quan trong chú thích** ở ranh giới Disclaimer/Disclose (κ=0,71 ngụ ý ~14–15% bị tranh chấp) — đã được thừa nhận trong Limitations. *Giải pháp:* phán xét của chuyên gia cho các trường hợp ranh giới; báo cáo κ thực; ablation tập con IAA (T9) chỉ có ý nghĩa với nhãn thực.
7. **Tính phi xác định / trôi dạt mô hình API** cho các baseline GPT-4. *Giải pháp:* ghim `gpt-4-turbo-2024-04-09`, nhiệt độ 0, cache phản hồi vào replay traces (thỏa mãn Prop 1).
8. **Suy giảm trên single-passage / low-evidence** (dispersion không xác định) là giới hạn phương pháp thực sự. *Giải pháp:* triển khai fallback dispersion nội bộ cấp câu mà bài báo tuyên bố (+1,8 điểm) và đo lường trung thực.

---

## 10. Danh sách kiểm tra thực thi

- [ ] Tạo `configs/{datasets,models,policy,eval}.yaml` từ văn xuôi trong `04_experimental_setup.tex` + `appendix/a_reproducibility.tex`.
- [ ] Tải xuống + xác minh WhoQA, NoMIRACL-nonrel, ConflictQA, FaithDial, PubMedQA, ASQA; ghi lại giấy phép + phiên bản + SHA-256 + số lượng sau dedup (→ T2).
- [ ] Quyết định số phận của GPT-4-CoT / GPT-4o-Direct (triển khai+tài liệu hóa, hoặc xóa khỏi T1/T8/hình).
- [ ] Triển khai các bộ trích xuất đặc trưng (dispersion, NLI, claim support, lang/domain) + logistic policy + sinh 4-action + generator Llama-3.1-8B-Q4 + replay-trace logging (thỏa mãn Prop 1 trong `appendix/c`).
- [ ] Triển khai tất cả baseline (No-filter, Naive, Confidence, NLI Detector, GPT-4 Verifier [+ tùy chọn các biến thể GPT-4]).
- [ ] Chú thích nhãn policy 4 chiều cho WhoQA/FaithDial; tính Cohen's κ thực; xây dựng phân chia dev/test (30/70, NoMIRACL chỉ eval).
- [ ] Chạy Exp-A (chính, ≥5 seeds, bootstrap CI + Cohen's d) → `results/main_results.json`, `per_dataset.json`, `efficiency.json`.
- [ ] Chạy Exp-B ablation (loại bỏ đặc trưng đơn + kết hợp) → `results/ablations.json`.
- [ ] Chạy Exp-C sensitivity (quét τ_s; quét C) → `results/sensitivity.json`.
- [ ] Chạy Exp-D chuyển giao OOD (PubMedQA, ASQA, zero-shot) → `results/generalization.json`.
- [ ] Chạy θ-sweep, learning-curve sub-sampling, robustness-slice, calibration (266 ví dụ), high-IAA-subset → JSON tương ứng.
- [ ] Phân loại lỗi thủ công 200 lỗi → `results/error_taxonomy.csv` (T4); lấy các bản ghi định tính thực (T13/T14).
- [ ] Định nghĩa **một** metric Overall-F1; đối chiếu `fig:scaling` vs `fig:learning-curve` và thang T1/T6 vs T9/T10.
- [ ] Viết `results/result_schema.json`; tái cấu trúc `make_artifacts.py` để đọc JSON (xóa `DATA`, bỏ bộ viết fake-PDF).
- [ ] Tái cấu trúc `generate_figures.py` để tải tất cả mảng từ `results/*.json`; thêm thanh sai số CI vào fig2/fig3.
- [ ] Chuyển đổi các hình TikZ nội tuyến và tất cả 14 bảng sang các đoạn `\input{}` được tự động tạo từ `results/` (chống sai lệch).
- [ ] Tái tạo hình; biên dịch lại `main.tex`; xác nhận không còn chuỗi `Synthetic`/`\syntheticlabel`/`\todoexp` nào còn lại.
- [ ] Xây dựng lại `claim_ledger.csv` (các hàng đã xác minh); cập nhật/thay thế các số trong `reviews/*author_response*.md` và `notes/review_round_*.md` trích dẫn kết quả tổng hợp.
- [ ] Kiểm toán tính toàn vẹn trích dẫn (đặc biệt URL WhoQA + metadata `whoqa2025`/`conflictqa2023`).
- [ ] Cổng nhất quán cuối cùng: bài báo == kết quả == hình == ledger.

---

*Chỉ là kế hoạch — không có thí nghiệm nào được chạy, không có hình nào được tái tạo, không có file nào khác ngoài `PLAN.md` này được sửa đổi.*
