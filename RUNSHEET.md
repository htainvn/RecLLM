# Runsheet — SigLLM + SeLLa port (branch `feat/SeLLaRecQformer`)

Bám theo notebook `Copy_of_SigLLM_new_ipynb_SeLLaRec (1).ipynb`, cập nhật cho code mới.

## ⚠️ Thay đổi thứ tự so với notebook cũ

**Distill bank PHẢI chạy TRƯỚC khi train MF.** MF giờ tiêu thụ
`item_llm_emb.pt` (`run.rec_baseline.item_llm_emb_path`) cho SeLLa Step-2
alignment — thiếu file này `train_rec_baseline.py` sẽ raise
`FileNotFoundError`. Notebook cũ chạy MF (cell "Stage 0") trước distill —
thứ tự đó không còn đúng.

Thứ tự mới:

```
prep data → pull LLM → distill 2 banks (base LLM) → MF(align) →
stage 1 → stage 2 → step 1 (LoRA) → step 2 (CIE, direct-id + MF unfrozen) → eval
```

## 🔀 Từ giờ có HAI nhánh

| | Nhánh A — QFormer soft-token (cũ) | Nhánh B — **SeLLa-gated** (mới, §4B) |
|---|---|---|
| arch | `mini_gpt4rec_v2` | `sella_gated_rec_llm` |
| vị trí soft token | 3 (SeLLa) **+ 16 mới** (`<UserProfile>` 8 + `<TargetItemID>` 8) | **đúng 3** (`<UserID>`, `<ItemID>`, `<Warm_ID>`) |
| QFormer vào LLM bằng | token mới | **cộng có cổng vào `<UserID>`**: `e_user += g·delta` |
| loss | CE 2 chiều tại 1 vị trí (+ BPR phụ) | **LM CE toàn chuỗi** (SeLLa) |
| eval | softmax Yes/No | **giống hệt** |
| QFormer | 768 / 4 layer / CAF 2 / 8 query / ~76M + BERT | **256 / 2 / 1 / 2 / 4.35M, không BERT** |
| stage cần chạy | 1, 2, step 1, step 2 | **chỉ step 1 (LoRA) rồi §4B** |

Nhánh B **không đụng gì** vào nhánh A: model/pipeline/prompt riêng, chỉ thêm
block `model.sella_gated` + `run.sella_gated_step3` vào config. Chạy song song
được, so sánh trực tiếp được.

MF đổi geometry so với checkpoint cũ (train lại với InfoNCE), nên **Stage 1/2
bắt buộc train lại trên MF mới** — không tái dùng `qformer_stage1/best`,
`qformer_stage2/best` cũ. (Ngược lại: các flag mới `candidate_fusion` /
`direct_id_tokens` KHÔNG phá tương thích checkpoint — fuse layers zero-init
load `strict=False`, nên nếu chỉ muốn bật chúng trên MF cũ thì không cần rerun
Stage 1/2; nhưng khi đó không có trans warm-start và `<Warm_ID>`/InfoNCE
alignment.)

---

## 0. Setup (như notebook, không đổi)

```bash
# clone + checkout
git clone https://github.com/htainvn/RecLLM.git && mv /content/RecLLM /content/SigLLM
cd /content/SigLLM && git checkout feat/SeLLaRecQformer && git pull origin feat/SeLLaRecQformer

# conda env "sigllm" (restore từ R2 cache như cell sẵn có), rồi:
source /usr/local/etc/profile.d/conda.sh && conda activate sigllm
export TOKENIZERS_PARALLELISM=false
```

## 1. Data prep (không đổi)

```bash
python /content/SigLLM/src/sigllm/datasets/data_preprocessing.py
python /content/SigLLM/src/sigllm/datasets/preprocess_test_cold_warm.py
python /content/SigLLM/src/sigllm/pipelines/multimodal/build_qformer_dataset.py \
    --cfg-path /content/SigLLM/configs/config.yaml
```

## 2. Pull LLM (không đổi)

```bash
python -c "from sigllm.pipelines.llm.pull_llm_model import pull_model, smoke_test_model; \
d=pull_model('Qwen/Qwen2-7B','/content/SigLLM/ckpt/llm/qwen2-7b-base'); smoke_test_model(d)"
```

## 3. Distill 2 banks — CHẠY TRƯỚC MF (mới)

```bash
cd /content/SigLLM

# Bank A — input space (L_llm target của Stage 1)
python -m sigllm.pipelines.multimodal.distill_item_llm_embeddings \
  --llm-model /content/SigLLM/ckpt/llm/qwen2-7b-base \
  --data-pkl data/processed/ml-1m/{train,valid,test}_ood2.pkl \
  --item-num 3256 --space input \
  --output /content/SigLLM/data/processed/ml-1m/item_llm_emb_input.pt

# Bank B — last-hidden space (sem_source + <Warm_ID> + MF alignment MỚI)
python -m sigllm.pipelines.multimodal.distill_item_llm_embeddings \
  --llm-model /content/SigLLM/ckpt/llm/qwen2-7b-base \
  --data-pkl data/processed/ml-1m/{train,valid,test}_ood2.pkl \
  --item-num 3256 --space last_hidden \
  --output /content/SigLLM/data/processed/ml-1m/item_llm_emb.pt
```

Check coverage (kỳ vọng 3255/3256, id 0 là padding):

```python
import torch
for f in ["item_llm_emb_input.pt", "item_llm_emb.pt"]:
    b = torch.load(f"/content/SigLLM/data/processed/ml-1m/{f}", map_location="cpu")
    e = b["item_llm_emb"] if isinstance(b, dict) else b
    print(f, tuple(e.shape), "covered", (e.norm(dim=-1) > 0).sum().item())
# item_llm_emb.pt phải có dim cuối = 3584 (hidden của Qwen2-7B) —
# điều kiện để trans_2 warm-start được id_proj ở step 2/3.
```

## 4. Stage 0 — MF với SeLLa Step-2 alignment (đổi hành vi)

```bash
python /content/SigLLM/src/sigllm/pipelines/rec/train_rec_baseline.py
```

- Log phải có: `SeLLa Step-2 alignment ACTIVE | loss = BCE + 1.0 * InfoNCE(tau=0.2)`.
- Checkpoint `/content/SigLLM/ckpt/mf/mf_model.pth` giờ mang thêm
  `item_embedding_llm.*`, `trans_1.*`, `trans_2.*` (~+15M params).
- So sánh AUC/uAUC với run MF thuần cũ (~0.7077 uAUC): InfoNCE là loss phụ,
  AUC không được tụt sâu; tụt >1 điểm → giảm `run.rec_baseline.align_weight`
  (0.5, 0.2) qua `--options`.
- Chạy MF THUẦN (đối chứng): thêm
  `--options run.rec_baseline.item_llm_emb_path=null`.

## 4B. Nhánh SeLLa-gated — BỎ Stage 1/2, train QFormer chung ở một step

Chỉ cần: §1 → §2 → §3 → §4 → **§4B.1** → **§4B.2**. Không chạy §5, §6, §7, §8.

> ⚠️ **KHÔNG dùng `train_qformer_stage3_step1_lora` cho nhánh này.** Script đó
> dựng `QRecLLM`, và `QRecLLM.__init__` load `model.qformer_config.qformer_ckpt`
> + `llm_proj_ckpt` **vô điều kiện** — cả hai là output của Stage 2, thứ nhánh
> này không chạy. Kết quả:
> `FileNotFoundError: .../qformer_stage2_qwen2/qformer_stage2_best_qformer.pth`.
> Ngoài ra prompt của nó (`no_hist_text`) có `<UserProfile>`/`<TargetItemID>`,
> nên không có ckpt Stage 2 thì LoRA sẽ được train chống lại một QFormer **vừa
> ngẫu nhiên vừa đóng băng** — tệ hơn cả không có.

### §4B.1 — step 1: LoRA text-only (đúng TALLRec/SeLLa step 1)

```bash
cd /content/SigLLM
python -m sigllm.pipelines.multimodal.train_sella_gated_step1_lora \
    --cfg-path configs/config.yaml
```

Script này tự set `qformer_ckpt=llm_proj_ckpt="not_have"` (sentinel mà
`_init_qformer`/`_init_projection` đã hiểu là "bỏ qua"), `warm_token=False`,
`sem_source=False`, tắt `ranking_loss`/`align_rank_loss`/`user_grouped_batch`,
và **hard-error nếu prompt không phải text-only**. Prompt text-only chỉ có
`<ItemTitleList>` + `<TargetItemTitle>` → `get_placeholder_order()` trả `[]` →
không có injection nào, QFormer ngẫu nhiên không bao giờ bị đọc. Kênh cộng tác
xuất hiện lần đầu ở §4B.2, đúng như SeLLa.

Log phải thấy: `Prompt check OK | 3 text-only prompt line(s), no soft-token
placeholders`. Ghi vào `ckpt/sella_gated_step1_lora_qwen2/` (khác thư mục step 1
của nhánh A, nên hai nhánh không đè nhau).

**Rủi ro đã biết, nói thẳng:** config của chính repo này ghi lại rằng step 1
text-only từng làm LoRA học trả lời bằng title rồi phớt lờ kênh soft token
("Flat Step-2 uAUC"). SeLLa vẫn làm đúng thế này, và nhánh B khác ở chỗ có ý
nghĩa: tín hiệu vào bằng `<UserID>`/`<ItemID>` **trực tiếp** (vector MF chưa nén,
`id_proj` warm-start từ `trans_1/trans_2` của MF) chứ không phải kênh nén 8
token, và `id_proj`/`warm_proj`/MF vẫn train tiếp ở step 3. Nếu muốn đối chứng:
`--options run.sella_gated_step3.ckpt_from=step1` để lấy LoRA của nhánh A (đã
được adapt cùng soft token) — nhưng cái đó cần Stage 1/2 đã chạy.

### §4B.2 — step 3: train QFormer chung, cổng zero-init

```bash
python -m sigllm.pipelines.multimodal.train_sella_gated_step3 \
    --cfg-path configs/config.yaml
```

`ckpt_from` mặc định là `sella_step1`. Các giá trị khác: `step1`/`step2` (ckpt
nhánh A), `self` (resume / eval-only), `none` (LoRA lạnh — thí nghiệm khác).

### Tại sao bỏ được Stage 1/2

Cổng `g` zero-init làm việc mà Stage 1/2 vốn phải làm: ở `g = 0` model **đúng
bằng SeLLa từng bit** (số hạng cộng thêm là `+ 0` thật, không phải nhiễu nhỏ),
nên một module 4.35M khởi tạo ngẫu nhiên train chung là an toàn — nó không thể
làm run tệ hơn baseline. Đổi lại bỏ được hai giai đoạn đang mang lỗi đã ghi
nhận: Stage 1 InfoNCE gain ở mức ngẫu nhiên, Stage 2 bị `<ItemTitleList>` lấn át.

### ⚠️ Ngân sách train — biến quan trọng nhất, đọc từ `train_sella.sh`

SeLLa step 3: `num_train_epochs 1`, batch 5 × accum 30 × 2 GPU = **effective 300**,
lr **2e-4**, cosine, `warmup_ratio 0`, `weight_decay 0.01`. Trên ml-1m (32615
row) đó là **~109 optimizer update cho TOÀN BỘ run**.

| | updates | vs SeLLa |
|---|---|---|
| SeLLa step 3 (cả run) | 109 | 1x |
| run đầu, epoch 0 → AUC 0.752 / uAUC 0.698 | 800 | 7.4x |
| run đầu, epoch 1 → 0.636 / 0.618 | 1600 | 14.7x |
| `max_epoch: 30` (cấu hình cũ) | 24 000 | 221x |

Collapse ở epoch 1 **không** phải objective sai — là **over-train** dưới một
objective chỉ an toàn ở ngân sách ngắn của SeLLa. Config giờ đặt
`accum_grad_iters: 38`, `batch_size_train: 8` (effective 304),
`iters_per_epoch: 836`, `max_epoch: 5` → **110 update, 5 điểm eval**. Pipeline in
ra dòng `Budget | ... Ratio vs SeLLa: 1.0x` và **warn nếu vượt 400 update**.

Muốn train dài hơn SeLLa: tăng `max_epoch` **và** đặt
`model.sella_gated.lm_loss_scope=answer` — `full` không sống nổi ở đó.

### 6 dòng log PHẢI đọc

1. `History Q-Former built | GatedHistoryQFormer(...params=4.35M)` — xác nhận đã
   cắt: `d_model=256, layers=2, queries=2, cross_attention_frequency=1,
   text_branch=off`. Nếu thấy `params=...76M` hoặc `text_branch` khác `off` →
   đang chạy nhánh A, dừng.
2. `Freeze policy | base LLM frozen; LoRA frozen (SeLLa parity); trainable
   modules: id_proj + warm_proj + history_qformer; MF TRAINABLE` — dòng này liệt
   kê **đúng các module thật sự tồn tại**, dùng nó để biết đang chạy arm nào.
3. `Prompt injection stats | placeholders=['<UserID>', '<ItemID>', '<Warm_ID>'],
   slots_per_sample=3` — phải là **3**, không phải 18/19. Sai số slot là
   `RuntimeError` chứ không im lặng.
4. `LM loss | scope=full, supervised_positions=N/M` — objective SeLLa.
5. **`qformer gate (step N) | gate=..., delta_norm=..., delta_rel=...`** ← số
   phải báo cáo. `gate` là đóng góp của module; `delta_rel = ‖g·delta‖/‖e_user‖`
   là bản không phụ thuộc scale. Cả hai dính 0 sau vài trăm step → kết luận
   trung thực là "QFormer không đóng góp". `delta_rel` vượt ~1 → số hạng cộng
   đang lấn át token user, hạ `init_lr` chứ đừng coi là thắng.
6. `MF drift (step N)` — drift tăng mà val uAUC đi ngang → hạ
   `run.sella_gated_step3.rec_lr_scale`.

Cuối run: `FINAL gated Q-Former contribution | gate=..., delta_rel=...`.

### Epoch đầu phải ≈ SeLLa

Vì `g = 0` lúc bắt đầu, val uAUC epoch 0 phải xấp xỉ step-1 LoRA. Tụt mạnh ngay
epoch 0 là **lỗi**, không phải noise — kiểm tra `slots_per_sample` và
`Prompt example` trước khi train tiếp.

### A/B và ablation (mỗi cái một run)

```bash
B=sigllm.pipelines.multimodal.train_sella_gated_step3
C="--cfg-path configs/config.yaml --options"

# ĐỐI CHỨNG SeLLa: tắt hẳn QFormer, mọi thứ khác y nguyên
python -m $B $C model.sella_gated.use_qformer=False

# bỏ slot ngữ nghĩa (memory chỉ còn CF + slot user×target)
python -m $B $C model.sella_gated.use_sem=False

# bỏ slot tương tác user×target
python -m $B $C model.sella_gated.use_user_slot=False

# bật fusion theo từng slot lịch sử (DIN nhân, zero-init) — thử khi gate tăng
# mà uAUC không tăng
python -m $B $C model.sella_gated.hist_target_fusion=True

# 1 query thay vì 2
python -m $B $C model.sella_gated.num_queries=1

# đóng băng MF (tách đóng góp của riêng QFormer)
python -m $B $C run.sella_gated_step3.freeze_rec=True

# LoRA train chung (KHÔNG phải SeLLa parity — thí nghiệm khác)
python -m $B $C model.sella_gated.lora_trainable=True

```

### Eval một checkpoint đã train

`run.evaluate=True` làm `runner.train()` bỏ qua vòng train và chạy thẳng
`evaluate(cur_epoch="best", skip_reload=True)` trên **cả** `test_splits`
(`test`, `test_warm`, `test_cold`). `skip_reload=True` nghĩa là nó **không**
nạp lại `checkpoint_best.pth` từ `output_dir` mà dùng đúng model vừa build —
tức `lora_ckpt` (LoRA từ step 1) + `ckpt` (phần cộng tác từ step 3).

```bash
B=sigllm.pipelines.multimodal.train_sella_gated_step3
C="--cfg-path configs/config.yaml --options"

# 1) số chính thức của checkpoint đã train
python -m $B $C run.evaluate=True run.sella_gated_step3.ckpt_from=self

# 2) CÙNG checkpoint, zero số hạng cộng của QFormer -> đo đóng góp trực tiếp
python -m $B $C run.evaluate=True run.sella_gated_step3.ckpt_from=self \
                model.sella_gated.ablate_qformer=True
```

Hiệu số uAUC giữa (1) và (2) là đóng góp của QFormer đo trên cùng một bộ trọng
số — độc lập với `gate`, và là con số đáng tin hơn để báo cáo.

> ⚠️ **Đừng override `run.sella_gated_step3.output_dir` khi eval.**
> `ckpt_from=self` phân giải checkpoint từ **chính key đó**
> (`<output_dir>/<slug>/checkpoint_best.pth`), nên đổi output_dir là trỏ luôn
> sang chỗ không có checkpoint. Eval sẽ ghi thêm vào `log.txt` của thư mục
> train — chấp nhận được, đừng đổi đường dẫn để tránh.

### ⚠️ Bẫy `--options`: cái gì override được, cái gì không

`Config.__init__` merge `--options` **trước**, rồi `main()` mới gọi
`apply_overrides()`. Nên mọi key mà `apply_overrides` **ghi** sẽ bị đè lại và
`--options` của bạn im lặng không có tác dụng:

| Override cái này thì VÔ HIỆU | Dùng cái này thay thế |
|---|---|
| `model.freeze_rec` | `run.sella_gated_step3.freeze_rec` |
| `model.prompt_path` | `run.sella_gated_step3.prompt_path` |
| `model.ckpt` / `model.lora_ckpt` | `run.sella_gated_step3.ckpt_from` / `.lora_from` |
| `run.init_lr`, `run.min_lr`, `run.max_epoch` | `run.sella_gated_step3.<cùng tên>` |
| `run.batch_size_train/eval`, `run.iters_per_epoch` | `run.sella_gated_step3.<cùng tên>` |
| `run.rec_lr_scale`, `run.rec_weight_decay` | `run.sella_gated_step3.<cùng tên>` |
| `run.output_dir` | `run.sella_gated_step3.output_dir` |

Override trực tiếp được (pipeline không ghi vào): mọi key `model.sella_gated.*`,
`run.evaluate`, `datasets.movie_ood.*`.

### LoRA phải được nạp — giờ là lỗi, không phải cảnh báo

`RunnerBase._save_checkpoint` xoá mọi tensor `requires_grad=False`, và nhánh này
đóng băng LoRA (`lora_trainable: false`, đúng SeLLa). Nên **checkpoint do step 3
ghi ra không chứa key `lora_*` nào** (đo được: 69 tensor, 0 lora). Nạp một mình
nó thì `lora_B` vẫn là zeros → adapter thành identity → bạn eval trên base model
chưa adapt và chỉ thấy điểm thấp hơn, không có lỗi nào.

Vì vậy `ckpt_from=self` giờ tự lấy LoRA từ `run.sella_gated_step3.lora_from`
(mặc định `sella_step1`), và model **raise** nếu LoRA bị đóng băng mà không có
tensor `lora_*` nào được nạp. Muốn cố ý eval base model thì `ckpt_from=none`
(nó set `allow_cold_lora`). Log xác nhận:

```
LoRA restored from | .../sella_gated_step1_lora_qwen2/qwen2-7b-base/checkpoint_best.pth (N lora_* tensors)
LoRA check OK      | N lora_* tensors loaded from ...; frozen for training (SeLLa parity)
```

Không thấy hai dòng này ⇒ run không hợp lệ.

### OOM

LM CE toàn chuỗi trên vocab 151936 tốn ~0.5 GB/sample transient (nhánh A chỉ
chạm 1 vị trí). Thứ tự xử lý:

```bash
--options model.sella_gated.lm_loss_scope=answer        # 1 vị trí/sample, rẻ nhất
--options run.sella_gated_step3.batch_size_train=4
--options datasets.movie_ood.max_history_length=10      # prompt ngắn lại
```

`lm_loss_scope` còn giá trị `full_no_soft` (CE toàn chuỗi nhưng mask 3 vị trí
soft token). Đáng thử: SeLLa có 3 special id **khác nhau**, còn ở đây cả 3 slot
dùng **cùng một** reserved id, nên `full` đang tốn gradient cho việc "đoán ra
token placeholder".

### Backup

```bash
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/sella_gated_step3_qwen2 \
    s3://sigllm/sella_gated_step3/<tag>/
```

---

## 5. Stage 1 — Representation pretraining (lệnh không đổi, MF mới)

```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage1_representation \
    --cfg-path configs/config.yaml
```

Vẫn theo dõi các mốc cũ: `val_gain_repr` peak (~ep16 ở run trước), sem_off vs
sem_on, "zero-init paths" (user_proj phải rời 0).

## 6. Stage 2 — Generative pretraining (không đổi)

```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage2_generative \
    --cfg-path configs/config.yaml
```

## 7. Step 1 — LoRA (lệnh không đổi)

```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step1_lora \
    --cfg-path configs/config.yaml
```

- Prompt mặc định của step 1 vẫn là `no_hist_text` (không có
  `<UserID>/<ItemID>` — direct-id chỉ vào từ step 2; LoRA không cần thấy nó
  vì id_proj bị freeze ở step 1).
- Log mới sẽ hiện: `DIRECT ID TOKENS ACTIVE`, `CANDIDATE FUSION ACTIVE`,
  `Direct ID proj (shared) warm-started from MF trans_1/trans_2` — dòng cuối
  là bằng chứng SeLLa `pretrained_with_small=True` đã ăn. Nếu ra
  `fresh init` → MF ckpt không có trans (đã train MF thuần?) — dừng, kiểm tra
  bước 4.

## 8. Step 2 — CIE (đổi nhiều: direct-id prompt + MF unfrozen)

```bash
python /content/SigLLM/src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py \
    --cfg-path /content/SigLLM/configs/config.yaml
```

Mặc định mới (từ `run.qformer_stage3_step2` trong config):
- `prompt_path = prompts/qformer_prompt_movie_direct_id.txt` (thêm `<UserID>`,
  `<ItemID>`);
- `freeze_rec: false` — MF vào param group riêng, lr ≈ 1e-4
  (`rec_lr_scale: 3.33` × init_lr 3e-5) vs QFormer 3e-5.

Checks bắt buộc trong log vài trăm step đầu:
1. `Prompt injection stats`: `sample_soft_tokens = 18` (8 UserProfile + 8
   Target + 1 UserID + 1 ItemID; warm_token off) và `sample_unk_slots` khớp.
2. Epoch đầu: val uAUC phải ≈ điểm cuối của step 1 (warm start no-op —
   fuse_cf zero-init, user slot = copy của target slot). Tụt mạnh ngay
   epoch 0 → báo lỗi warm start, dừng và kiểm tra.
3. `MF drift (step N)`: drift tăng dần là chủ đích; drift tăng mà val uAUC
   đi ngang/giảm → hạ `run.qformer_stage3_step2.rec_lr_scale` (1.0 → 0.3).

Ablation tách đóng góp (mỗi cái một run, so với default):

```bash
# tắt direct-id (giữ fusion):
  --options model.direct_id_tokens=False \
            run.qformer_stage3_step2.prompt_path=/content/SigLLM/prompts/qformer_prompt_movie.txt
# tắt candidate fusion (giữ direct-id):
  --options model.qformer_config.candidate_fusion=False
# đóng băng MF như CoLLM gốc:
  --options run.qformer_stage3_step2.freeze_rec=True
```

## 9. Eval (không đổi lệnh)

```bash
python -m sigllm.pipelines.multimodal.eval_test --cfg-path configs/config.yaml --step 2
# ablate soft tokens lúc inference:
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options model.ablate_soft_tokens=True run.evaluate=True
```

Lưu ý: step-2 checkpoint giờ MANG `rec_encoder.*` (MF đã train tiếp) —
`from_config` tự giữ MF từ checkpoint, KHÔNG reload `mf_model.pth` đè lên
(log: `Kept MF weights from the checkpoint`). Thấy log
`Restored pretrained MF` khi eval step-2 ckpt là sai — báo lỗi.

## 10. (Tuỳ chọn) Vòng SeLLa-parity — bank từ LLM đã finetune

Đúng SeLLa hơn (bank distill từ LLM-sau-LoRA thay vì base), giá là một vòng
train lại:

```bash
# sau step 1, distill lại Bank B từ LoRA ckpt (KHÔNG còn cần mf_model.pth):
python -m sigllm.pipelines.multimodal.distill_item_llm_embeddings \
    --rec-cfg  configs/config.yaml \
    --rec-ckpt /content/SigLLM/ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-base/checkpoint_best.pth \
    --data-pkl data/processed/ml-1m/{train,valid,test}_ood2.pkl \
    --item-num 3256 --space last_hidden \
    --output /content/SigLLM/data/processed/ml-1m/item_llm_emb.pt
# rồi lặp lại từ bước 4 (MF align) → 5 → 6 → 7 → 8.
```

Chạy vòng base-LLM trước để có số liệu; chỉ vào vòng parity nếu kết quả
hứa hẹn.

## Backup checkpoint (như notebook)

```bash
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/mf s3://sigllm/mf/<tag>/
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/qformer_stage1 s3://sigllm/qformer_stage1/<tag>/
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/qformer_stage2_qwen2 s3://sigllm/qformer_stage2/<tag>/
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/qformer_stage3_step1_lora_qwen2 s3://sigllm/qformer_stage31/<tag>/
aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/SigLLM/ckpt/qformer_stage3_step2_cie_qwen2 s3://sigllm/qformer_stage32/<tag>/
```

Nhớ dùng `<tag>` MỚI (vd `sella-align-v1`) — MF/Stage1/Stage2 mới không tương
thích ngữ nghĩa với `best-0.75-*` cũ, đừng sync đè.
