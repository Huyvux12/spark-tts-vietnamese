# 🎤 Spark-TTS Vietnamese - Text-to-Speech Tiếng Việt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/huyvux3005/spark-tts-vietnamese-5hourr_16bit)

Dự án fine-tune mô hình **Spark-TTS** cho tiếng Việt với 2 giọng nói khác nhau, sử dụng thư viện **Unsloth** để tối ưu tốc độ training x2. Dự án được thiết kế để chạy trên **Google Colab/Kaggle** với GPU T4 miễn phí.

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Cấu Trúc Project](#-cấu-trúc-project)
- [Demo Âm Thanh](#-demo-âm-thanh)
- [Hướng Dẫn Finetune](#-hướng-dẫn-finetune)
  - [Trên Colab/Kaggle](#trên-colabkaggle)
  - [Trên Local](#trên-local)
- [Hướng Dẫn Inference](#-hướng-dẫn-inference)
- [Tài Liệu Tham Khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Giới Thiệu

### Công nghệ sử dụng

| Công nghệ | Mô tả |
|-----------|-------|
| **Spark-TTS** | Mô hình TTS dựa trên LLM, sử dụng BiCodec để mã hóa audio thành token |
| **Unsloth** | Thư viện tối ưu training, tăng tốc x2 và giảm VRAM |
| **LoRA** | Kỹ thuật fine-tune hiệu quả, chỉ train 12% tham số |
| **BiCodec** | Tokenizer chuyển đổi audio ↔ semantic/global tokens |

### 2 Giọng nói được train

| Giọng | Nguồn | Đặc điểm |
|-------|-------|----------|
| `@W2WMovie` | Kênh YouTube W2WMovie | Giọng review phim, rõ ràng, chuyên nghiệp |
| `@ThanhPahm` | Kênh YouTube ThanhPahm | Giọng tự nhiên, phong cách riêng |

---

## 💻 Yêu Cầu Hệ Thống

### Chạy trên Colab/Kaggle (Khuyến nghị)

| Thành phần | Yêu cầu |
|------------|---------|
| GPU | NVIDIA T4 (miễn phí) |
| VRAM | ~15GB |
| Runtime | Python 3.10+ |

### Chạy trên Local

| Thành phần | Yêu cầu |
|------------|---------|
| GPU | NVIDIA với VRAM ≥ 15GB (RTX 3090, 4090, A100...) |
| CUDA | 12.x |
| Python | 3.10+ |
| RAM | ≥ 16GB |

---

## 📁 Cấu Trúc Project

```
projecet3/
├── finetune_training.ipynb      # Notebook fine-tune model
├── spark_tts_inference.ipynb    # Notebook inference (chạy model)
├── thanhphamtesst.wav           # Demo giọng @ThanhPahm
├── w2wmovie.wav                 # Demo giọng @W2WMovie
├── paper_sparktts.pdf           # Paper gốc Spark-TTS
├── 2006.13979v2.pdf             # Paper tham khảo
└── README.md                    # File này
```

---

## 🔊 Demo Âm Thanh

Nghe thử 2 giọng nói đã được train:

### Giọng @ThanhPahm
<audio controls>
  <source src="thanhphamtesst.wav" type="audio/wav">
  Trình duyệt không hỗ trợ audio. <a href="thanhphamtesst.wav">Tải file tại đây</a>
</audio>

📥 [Tải file thanhphamtesst.wav](thanhphamtesst.wav)

### Giọng @W2WMovie  
<audio controls>
  <source src="w2wmovie.wav" type="audio/wav">
  Trình duyệt không hỗ trợ audio. <a href="w2wmovie.wav">Tải file tại đây</a>
</audio>

📥 [Tải file w2wmovie.wav](w2wmovie.wav)

> **Lưu ý:** GitHub không hỗ trợ phát audio trực tiếp trong README. Clone repo về local hoặc sử dụng GitHub Pages để nghe.

---

## 🚀 Hướng Dẫn Finetune

### Trên Colab/Kaggle

#### Bước 1: Cài đặt dependencies

```python
%%capture
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install transformers==4.55.4
!pip install --no-deps trl==0.22.2
!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx huggingface_hub
```

#### Bước 2: Load model với Unsloth

```python
from unsloth import FastModel
from huggingface_hub import snapshot_download
import torch

# Download Spark-TTS base model
snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="Spark-TTS-0.5B")

# Load model với Unsloth (tăng tốc x2)
model, tokenizer = FastModel.from_pretrained(
    model_name="Spark-TTS-0.5B/LLM",
    max_seq_length=2048,
    dtype=torch.float32,  # Spark-TTS chỉ hoạt động với float32
    full_finetuning=False,
    load_in_4bit=False,
)
```

#### Bước 3: Cấu hình LoRA (12% tham số)

```python
model = FastModel.get_peft_model(
    model,
    r=128,  # Kích thước ma trận LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,  # Đặt 0 để tối ưu VRAM
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

#### Bước 4: Chuẩn bị Dataset

**Cách 1: Từ HuggingFace Datasets**
```python
from datasets import load_dataset
dataset = load_dataset("your_username/your_dataset", split="train")
```

**Cách 2: Từ Kaggle (dạng Arrow)**
```python
from datasets import load_from_disk, concatenate_datasets

dataset1 = load_from_disk("/kaggle/input/thanhpahm-tts-standardized")
dataset2 = load_from_disk("/kaggle/input/w2wmovie-voice-2-standardized")
dataset = concatenate_datasets([dataset1, dataset2]).shuffle(seed=42)
```

#### Bước 5: Training với SFTTrainer

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        optim="adamw_8bit",
        output_dir="outputs",
        report_to="wandb",  # Optional: log lên Weights & Biases
    ),
)

trainer.train()
```

#### Bước 6: Push model lên HuggingFace

```python
model.push_to_hub("your_username/spark-tts-vietnamese", token="hf_xxxxx")
tokenizer.push_to_hub("your_username/spark-tts-vietnamese", token="hf_xxxxx")
```

---

### Trên Local

#### Bước 1: Clone repository

```bash
git clone https://github.com/your_username/spark-tts-vietnamese.git
cd spark-tts-vietnamese
```

#### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### Bước 3: Cài đặt dependencies

```bash
# Cài PyTorch với CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Cài các thư viện còn lại
pip install unsloth transformers==4.55.4 trl==0.22.2 omegaconf einx huggingface_hub wandb jupyter

# Clone Spark-TTS
git clone https://github.com/SparkAudio/Spark-TTS
```

#### Bước 4: Chạy Jupyter Notebook

```bash
jupyter notebook finetune_training.ipynb
```

#### ⚠️ Lưu ý khi train local

- **VRAM tối thiểu**: 15GB (RTX 3090, 4090, A100...)
- **Kiểu dữ liệu**: Phải dùng `torch.float32`, không hỗ trợ fp16/bf16
- **Gradient checkpointing**: Bật `"unsloth"` để tiết kiệm VRAM
- **Batch size**: Giảm xuống 1 nếu bị OOM (Out of Memory)

---

## 🎙️ Hướng Dẫn Inference

### Cài đặt nhanh

```python
!pip install einx transformers soundfile huggingface_hub
!git clone https://github.com/SparkAudio/Spark-TTS
```

### Load model từ HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys

sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer

# Load LLM đã fine-tune
model = AutoModelForCausalLM.from_pretrained("huyvux3005/spark-tts-vietnamese-5hourr_16bit")
tokenizer = AutoTokenizer.from_pretrained("huyvux3005/spark-tts-vietnamese-5hourr_16bit")

# Load BiCodec tokenizer
snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="Spark-TTS-0.5B")
audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")
```

### Chạy inference với giọng nói tùy chọn

```python
import torch
import re
import soundfile as sf

device = torch.device("cuda")
model.to(device)

# Chọn giọng: @W2WMovie hoặc @ThanhPahm
chosen_voice = "@W2WMovie"
text = "Bộ phim này thực sự là một kiệt tác!"

# Tạo prompt
input_text = f"{chosen_voice}: {text}"
prompt = f"<|task_tts|><|start_content|>{input_text}<|end_content|><|start_global_token|>"

# Generate tokens
inputs = tokenizer([prompt], return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=2048)

# Parse tokens
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
semantic_tokens = re.findall(r"<\|bicodec_semantic_(\d+)\|>", output_text)
global_tokens = re.findall(r"<\|bicodec_global_(\d+)\|>", output_text)

# Detokenize thành audio
audio_tokenizer.model.to(device)
wav = audio_tokenizer.detokenize(
    torch.tensor([int(t) for t in global_tokens]).unsqueeze(0).to(device),
    torch.tensor([int(t) for t in semantic_tokens]).unsqueeze(0).to(device)
)

# Lưu file
sf.write("output.wav", wav, 16000)
```

### Nghe kết quả trong Notebook

```python
from IPython.display import Audio, display
display(Audio("output.wav"))
```

---

## 📚 Tài Liệu Tham Khảo

- 📄 [Spark-TTS Paper (arXiv:2503.01710)](https://arxiv.org/abs/2503.01710) — Paper gốc về kiến trúc Spark-TTS
- 📘 [Unsloth TTS Fine-tuning Guide](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning) — Hướng dẫn fine-tune TTS với Unsloth
- 🔧 [Spark-TTS GitHub](https://github.com/SparkAudio/Spark-TTS) — Repository chính thức
- 🤗 [Model trên HuggingFace](https://huggingface.co/huyvux3005/spark-tts-vietnamese-5hourr_16bit) — Model đã fine-tune

---


