# AIC Search: ban chay local cho dong doi

## Thu muc can tai ve tu Drive

Dat ba folder sau cung cap, ten folder goc tuy y:

```text
AIC_TEAM/
  CODE_FOR_TEAM/
    install_requirements_wsl.sh
    run_local_server_wsl.sh
    README.md
    ...cac file code khac
  DATA_FOR_TEAM/
    search_index_ocr/
      meta.db
      faiss_index/
      clip_text/
      asr_cache.json
      serve_manifest.json
      serve_stats.json
    models/multilingual-e5-small/
    redis_runtime/
    redis_data/appendonlydir/
    inventory.json
  filtered_data_all_873_moved/
    ...giu nguyen cac folder chu de va keyframes ben trong
    map-keyframes/
    transcripts_asr/
  videos/                    (tuy chon: video goc)
```

Tai day du ve o dia local va giai nen truoc khi chay. Khong chay truc tiep
tren o Drive dang stream. Khong doi ten cac folder con/keyframe.
DATA_FOR_TEAM la ban sao da xu ly; khong can chay OCR, ASR, CLIP indexing
hay migration Redis. Khong gui .venv hoac Python environment cua may nguon.

## Cai lan dau

Dung Windows + WSL Ubuntu 24.04 x86_64. Neu chua co WSL, mo PowerShell
voi quyen Administrator va chay `wsl --install -d Ubuntu-24.04`, khoi dong
lai neu Windows yeu cau, sau do tao user Ubuntu.

Mo terminal Ubuntu. Vi du ban tai bo du lieu vao D:\AIC_TEAM:

```bash
cd /mnt/d/AIC_TEAM/CODE_FOR_TEAM
bash install_requirements_wsl.sh
```

Can Internet de tai dependencies va mat khau Ubuntu de cai goi he thong.
Moi truong Python duoc tao tai ~/.venvs/aic-team. Goi PyTorch ho tro CUDA
11.8; may khong co GPU NVIDIA van chay bang CPU, toc do tim kiem cham hon.
Moi may dung ban sao data va Redis rieng, khong can ket noi may cua dong doi.

## Chay moi lan

```bash
cd /mnt/d/AIC_TEAM/CODE_FOR_TEAM
bash run_local_server_wsl.sh
```

Launcher uu tien `TEAM_VENV` neu duoc chi dinh; mac dinh dung
`~/.venvs/aic-team`. Neu moi truong mac dinh chua co, launcher thu `.venv`
trong `CODE_FOR_TEAM`, roi `../CODE/.venv`. Co the dung lai moi truong da
cai requirements bang `TEAM_VENV=../CODE/.venv bash run_local_server_wsl.sh`.
`DATA_ROOT` neu duoc truyen vao se duoc giu nguyen.

Cho log `Application startup complete`, mo http://localhost:8080.
Kiem tra http://localhost:8080/ready. Nhan Ctrl+C de tat ca API va Redis
do launcher nay khoi dong. Redis can nap snapshot vao RAM moi lan bat;
CLIP text encoder duoc nap va chay thu tim kiem truoc khi server nhan ket noi.
Duong dan server chi duoc in sau khi readiness thanh cong.
Model CLIP va summary da kem trong bo data.

Frame search va include/exclude dung SQLite/FAISS. OCR, ASR, summary dung
Redis tai 127.0.0.1. API cung chi nghe tren may local.

Neu port ban, tat server cu hoac chon port khac:

```bash
SERVER_PORT=8081 REDIS_PORT=16380 bash run_local_server_wsl.sh
```

Neu de folder anh hoac data o o dia khac:

```bash
TEAM_DATA=/home/your-user/AIC_DATA \
KEYFRAMES_DIR=/mnt/e/filtered_data_all_873_moved \
bash run_local_server_wsl.sh
```

Nen de DATA_FOR_TEAM trong filesystem Linux cua WSL neu o /mnt/c hoac
/mnt/d nap cham. Anh co the van de tren Windows. Neu muon xem video goc:

```bash
VIDEOS_DIR=/mnt/e/videos bash run_local_server_wsl.sh
```

## Kiem tra file sau khi tai

```bash
python3 verify_package.py
```

Lenh nay doi chieu SHA-256 cac file xu ly trong DATA_FOR_TEAM. Chay truoc
lan bat server dau tien: Redis se cap nhat AOF khi chay nen checksum snapshot
co the thay doi sau do. Lenh nay cung kiem tra duong dan cua toan bo 177.321
keyframe trong index deu co trong folder anh; giu nguyen cau truc khi upload.

## File chi danh cho nguoi dong goi

package_processed_data.py chi chay tren may nguon de tao ban sao ban dau.
Dong doi KHONG chay file nay, cac script build/index hay migrate_to_redis.py.
Chi can install_requirements_wsl.sh va run_local_server_wsl.sh.

## Interface maintenance

See [ARCHITECTURE.md](ARCHITECTURE.md) for the UI layout, shared components, and validation commands. Include the entire static/ folder when copying CODE_FOR_TEAM.

