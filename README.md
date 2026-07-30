# Structural DPO Rebuttal Experiments

本仓库包含八类结构化负样本、可识别性实验、长度控制、DPO 训练和诊断评测代码。正式服务器数据位于
`data/rebuttal_release/`，包含互不泄漏的 `train.jsonl`、`validation.jsonl` 和 `test.jsonl`，每个保留题目都同时具有以下八类负样本：

`computation`、`operation_substitution`、`dependency_redirect`、`extra_edge`、`missing_node`、`spurious_node`、`disorder`、`wrong_target`。

## 服务器快速开始

要求：Linux、Python 3.10/3.11、NVIDIA CUDA GPU，以及论文使用的 OP=10 Qwen2-0.5B SFT checkpoint。

```bash
git clone --branch codex/rebuttal-release --single-branch \
  https://github.com/junqixu/What-Rejected-Responses-Teach.git
cd What-Rejected-Responses-Teach
bash scripts/setup_server.sh
source .venv/bin/activate

export SFT_CHECKPOINT=/path/to/qwen2_0.5b_sft_op10/checkpoint-10339
python scripts/preflight_server.py --checkpoint "$SFT_CHECKPOINT"
```

如果希望用最少命令完成环境检查、安装、GPU 冒烟、断点续跑训练和验证，使用
[`SERVER_RUNBOOK_ZH.md`](SERVER_RUNBOOK_ZH.md) 与统一入口：

```bash
bash scripts/server_pipeline.sh bootstrap /path/to/sft_checkpoint
bash scripts/server_pipeline.sh smoke
bash scripts/server_pipeline.sh launch-full
bash scripts/server_pipeline.sh status
```

`setup_server.sh` 默认安装兼容范围更广的 PyTorch 2.5.1 CUDA 12.1 wheel；其他 CUDA 环境可覆盖：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_server.sh
```

## 训练

先进行不会启动 GPU 训练的全矩阵 dry-run：

```bash
python scripts/run_train_matrix.py \
  --checkpoint "$SFT_CHECKPOINT" \
  --suite p0 \
  --max_samples 8
```

P0 正式训练包含八类混合标准 DPO、token-mean 长度控制、Operation 单错误和 Wrong-Target 单错误，默认使用三个 seed：414、6201、2026。

```bash
python scripts/run_train_matrix.py \
  --checkpoint "$SFT_CHECKPOINT" \
  --suite p0 \
  --execute
```

运行全部八种单错误训练时使用 `--suite full --execute`。默认参数保持论文设置：1 epoch、effective batch size 8、learning rate `5e-6`、DPO beta `0.1`。

Confounded/Orthogonal 可识别性矩阵使用独立入口：

```bash
python scripts/run_identifiability_matrix.py \
  --checkpoint "$SFT_CHECKPOINT" \
  --max_samples 8

python scripts/run_identifiability_matrix.py \
  --checkpoint "$SFT_CHECKPOINT" \
  --execute
```

## 诊断评测

先 dry-run 检查所有 checkpoint 和 held-out 数据接口：

```bash
python scripts/run_eval_matrix.py --suite p0 --split validation --max_samples 8
```

执行验证集诊断并自动生成 transfer matrix：

```bash
python scripts/run_eval_matrix.py --suite p0 --split validation --execute
```

只有在配置冻结后才应运行一次正式 test：

```bash
python scripts/run_eval_matrix.py --suite p0 --split test --execute
```

Answer@128 使用固定的、未参与训练的 clean test source：

```bash
pip install -r requirements-eval.txt
python -m rebuttal.evaluation.eval_answer_at_k \
  --checkpoint outputs/rebuttal_taxonomy/checkpoints/mix_sum/seed_414 \
  --out outputs/rebuttal_taxonomy/answer_at_k/mix_sum/seed_414
```

Reasoning@128 仍必须接入论文实际使用的 reasoning evaluator；仓库不会用 Answer@128 冒充该指标。

## 数据验证

```bash
python scripts/preflight_server.py --data_only
python -m unittest discover -s rebuttal/tests -v
```

发布数据的数量、SHA256、DAG family 数量和跨 split 重叠检查记录在
`data/rebuttal_release/dataset_manifest.json`。训练入口还会再次校验重复 ID、chosen/rejected hash、同题 chosen 一致性，并在每个输出目录保存 `run_manifest.json`。

## 目录

```text
configs/rebuttal/              实验配置
data/rebuttal/                 confounded/orthogonal 可识别性数据
data/rebuttal_release/         服务器直接使用的八类正式数据
data/rebuttal_v2/source/       去重且按 DAG family 隔离的干净源池
rebuttal/                      数据、训练、评测、统计代码
scripts/                       环境预检与一键运行入口
outputs/rebuttal_*/            可复现的离线审计结果
```

DeepSeek 仅用于本地数据重建。正式发布数据已经包含 rejected responses，服务器训练不需要 API key。`.env`、原始 API 响应、大型旧数据、模型 checkpoint 和本地日志均被 `.gitignore` 排除。
