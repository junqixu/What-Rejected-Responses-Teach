# A800 单卡服务器运行手册

本手册面向单卡 NVIDIA A800 80GB。目标是用尽量少的命令完成环境检查、公开基础模型下载、干净 OP=10 SFT、真实 GPU 冒烟、完整八类型 DPO、验证集诊断和最终测试。服务器不需要 DeepSeek API，也不需要 `.env`；发布数据已经包含所有 rejected responses。

## 最终会得到什么

流水线首先从 `Qwen/Qwen2-0.5B` 训练一个公共 SFT 起点，输出到 `outputs/rebuttal_sft/qwen2_0.5b_op10/`。SFT 只读取 `train_clean.jsonl` 的 1000 条 `split=train、op=10` 正确解答，并在启动前确认与 validation/test 的 prompt 和 DAG family 均无重叠。随后主实验默认训练 10 个条件、3 个随机种子，共 30 个模型：

- 八类型混合标准 DPO：`mix_sum`。
- 八类型混合 token-mean 长度控制：`mix_mean`。
- 八种结构错误各自单独训练。
- 随机种子：414、6201、2026。

训练后自动在 validation 上计算八类型 transfer matrix、长度归一化诊断正确率、raw margin 和 length-normalized margin。test 不会被自动运行，必须在配置冻结后单独执行，防止反复查看测试集。

## 第 1 步：在远程 VS Code 中打开终端

确认 VS Code 左下角显示：

```text
SSH: connect.bjb1.seetacloud.com
```

选择 `Terminal -> New Terminal`。后续命令全部输入这个远程 Linux 终端，不要输入本机 PowerShell。

目的：确保下载、依赖和训练都发生在 A800 服务器，而不是本地电脑。

## 第 2 步：拉取代码

首次下载：

```bash
cd /root/autodl-tmp
git clone --branch codex/rebuttal-release --single-branch https://github.com/junqixu/What-Rejected-Responses-Teach.git
cd What-Rejected-Responses-Teach
```

如果目录已经存在：

```bash
cd /root/autodl-tmp/What-Rejected-Responses-Teach
git checkout codex/rebuttal-release
git -c http.version=HTTP/1.1 pull origin codex/rebuttal-release
```

目的：服务器只下载已审计的发布分支。DeepSeek key、原始 API 日志和本地 checkpoint 都没有上传。

## 第 3 步：生成环境报告

```bash
bash scripts/server_pipeline.sh inspect
```

报告同时显示在终端并保存到：

```text
outputs/server_preflight/environment.json
```

它检查 Python、GPU 名称和显存、驱动、CUDA 编译器、PyTorch、CPU、内存、磁盘、Git commit 和 checkpoint 状态，不读取或打印任何密钥。

第一次执行时 `torch.installed: false` 和 `SFT_CHECKPOINT is not set` 属于正常现象。请把这个 JSON 内容发给 Codex，以便根据实际驱动和磁盘进一步调整。

## 第 4 步：自动生成 SFT checkpoint

你当前没有基础模型文件或 checkpoint，直接运行：

```bash
bash scripts/server_pipeline.sh bootstrap-sft
```

这一个命令会：

1. 将模型和输出配置写入被 Git 忽略的 `.server.env`。
2. 使用 Python 3.12 创建 `.venv`，安装固定版本依赖。
3. 从 Hugging Face 下载公开的 `Qwen/Qwen2-0.5B`；它是预训练基础模型，不是随机初始化重训。
4. 校验 SFT 输入只能是 1000 条 train/OP=10 正确解答，并检查 validation/test 无 prompt 或 DAG family 泄漏。
5. 训练 2 epochs，只对 `Answer:` 后的答案 token 计算 loss，最终只保存一份完整 checkpoint。
6. 校验正式 DPO 数据、CUDA、BF16 和新 checkpoint，再运行全部离线单元测试。

基础模型与 pip 缓存都放在 `/root/autodl-tmp` 下的仓库 `.cache/`，避免占满系统盘；SFT checkpoint 位于：

```text
outputs/rebuttal_sft/qwen2_0.5b_op10/
```

预计首次依赖安装和模型下载约 10–30 分钟，SFT 约 5–20 分钟；网络速度是最大变量。重复运行时，若 `run_manifest.json` 已标记 `complete`，会直接复用而不会重训。

目的：所有 DPO 条件从同一个、未接触 validation/test 和 rejected responses 的 SFT 起点开始，保证比较公平。

## 第 5 步：一条命令启动整套实验

如果希望输入最少，直接运行：

```bash
bash scripts/server_pipeline.sh start
```

它依次完成上一节的 SFT、8 样本真实 GPU smoke test，并用 `nohup` 在后台启动完整 DPO + validation。终端最后会返回 PID 和日志路径，之后可以安全断开 SSH。

如果已有自己的同规模 SFT checkpoint，仍可使用：

```bash
bash scripts/server_pipeline.sh bootstrap /你的/checkpoint/路径 Qwen/Qwen2-0.5B
```

成功标志是预检输出包含：

```json
"ready": true
```

并且单元测试最后显示 `OK`。

如果服务器默认 `python3` 不在 3.10--3.12 范围内，例如系统有 `python3.10`：

```bash
PYTHON_BIN=python3.10 bash scripts/server_pipeline.sh bootstrap-sft
```

如果驱动明确支持 CUDA 12.4，并希望使用 cu124：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/server_pipeline.sh bootstrap-sft
```

## 第 6 步：运行真实 GPU 冒烟

如果上一节已经用了 `start`，这一步已自动完成；下面命令用于单独运行或重新检查。

```bash
bash scripts/server_pipeline.sh smoke
```

它使用 8 条样本、一个 seed，实际训练 P0 的四个小模型。目的不是得到论文结果，而是尽早发现显存、tokenizer、模型格式、bitsandbytes 或 CUDA 问题。

成功后会看到：

```text
Smoke run complete.
```

产物位于：

```text
outputs/smoke_taxonomy/checkpoints/
```

重复运行不会重训已经完成的条件；`--resume` 会自动跳过 `status=complete` 的目录，并重建失败或不完整的目录。

## 第 7 步：后台启动完整训练与 validation

如果已经用了 `start`，后台任务也已启动；不要紧接着重复执行。下面命令用于手动分步启动或中断后的矩阵级续跑。

```bash
bash scripts/server_pipeline.sh launch-full
```

它会立即返回 PID 和日志路径。后台任务依次：

1. 训练或续跑全部 30 个模型。
2. 对每个完成模型运行 validation 八类型诊断。
3. 自动生成跨 seed 汇总表。

断开 SSH 不会终止 `nohup` 后台任务。A800 80GB 上，完整 30 模型 DPO + validation 预计约 8–18 小时；加上首次环境、下载和 SFT，整条流水线通常约 9–20 小时。实际时间取决于网络、平均序列长度、磁盘速度和 GPU 当前负载。默认只保存每个模型的最终权重，不保存 step-200/400 中间副本；矩阵级断点续跑仍然有效。

## 第 8 步：查看进度

```bash
bash scripts/server_pipeline.sh status
```

该命令显示 GPU 占用、训练状态计数、已完成诊断数量和最近 40 行日志。可以反复执行，不会影响训练。

也可以直接查看完整日志：

```bash
cat "$(cat outputs/server_logs/latest_full_log.txt)"
```

训练中断后，不要删除输出目录，重新执行即可续跑：

```bash
bash scripts/server_pipeline.sh launch-full
```

## 第 9 步：检查 validation 结果

主要汇总文件：

```text
outputs/rebuttal_taxonomy_full/diagnostics/validation/aggregate/expanded_diagnostic_matrix.csv
outputs/rebuttal_taxonomy_full/diagnostics/validation/aggregate/expanded_diagnostic_summary.csv
outputs/rebuttal_taxonomy_full/diagnostics/validation/aggregate/expanded_diagnostic_manifest.json
```

重点关注：

- `mean_diagnostic_accuracy`：越高表示越稳定地偏好正确推理。
- `mean_length_normalized_margin`：大于 0 表示排除长度影响后仍偏好 chosen。
- `mix_sum` 对比 `mix_mean`：检验长度控制是否改变结论。
- 对角线与非对角线：分别表示直接学习和跨错误类型泛化。

## 第 10 步：只运行一次正式 test

确认 validation 配置不再修改后：

```bash
bash scripts/server_pipeline.sh test
```

结果位于：

```text
outputs/rebuttal_taxonomy_full/diagnostics/test/
```

目的：test 只用于最终无偏报告，不能拿来反复调参数。

## 可选：可辨识性实验

```bash
bash scripts/server_pipeline.sh ident
```

它运行 Missing/Disorder 与 Computation/Dependency 的 confounded、orthogonal 和 factorial 条件，同样支持断点续跑。结果位于 `outputs/rebuttal_identifiability/`。

## 最少命令清单

```bash
cd /root/autodl-tmp
git clone --branch codex/rebuttal-release --single-branch https://github.com/junqixu/What-Rejected-Responses-Teach.git
cd What-Rejected-Responses-Teach
bash scripts/server_pipeline.sh inspect
bash scripts/server_pipeline.sh start
bash scripts/server_pipeline.sh status
```

完整训练和 validation 完成后，再执行：

```bash
bash scripts/server_pipeline.sh test
```

## 常见问题

### `Python 3.10, 3.11, or 3.12 is required`

使用服务器已有的 Python 3.10--3.12，或创建对应 Conda 环境后通过 `PYTHON_BIN` 指定。

### `SFT checkpoint does not exist`

自动 SFT 尚未完成，或者配置中的路径错误。先看 `outputs/rebuttal_sft/qwen2_0.5b_op10/run_manifest.json` 和终端报错，再重新执行 `bootstrap-sft`；完整状态会复用，失败状态不会被误当成 checkpoint。

### Hugging Face 下载失败

先直接重试 `bootstrap-sft`；缓存支持续传。若服务器所在网络无法访问 Hugging Face，请配置你有权使用的模型源或在可访问机器下载 `Qwen/Qwen2-0.5B` 后把本地模型目录作为 `start` 的第一个参数。不要把访问 token 写进 Git。

### CUDA 不可用

先执行 `nvidia-smi`。如果该命令失败，是服务器驱动或 GPU 实例问题；如果它成功但 PyTorch CUDA 失败，把 `outputs/server_preflight/environment.json` 和预检输出发给 Codex。

### 训练中途失败

保留现有目录，直接重新执行 `launch-full`。完整条件会跳过，失败条件会从干净输出目录重新开始。

### 磁盘不足

0.5B 的 30 模型矩阵采用 final-only 保存时建议至少预留 60GB，100GB 可用但应持续监控。1.5B 或更大模型需要扩容或改为 LoRA。运行 `df -h` 和 `bash scripts/server_pipeline.sh status` 检查。

### Codex 在远端无法登录

这不影响训练。Codex 用本地电脑完成代码修改和 Git 推送；远端 VS Code 只负责 `git pull`、终端运行与日志查看。不要将本地登录 token 或 API key 复制到远程服务器。
