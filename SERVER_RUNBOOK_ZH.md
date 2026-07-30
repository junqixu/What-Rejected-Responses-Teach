# A800 单卡服务器运行手册

本手册面向单卡 NVIDIA A800 80GB。目标是用尽量少的命令完成环境检查、依赖安装、真实 GPU 冒烟、完整八类型 DPO 训练、验证集诊断和最终测试。服务器不需要 DeepSeek API，也不需要 `.env`；发布数据已经包含所有 rejected responses。

## 最终会得到什么

主实验默认训练 10 个条件、3 个随机种子，共 30 个模型：

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
cd /root
git clone --branch codex/rebuttal-release --single-branch https://github.com/junqixu/What-Rejected-Responses-Teach.git
cd What-Rejected-Responses-Teach
```

如果目录已经存在：

```bash
cd /root/What-Rejected-Responses-Teach
git checkout codex/rebuttal-release
git pull origin codex/rebuttal-release
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

## 第 4 步：准备 SFT checkpoint

Git 仓库不包含模型。你必须准备与规模匹配的 OP=10 SFT checkpoint，例如：

```text
/root/autodl-tmp/models/qwen2_0.5b_sft_op10/checkpoint-10339
```

检查目录：

```bash
ls -lh /root/autodl-tmp/models/qwen2_0.5b_sft_op10/checkpoint-10339
```

目录中至少应有 `config.json`、模型权重和 tokenizer 文件。0.5B checkpoint 只能配合 Qwen2-0.5B；不能给 1.5B 或 7B 使用。

目的：所有 DPO 条件必须从相同的 SFT 起点开始，才能公平比较。

## 第 5 步：一键安装和预检

把下面路径换成你的真实 checkpoint：

```bash
bash scripts/server_pipeline.sh bootstrap /root/autodl-tmp/models/qwen2_0.5b_sft_op10/checkpoint-10339
```

这一个命令会：

1. 将 checkpoint 路径保存到本地 `.server.env`；该文件被 Git 忽略。
2. 检查 Python 必须是 3.10、3.11 或 3.12。
3. 创建 `.venv`。
4. 安装 PyTorch 2.5.1 CUDA 12.1 和固定版本依赖。
5. 校验正式数据、CUDA、BF16 和 checkpoint。
6. 运行全部离线单元测试。

成功标志是预检输出包含：

```json
"ready": true
```

并且单元测试最后显示 `OK`。

如果服务器默认 `python3` 不在 3.10--3.12 范围内，例如系统有 `python3.10`：

```bash
PYTHON_BIN=python3.10 bash scripts/server_pipeline.sh bootstrap /你的/checkpoint/路径
```

如果驱动明确支持 CUDA 12.4，并希望使用 cu124：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/server_pipeline.sh bootstrap /你的/checkpoint/路径
```

## 第 6 步：运行真实 GPU 冒烟

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

```bash
bash scripts/server_pipeline.sh launch-full
```

它会立即返回 PID 和日志路径。后台任务依次：

1. 训练或续跑全部 30 个模型。
2. 对每个完成模型运行 validation 八类型诊断。
3. 自动生成跨 seed 汇总表。

断开 SSH 不会终止 `nohup` 后台任务。A800 80GB 预计需要约 8–18 小时，实际时间取决于平均序列长度、磁盘速度和 GPU 当前负载。默认只保存每个模型的最终权重，不保存 step-200/400 中间副本；矩阵级断点续跑仍然有效。

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
cd /root
git clone --branch codex/rebuttal-release --single-branch https://github.com/junqixu/What-Rejected-Responses-Teach.git
cd What-Rejected-Responses-Teach
bash scripts/server_pipeline.sh inspect
bash scripts/server_pipeline.sh bootstrap /你的/checkpoint/路径
bash scripts/server_pipeline.sh smoke
bash scripts/server_pipeline.sh launch-full
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

路径写错或模型尚未上传。检查 `ls -lh /真实/路径`，然后重新执行 bootstrap。

### CUDA 不可用

先执行 `nvidia-smi`。如果该命令失败，是服务器驱动或 GPU 实例问题；如果它成功但 PyTorch CUDA 失败，把 `outputs/server_preflight/environment.json` 和预检输出发给 Codex。

### 训练中途失败

保留现有目录，直接重新执行 `launch-full`。完整条件会跳过，失败条件会从干净输出目录重新开始。

### 磁盘不足

0.5B 的 30 模型矩阵采用 final-only 保存时建议至少预留 60GB，100GB 可用但应持续监控。1.5B 或更大模型需要扩容或改为 LoRA。运行 `df -h` 和 `bash scripts/server_pipeline.sh status` 检查。

### Codex 在远端无法登录

这不影响训练。Codex 用本地电脑完成代码修改和 Git 推送；远端 VS Code 只负责 `git pull`、终端运行与日志查看。不要将本地登录 token 或 API key 复制到远程服务器。
