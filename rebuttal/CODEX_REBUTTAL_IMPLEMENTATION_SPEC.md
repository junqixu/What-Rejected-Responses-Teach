# Codex 实现文档：DPO 结构化负样本的可识别性、混淆与误差分类扩展

**目标仓库**：`https://github.com/junqixu/What-Rejected-Responses-Teach`  
**适用论文**：*What Rejected Responses Teach: Understanding DPO Generalization for Mathematical Reasoning*  
**文档目的**：指导 Codex 在现有仓库中实现针对审稿意见的新增理论诊断、数据构造、训练、评估和结果汇总代码。  
**优先级原则**：先完成能够直接回应审稿人“结果只是 DPO 对比学习的必然现象”和“哪些结构不可识别”的实验，再扩展误差分类和自然错误分析。

---

## 0. 给 Codex 的总指令

在仓库根目录读取本文件后，按以下方式工作：

1. **先审计，后修改。** 不要假定仓库结构。先遍历代码、配置、数据格式、训练入口和评估入口，生成：
   - `rebuttal/REPO_AUDIT.md`
   - `rebuttal/IMPLEMENTATION_PLAN.md`
2. **复用现有实现。** 优先复用当前 DAG 表示、四类 perturbation、DPO 训练器、Reasoning@k、diagnostic preference 和现有 checkpoint 加载逻辑。
3. **不得覆盖已有实验。** 所有新增结果写入独立目录，例如：
   - `outputs/rebuttal_identifiability/`
   - `outputs/rebuttal_gradient/`
   - `outputs/rebuttal_taxonomy/`
4. **保持主实验训练条件不变。** 除明确消融外，继续使用论文中的公共设置：
   - Base：Qwen2-0.5B
   - 初始化：同一个 OP=10 SFT checkpoint
   - DPO epoch：1
   - effective batch size：8
   - learning rate：`5e-6`
   - DPO beta：`0.1`
   - prompt、chosen response、训练深度和优化配置固定
5. **所有新增数据必须带 manifest。** 至少保存：数据来源、prompt ID、chosen hash、rejected hash、error labels、DAG metadata、seed、generator version 和 split。
6. **先做 smoke test。** 所有训练入口必须支持 `--max_samples`、`--dry_run` 和小数据 smoke test。不要在脚本未通过测试时直接启动全量 GPU 训练。
7. **禁止结果导向修改。** 不得根据正式测试结果调整阈值、数据构造或评价规则。任何探索性修改必须写入 `rebuttal/EXPLORATORY_LOG.md`。
8. **失败也要输出。** 若新增实验不支持预期假设，仍应生成完整结果和说明，不得静默丢弃。

---

# 1. 审稿意见与新增实验的对应关系

审稿人的核心批评可以拆成四点：

| 编号 | 审稿意见 | 需要提供的证据 |
|---|---|---|
| C1 | DPO 本来就是比较 chosen 与 rejected；负样本在哪个维度变化，模型学习哪个维度，这并不意外 | 不再只证明“存在信号”，而是证明**信号何时可识别、何时不可识别，以及不同结构如何产生冲突或泛化差异** |
| C2 | 论文没有展示超出 Bradley–Terry / logit 模型直接预测的现象 | 构造相同 DPO 目标但不同设计矩阵秩的训练集，测试结构维度的可识别性 |
| C3 | 更有意义的问题是：是否存在即使数据足够，DPO 仍无法学习的 DAG 或误差结构 | 给出**偏好差分空间的零空间命题**，并通过 confounded vs orthogonal 数据实验验证 |
| C4 | 四种错误类型看起来是人工选择的，没有说明为什么是这四种，而不是其他错误 | 从 DAG 基本编辑操作推导误差分类，新增错误类型，并分析自然模型错误对分类的覆盖率 |

本轮新增工作不应继续强化以下较弱表述：

> “DPO 会对训练中出现的错误类型变得更敏感。”

应当升级为：

> “偏好数据只有在结构差分方向可识别时，才可能使 DPO 学到相应约束；当多个错误维度在训练对中完全混淆时，DPO 目标只能约束它们的组合方向，无法识别各个结构因素。负样本设计因此决定的不只是学习内容，也决定了哪些推理约束在统计上可学习。”

---

# 2. 新增理论诊断：偏好差分空间与结构可识别性

## 2.1 需要在代码中支持的表示

为每个 chosen/rejected response 定义结构误差特征：

\[
\phi(y) \in \mathbb{R}^{m}.
\]

第一版至少包括：

```text
computation
operation_substitution
dependency_redirect
extra_edge
missing_node
spurious_node
disorder
wrong_target
```

每一维可以同时保存：

- `binary`：是否存在该错误；
- `count`：该错误发生次数；
- `severity`：错误影响的节点比例或归一化严重度。

对每个偏好对定义：

\[
\Delta\phi_i = \phi(y_i^+) - \phi(y_i^-).
\]

由于 chosen 通常无错误，也可额外保存正向的 rejected error exposure：

\[
e_i = \phi(y_i^-)-\phi(y_i^+).
\]

## 2.2 需要实现的设计矩阵审计

新增模块建议：

```text
rebuttal/
  identifiability/
    feature_schema.py
    build_design_matrix.py
    matrix_diagnostics.py
```

输入：任意 preference JSONL。  
输出：

- `design_matrix.npy`
- `feature_names.json`
- `matrix_report.json`
- `singular_values.csv`
- `nullspace_basis.npy`
- `design_spectrum.pdf` 或 `.png`

至少计算：

```text
n_samples
n_features
matrix_rank
effective_rank
singular_values
condition_number
nullspace_dimension
pairwise_feature_correlation
variance_per_feature
```

有效秩建议同时报告：

\[
r_{\mathrm{eff}}=\exp\left(-\sum_j p_j\log p_j\right),
\qquad
p_j=\frac{\sigma_j}{\sum_k\sigma_k}.
\]

## 2.3 论文中可使用的命题

代码不负责撰写正式证明，但需要生成支持该命题的数值结果。

**命题：Preference-difference identifiability。** 设局部效用可写为：

\[
V_\theta(x,y)=\theta^\top\phi(x,y),
\]

偏好训练集的差分矩阵为：

\[
X=[\Delta\phi_1^\top;\ldots;\Delta\phi_N^\top].
\]

DPO/Bradley–Terry 似然只依赖于 \(X\theta\)。对于任意

\[
v\in\operatorname{Null}(X),
\]

参数 \(\theta\) 与 \(\theta+v\) 在所有观测偏好对上产生相同的似然。因此，位于差分矩阵零空间中的结构方向无法从该偏好数据中识别；若两个错误维度始终共现，只能识别它们的组合效用，而不能分别识别各自效用。

注意：正式论文中必须将其表述为**线性特征模型或神经网络局部线性化下的可识别性结论**，不得夸大为任意非线性模型的全局定理。

---

# 3. P0 主实验：Confounded vs Orthogonal Preference Design

这是本轮最重要的新增实验，直接回答审稿人提出的“哪些错误结构 DPO 无法学习”。

## 3.1 实验问题

当两个错误维度总是同时出现在 rejected response 中时，DPO 是否只能学到联合拒绝方向，而无法分别识别两个错误？将同样的误差维度正交化后，能否恢复单独的结构敏感性？

## 3.2 默认错误对

至少运行两组：

```text
Pair A: Missing-Node + Disorder
Pair B: Computation + Dependency
```

第一组代表链完整性与拓扑顺序，第二组代表局部数值与依赖关系。

## 3.3 数据条件

对每个错误对 \((A,B)\) 构造以下训练条件。

### R1. Confounded-AB

每个 rejected response 同时包含 A 和 B：

```text
chosen = clean trace
rejected = apply_B(apply_A(clean trace))
```

其结构设计矩阵在 A/B 子空间上为 rank 1。

### R2. Orthogonal-AB-SizeMatched

总 preference pair 数与 Confounded-AB 相同：

```text
50%: clean vs A-only
50%: clean vs B-only
```

其 A/B 子空间通常为 rank 2。

### R3. Orthogonal-AB-ExposureMatched

为匹配每个误差维度的总暴露次数，对每个 prompt 同时生成：

```text
clean vs A-only
clean vs B-only
```

该条件的 pair 数可能是 Confounded-AB 的两倍。结果中必须明确区分：

- pair-count matched；
- error-exposure matched。

### R4. Factorial-AB

建议比例：

```text
40% A-only
40% B-only
20% A+B
```

用于测试部分联合暴露是否保持可识别性。

### R5. Single-A / Single-B

复用现有单误差模型，作为上界和参照。

## 3.4 数据控制

各条件必须尽可能匹配：

- prompt 集合；
- chosen response；
- OP；
- DAG 节点数；
- DAG 深度；
- error node depth；
- rejected token 数；
- token edit distance；
- final-answer correctness；
- 每次 perturbation 的 severity；
- 训练步数或数据暴露量。

生成两个公平性版本：

1. **Step-matched**：固定 optimizer update 数；
2. **Exposure-matched**：固定每个结构错误的累计出现次数。

## 3.5 新增数据字段

每条 preference record 至少包含：

```json
{
  "id": "...",
  "prompt_id": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "regime": "confounded_ab",
  "error_labels": ["missing_node", "disorder"],
  "error_vector": {
    "computation": 0,
    "dependency_redirect": 0,
    "missing_node": 1,
    "disorder": 1
  },
  "op": 10,
  "dag_depth": 10,
  "error_node_depth": 0.7,
  "num_nodes_changed": 1,
  "num_edges_changed": 2,
  "chosen_tokens": 120,
  "rejected_tokens": 113,
  "token_edit_distance": 17,
  "final_answer_correct": true,
  "seed": 414,
  "source_hash": "...",
  "generator_version": "rebuttal-v1"
}
```

## 3.6 训练

所有 regime 从同一 OP=10 SFT checkpoint 开始。  
至少使用 3 个 seed。优先沿用仓库现有 seed 体系；若没有统一设置，使用：

```text
414
6201
2026
```

建议配置：

```text
configs/rebuttal/ident_missing_disorder_confounded.yaml
configs/rebuttal/ident_missing_disorder_orthogonal_size.yaml
configs/rebuttal/ident_missing_disorder_orthogonal_exposure.yaml
configs/rebuttal/ident_missing_disorder_factorial.yaml
configs/rebuttal/ident_compute_dependency_confounded.yaml
...
```

## 3.7 评价集

所有模型在完全相同的 held-out prompts 上评价：

```text
Eval-A: clean vs A-only
Eval-B: clean vs B-only
Eval-AB: clean vs A+B
Eval-CrossLocation: A/B 发生在未见节点深度
Eval-CrossTopology: A/B 发生在未见 DAG 拓扑
```

评价同时报告：

- length-normalized diagnostic preference accuracy；
- raw DPO margin；
- Answer@128；
- Reasoning@128；
- OP=2–10、11–14、15–20；
- seed 均值和标准差。

新增汇总指标：

\[
\text{MeanAxis}=\frac{Acc_A+Acc_B}{2},
\]

\[
\text{WorstAxis}=\min(Acc_A,Acc_B),
\]

\[
\text{AxisImbalance}=|Acc_A-Acc_B|.
\]

## 3.8 主结果表

生成：

```text
outputs/rebuttal_identifiability/table_identifiability.csv
```

列：

```text
error_pair
regime
n_pairs
exposure_A
exposure_B
design_rank
effective_rank
condition_number
train_pair_accuracy
diag_A
diag_B
diag_AB
mean_axis
worst_axis
axis_imbalance
answer_128
reasoning_128
op_11_14_reasoning
op_15_20_reasoning
seed
```

## 3.9 判定逻辑

不得预设结果，但应自动生成以下分析：

- Confounded-AB 是否在 `Eval-AB` 上较高，但在 A-only/B-only 上失衡；
- Orthogonal 条件是否提升 `WorstAxis` 和 `MeanAxis`；
- `design_rank` / `effective_rank` 是否与 isolated-axis accuracy 正相关；
- 结果是否跨两个错误对、三个 seed 保持方向一致。

统计：

- paired bootstrap 95% CI；
- 同 prompt 上二元 diagnostic correctness 使用 exact McNemar；
- 多比较使用 Holm correction；
- 报告 effect size，不只报告 p 值。

---

# 4. P0 控制实验：长度、表面形式和错误严重度

Missing-Node 的 rejected response 可能天然更短；标准 DPO 使用 response token log-probability 总和时，长度可能成为混杂因素。本轮必须显式审计并控制。

## 4.1 Surface statistics audit

新增：

```text
rebuttal/controls/surface_audit.py
```

对所有原始四类训练数据报告：

```text
prompt_tokens
chosen_tokens
rejected_tokens
length_delta
normalized_length_delta
token_edit_distance
step_count_delta
final_answer_correct
numeric_delta
error_node_depth
num_nodes_changed
num_edges_changed
```

输出按 error type 的均值、标准差、分位数和 effect size。

## 4.2 Metadata-only classifier

使用不含语义和 hidden state 的表面 metadata 预测 error type：

- multinomial logistic regression；
- group split by prompt ID；
- 报告 macro-F1 和 accuracy。

如果表面特征可高精度预测 error type，则必须在论文中承认并运行 matched control。

## 4.3 Length-normalized DPO

检查现有 DPO trainer 如何聚合 chosen/rejected token log-probability。新增可配置选项：

```text
--sequence_logp_reduction sum   # 原始标准 DPO
--sequence_logp_reduction mean  # token 平均 log-probability
```

不得默认改变原始结果。新增四类单误差训练对照：

```text
DPO-Computation-MeanLogP
DPO-Dependency-MeanLogP
DPO-Missing-MeanLogP
DPO-Disorder-MeanLogP
```

核心问题：Missing-Node 的优势是否在 length-normalized DPO 下仍然存在。

## 4.4 Matched subset

实现按以下变量匹配四类样本：

```text
OP
DAG depth
node count
rejected token length
edit distance
error location depth
final-answer correctness
```

优先使用 exact bucket + nearest-neighbor matching。输出：

```text
data/rebuttal/matched_four_types.jsonl
outputs/rebuttal_controls/matching_balance.csv
```

匹配后必须检查 standardized mean difference；建议绝对值 `< 0.1`，否则标记未充分平衡。

---

# 5. P1 实验：DPO 梯度几何、信息冲突与聚合损失

该实验用于回答审稿人提出的“不同 DAG 结构上的梯度是否混淆，以及 pairwise aggregation 是否破坏信息”。

## 5.1 目标

测量不同错误类型的 DPO 梯度方向：

- 同类错误是否具有更一致的梯度；
- 不同类型之间是否存在负 cosine；
- pairwise mixture 的 GDR 是否与 gradient conflict 有关；
- confounded 数据的梯度矩阵是否呈现较低有效秩。

## 5.2 实现位置

建议：

```text
rebuttal/
  gradient/
    collect_per_example_gradients.py
    gradient_projection.py
    gradient_geometry.py
    correlate_with_transfer.py
```

## 5.3 梯度对象

对单样本 DPO loss：

\[
\ell_i=-\log\sigma\left(
\beta[(\log\pi_\theta(y_i^+|x_i)-\log\pi_{ref}(y_i^+|x_i))
-(\log\pi_\theta(y_i^-|x_i)-\log\pi_{ref}(y_i^-|x_i))]
\right)
\]

计算：

\[
g_i=\nabla_\theta\ell_i.
\]

若当前训练为 LoRA，默认收集全部 trainable adapter 参数。  
若为 full fine-tuning，支持参数正则表达式，默认优先：

```text
last transformer block
lm_head
或现有可训练 adapter
```

不要直接保存完整高维梯度。实现确定性的 random projection 或 CountSketch，将每个样本投影到 512 或 1024 维。

## 5.4 采样

每个 error type 至少随机采样 128 条，保持 prompt 不重叠。分别在：

```text
SFT initialization
训练 10% checkpoint（若存在）
final DPO checkpoint
```

收集梯度。

## 5.5 指标

按类型 \(e_i,e_j\) 计算：

```text
mean cosine
median cosine
negative cosine rate
within-type variance
between-type centroid cosine
effective gradient rank
```

定义冲突率：

\[
Conflict(e_i,e_j)=P[\cos(g_i,g_j)<0].
\]

生成：

```text
gradient_cosine_heatmap.png
gradient_conflict_heatmap.png
gradient_singular_values.png
gradient_geometry.csv
```

将六个 pairwise mixture 的：

```text
mean cross-type cosine
conflict rate
GDR
Reasoning@128 delta
```

做 Spearman 相关。由于 pair 数只有 6，必须标记为 exploratory mechanism evidence，不得宣称强统计结论。

---

# 6. P0/P1 实验：从 DAG 基本编辑推导扩展误差分类

## 6.1 分类原则

不要将四类错误描述为数学推理错误的完整分类。应把它们定位为 DAG 上四种代表性编辑轴，并扩展为更系统的 graph-edit taxonomy。

建议分类：

| DAG 编辑对象 | 已有/新增错误 | 定义 |
|---|---|---|
| Node value | Computation | 保持操作与依赖，篡改局部计算值 |
| Node operator | Operation-Substitution | 保持父节点，替换 `+/-/*//` 等运算 |
| Edge redirect | Dependency | 将父节点重定向或误用依赖 |
| Edge insertion | Extra-Edge | 为节点加入不应存在的依赖并用于计算 |
| Node deletion | Missing-Node | 删除必要中间节点，但下游继续引用或隐式跳过 |
| Node insertion | Spurious-Node | 插入错误且被后续使用的中间节点 |
| Linearization/order | Disorder | 节点内容保留，但序列不满足拓扑顺序 |
| Query/root | Wrong-Target | 推理出一个合法中间量，但返回错误目标节点 |

## 6.2 新 perturbation API

把现有 perturbation 统一到接口：

```python
class GraphPerturbation(Protocol):
    name: str

    def applicable(self, graph, rng) -> bool: ...
    def apply(self, graph, rng) -> PerturbationResult: ...
    def validate(self, original_graph, result) -> ValidationReport: ...
```

`PerturbationResult` 至少包含：

```text
perturbed_graph
rendered_trace
error_labels
changed_nodes
changed_edges
error_location
severity
final_answer_changed
validation_report
```

## 6.3 每类错误必须有的 validator

### Operation-Substitution

- 父节点集合不变；
- 至少一个 operator 改变；
- 重新计算后目标值或中间值改变；
- 不同时引入 Dependency 或 Missing 错误。

### Extra-Edge

- 新增至少一条原图不存在的边；
- 新边不制造图环；
- 渲染后的方程实际使用新增父节点；
- 不退化为纯 Dependency redirect。

### Spurious-Node

- 插入节点不在原始图中；
- 插入节点被至少一个下游节点使用；
- 不是不影响结果的无关文本。

### Wrong-Target

- 内部推理可以保持局部正确；
- 最终 answer 指向非 query node 或错误目标；
- 明确标记为 query/root 错误。

## 6.4 扩展诊断

先用现有模型对新增错误做 diagnostic preference，成本较低：

```text
SFT
DPO-Random
DPO-Compute
DPO-Dependency
DPO-Missing
DPO-Disorder
DPO-Mix
```

评价新列：

```text
Operation
Extra-Edge
Spurious-Node
Wrong-Target
```

输出完整 transfer matrix：训练类型 × 诊断错误类型。

## 6.5 新增单误差训练

算力允许时，至少训练：

```text
DPO-Operation
DPO-WrongTarget
```

优先这两类，因为它们分别扩展 node semantics 和 query/root 维度，与现有四类差异最大。

新增结果用于回答：

- Missing-Node 是否仍然优于更广泛的错误类型；
- 现有结论是否只在四类人工选择上成立；
- 不同 graph-edit 轴是否产生可解释的 transfer pattern。

---

# 7. P1 实验：错误位置和 DAG 拓扑外推

审稿人询问是否存在某类 DAG 即使有足够训练数据也难以学习。除零空间实验外，再测试结构分布外推。

## 7.1 Error-location split

为每次 corruption 保存节点归一化深度：

\[
d(v)=\frac{\text{depth}(v)}{\max_u\text{depth}(u)}.
\]

构造：

```text
Shallow train: d <= 0.4
Deep test: d >= 0.6

Deep train: d >= 0.6
Shallow test: d <= 0.4
```

中间区域可丢弃，减少边界重叠。

## 7.2 Topology split

根据 DAG 结构划分：

```text
chain: max out-degree <= 1 and low width
branch: at least one branching node
diamond/merge: at least one node with in-degree >= 2 and multiple paths
```

至少完成：

```text
train chain -> test branch/merge
train branch/merge -> test chain
```

## 7.3 指标

- isolated diagnostic accuracy；
- Answer@128；
- Reasoning@128；
- Retention；
- generalization gap；
- 与同分布测试相比的相对下降。

该实验用于区分：

- 学到抽象错误约束；
- 只记住特定节点位置、深度或 DAG 模板。

---

# 8. P1/P2：自然模型错误覆盖分析

该部分直接回应“四类错误是否人工、是否覆盖真实推理失败”。Codex 负责生成、解析、启发式标注和人工标注队列，不替代作者人工审阅。

## 8.1 错误样本生成

从至少两个模型采样：

```text
SFT
DPO-Random
```

可选加入 Base 和 DPO-Mix。

建议流程：

- 从未用于训练的 prompts 采样；
- 每题生成 4–8 个 response；
- 过滤 final answer 或 Reasoning evaluator 判错的 response；
- 去重；
- 目标保留 500–1000 条 parsed wrong traces。

## 8.2 自动图差异标签

将预测 trace 与 gold DAG 比较，生成多标签：

```text
value_mismatch -> Computation
operator_mismatch -> Operation-Substitution
edge_redirect -> Dependency
extra_edge -> Extra-Edge
missing_required_node -> Missing-Node
extra_used_node -> Spurious-Node
order_violation -> Disorder
wrong_query_output -> Wrong-Target
parse_failure -> Unparsed
other -> Unknown
```

自然错误通常是 multi-label，不得强制单标签。

## 8.3 人工标注队列

输出：

```text
outputs/rebuttal_taxonomy/natural_error_annotation_queue.csv
```

列：

```text
item_id
prompt_id
prompt
gold_trace
model_response
gold_answer
pred_answer
heuristic_labels
parse_status
author1_labels
author2_labels
adjudicated_labels
notes
```

建议人工双标至少 200 条；Codex 只需提供可编辑 CSV 和汇总脚本。

## 8.4 汇总指标

```text
parse_rate
taxonomy_coverage_among_parsed
taxonomy_coverage_overall
single_label_rate
multi_label_rate
unknown_rate
frequency_by_error
frequency_by_model
```

若完成人工双标，再报告：

```text
Cohen's kappa per label
macro F1 of heuristic labels vs adjudicated labels
```

结论必须按结果表述：

- 高覆盖：说明 graph-edit taxonomy 对自然错误具有一定外部有效性；
- 低覆盖：说明当前分类仅是受控干预集合，论文需明确限制，不能声称代表一般数学推理错误。

---

# 9. 数据与评估防泄漏要求

1. train、diagnostic test、generation test 必须按 prompt ID 或原始 DAG family 分组切分。
2. 同一正确 trace 的不同 corruption 不得跨 train/test。
3. Confounded 和 Orthogonal 条件必须使用相同基础 prompt pool，但正式评价使用独立 prompt pool。
4. perturbation 位置选择规则只能根据训练数据确定，不得查看测试表现后修改。
5. 所有数据文件写入 SHA256 manifest。
6. 所有结果文件保存：
   - git commit；
   - config hash；
   - dataset hash；
   - checkpoint path；
   - model revision；
   - tokenizer revision；
   - seed；
   - CUDA/PyTorch/Transformers/TRL 版本。

---

# 10. 推荐目录结构

Codex 应根据现有仓库结构调整，但新增内容建议集中在：

```text
rebuttal/
  REPO_AUDIT.md
  IMPLEMENTATION_PLAN.md
  EXPERIMENT_STATUS.md
  identifiability/
    feature_schema.py
    build_design_matrix.py
    matrix_diagnostics.py
    generate_factorial_pairs.py
  controls/
    surface_audit.py
    metadata_classifier.py
    matching.py
    length_normalized_dpo.py
  gradient/
    collect_per_example_gradients.py
    gradient_projection.py
    gradient_geometry.py
    correlate_with_transfer.py
  taxonomy/
    base.py
    operation_substitution.py
    extra_edge.py
    spurious_node.py
    wrong_target.py
    validate_perturbations.py
    natural_error_diff.py
  evaluation/
    eval_identifiability.py
    eval_expanded_diagnostic.py
    eval_location_topology.py
  analysis/
    aggregate_identifiability.py
    aggregate_controls.py
    aggregate_gradient.py
    aggregate_taxonomy.py
    statistical_tests.py
  tests/
    test_design_matrix.py
    test_factorial_generation.py
    test_perturbation_validity.py
    test_split_leakage.py
    test_length_normalized_loss.py
    test_gradient_projection.py

configs/rebuttal/
data/rebuttal/
outputs/rebuttal_identifiability/
outputs/rebuttal_controls/
outputs/rebuttal_gradient/
outputs/rebuttal_taxonomy/
```

不要复制已有 DAG parser、renderer 或 evaluator；通过 import 复用。

---

# 11. CLI 设计

具体命令风格应适配现有仓库。至少提供等价功能：

```bash
# 1. 仓库和数据审计
python -m rebuttal.audit_repo --repo_root .
python -m rebuttal.controls.surface_audit \
  --data data/current_structured_preferences \
  --out outputs/rebuttal_controls/surface_audit

# 2. 生成 confounded / orthogonal 数据
python -m rebuttal.identifiability.generate_factorial_pairs \
  --base_data <OP10_CORRECT_TRACE_DATA> \
  --error_a missing_node \
  --error_b disorder \
  --regimes confounded,orthogonal_size,orthogonal_exposure,factorial \
  --seed 414 \
  --out data/rebuttal/missing_disorder

# 3. 设计矩阵审计
python -m rebuttal.identifiability.build_design_matrix \
  --data data/rebuttal/missing_disorder/confounded.jsonl \
  --out outputs/rebuttal_identifiability/missing_disorder/confounded

# 4. 训练
python <EXISTING_DPO_ENTRYPOINT> \
  --config configs/rebuttal/ident_missing_disorder_confounded.yaml

# 5. 独立轴诊断
python -m rebuttal.evaluation.eval_identifiability \
  --checkpoint <CHECKPOINT> \
  --eval_data data/rebuttal/eval_missing_disorder.jsonl \
  --out outputs/rebuttal_identifiability/...

# 6. 梯度几何
python -m rebuttal.gradient.collect_per_example_gradients \
  --checkpoint <SFT_OR_DPO_CHECKPOINT> \
  --data <PREFERENCE_DATA> \
  --per_type 128 \
  --projection_dim 1024 \
  --out outputs/rebuttal_gradient/...

# 7. 扩展分类诊断
python -m rebuttal.evaluation.eval_expanded_diagnostic \
  --checkpoint <CHECKPOINT> \
  --error_types computation,dependency,missing_node,disorder,operation_substitution,extra_edge,spurious_node,wrong_target \
  --out outputs/rebuttal_taxonomy/...

# 8. 汇总
python -m rebuttal.analysis.aggregate_identifiability --root outputs/rebuttal_identifiability
python -m rebuttal.analysis.aggregate_controls --root outputs/rebuttal_controls
python -m rebuttal.analysis.aggregate_gradient --root outputs/rebuttal_gradient
python -m rebuttal.analysis.aggregate_taxonomy --root outputs/rebuttal_taxonomy
```

每个 CLI 必须支持：

```text
--seed
--max_samples
--dry_run
--overwrite false
--log_level
```

---

# 12. 单元测试与数据不变量

## 12.1 Perturbation tests

每种新旧 perturbation 至少 20 个随机 DAG 单元测试，检查：

- 输出可解析；
- 原 chosen 不变；
- 只发生声明的图编辑；
- graph remains DAG，除非错误定义明确允许破坏 DAG；
- final answer correctness 状态与 metadata 一致；
- render 后重新 parse 可恢复声明的错误；
- 无空 rejected；
- 无 chosen == rejected。

## 12.2 Confounded data tests

```text
所有样本同时具有 A 和 B
A/B 子矩阵 rank == 1
prompt/chosen 与 orthogonal 条件一致
无 train/test prompt 重叠
```

## 12.3 Orthogonal data tests

```text
A-only 样本不包含 B
B-only 样本不包含 A
A/B 子矩阵 rank == 2
两类样本数量符合配置
```

## 12.4 Length-normalized DPO tests

使用人工 token log-probability 张量，验证：

```text
sum reduction 与当前标准实现一致
mean reduction 等于 response-token 平均
padding token 不参与分母
chosen/rejected mask 正确
```

## 12.5 Gradient tests

- 同一 seed 的 projection 完全可复现；
- 不同 seed projection 不完全相同；
- gradient norm 非零；
- 不保存 optimizer state 或巨型完整梯度；
- microbatch=1 与单样本手算结果一致。

---

# 13. 运行优先级与最小交付

## P0：必须完成

1. `REPO_AUDIT.md` 和代码路径映射；
2. 结构特征 schema 与设计矩阵 rank/SVD 审计；
3. Missing+Disorder 的 Confounded vs Orthogonal 三种条件；
4. Computation+Dependency 的最小复现实验；
5. isolated A/B/AB diagnostic evaluation；
6. surface/length audit；
7. length-normalized DPO 训练选项及四类核心对照；
8. 扩展 graph-edit taxonomy 的实现与 validator；
9. 现有模型在新增错误上的 diagnostic matrix；
10. 自动汇总表和统计检验。

## P1：强烈建议

1. 三 seed 全量；
2. gradient geometry；
3. error-location OOD；
4. topology OOD；
5. Operation 和 Wrong-Target 单误差训练；
6. natural error 自动标注和人工队列。

## P2：时间允许

1. Extra-Edge / Spurious-Node 单误差训练；
2. 双作者自然错误标注统计；
3. 更大模型复现；
4. 不同 beta 或 DPO variant 复现。

---

# 14. 结果文件清单

完成后必须存在：

```text
rebuttal/REPO_AUDIT.md
rebuttal/IMPLEMENTATION_PLAN.md
rebuttal/EXPERIMENT_STATUS.md

outputs/rebuttal_identifiability/table_identifiability.csv
outputs/rebuttal_identifiability/table_identifiability_summary.csv
outputs/rebuttal_identifiability/design_rank_vs_axis_accuracy.png
outputs/rebuttal_identifiability/isolated_axis_accuracy.png

outputs/rebuttal_controls/surface_statistics.csv
outputs/rebuttal_controls/metadata_classifier.json
outputs/rebuttal_controls/matching_balance.csv
outputs/rebuttal_controls/length_normalized_dpo.csv

outputs/rebuttal_gradient/gradient_geometry.csv
outputs/rebuttal_gradient/gradient_cosine_heatmap.png
outputs/rebuttal_gradient/gradient_conflict_heatmap.png
outputs/rebuttal_gradient/gradient_gdr_correlation.csv

outputs/rebuttal_taxonomy/expanded_diagnostic_matrix.csv
outputs/rebuttal_taxonomy/expanded_diagnostic_heatmap.png
outputs/rebuttal_taxonomy/natural_error_auto_labels.jsonl
outputs/rebuttal_taxonomy/natural_error_annotation_queue.csv
outputs/rebuttal_taxonomy/natural_error_coverage.csv

outputs/rebuttal_summary/rebuttal_key_numbers.json
outputs/rebuttal_summary/rebuttal_tables.md
outputs/rebuttal_summary/rebuttal_claim_checklist.md
```

`rebuttal_key_numbers.json` 应使用稳定字段名，方便直接填入 rebuttal：

```json
{
  "missing_disorder": {
    "confounded_rank": null,
    "orthogonal_rank": null,
    "confounded_diag_missing": null,
    "confounded_diag_disorder": null,
    "orthogonal_diag_missing": null,
    "orthogonal_diag_disorder": null,
    "worst_axis_gain": null
  },
  "compute_dependency": {},
  "length_control": {
    "missing_reasoning_standard": null,
    "missing_reasoning_mean_logp": null
  },
  "taxonomy": {
    "natural_parse_rate": null,
    "natural_coverage": null,
    "unknown_rate": null
  }
}
```

---

# 15. 自动生成的 rebuttal 证据模板

Codex 不应伪造结果，但可在所有结果产生后，根据 JSON 自动生成以下带占位符的 Markdown：

```markdown
We agree that the basic DPO objective predicts sensitivity to observed contrastive dimensions, and we have revised our claim accordingly. Our new analysis asks a stricter question: when is a structural error dimension identifiable from pairwise preference data?

Under a locally linear structural utility model, DPO depends on the preference-difference design matrix X. Any direction in Null(X) is observationally unidentifiable. We empirically test this by comparing rank-1 confounded negatives, in which [A] and [B] always co-occur, with rank-2 orthogonal negatives using the same prompts, chosen responses, initialization, and optimization settings.

The confounded design achieved [X]% accuracy on joint A+B corruption but only [Y]% / [Z]% on isolated A and B corruptions, whereas the orthogonal design improved the worst-axis accuracy by [D] points. This shows that the result is not merely that DPO learns whichever feature is varied: the co-occurrence structure of the preference data determines whether individual reasoning constraints are identifiable.

We also expanded the original four corruptions into a graph-edit taxonomy over node values, node operators, edges, vertices, topological order, and query nodes. On naturally generated wrong traces, the taxonomy covers [C]% of parsed errors, with [M]% multi-label cases and [U]% remaining unknown. We now present the original four types as representative controlled interventions rather than an exhaustive taxonomy.
```

若实验不支持上述方向，模板必须自动改为中性描述，不允许硬填正向结论。

---

# 16. 论文主张的修改边界

新增实验完成后，论文应避免以下过强主张：

```text
错误类型本身决定模型一定学到对应能力。
四类错误覆盖数学推理中的主要失败模式。
DPO-Missing 的优势完全来自 reasoning completeness，而不存在长度或表面形式因素。
```

可支持的更精确表述：

```text
1. Pairwise preference data constrain only structural directions represented in chosen–rejected differences.
2. Perfectly confounded structural errors are not separately identifiable under the observed preference design.
3. Orthogonalized negative construction improves isolated-constraint recovery relative to confounded construction, if supported by results.
4. Different graph-edit errors induce different transfer and gradient-interference patterns beyond training-pair fit.
5. The original four errors are representative controlled interventions over distinct DAG edit axes, not an exhaustive taxonomy.
6. Missing-Node results remain robust only if they survive length-normalized and matched-data controls.
```

---

# 17. 完成标准

Codex 只有在以下条件全部满足后，才标记 P0 为完成：

- 所有新增脚本具有 `--help`；
- smoke tests 通过；
- unit tests 通过；
- 数据 split 无泄漏；
- confounded/orthogonal 设计矩阵的秩符合定义；
- 原始 checkpoint 和结果未被覆盖；
- 至少一个完整错误对完成三 seed 或明确记录资源限制；
- 所有汇总数字可追溯到原始 prediction 文件；
- 失败运行和排除样本都有日志；
- `EXPERIMENT_STATUS.md` 明确列出已完成、未完成、失败和探索性项目。

---

# 18. Codex 首轮应返回的内容

在开始大规模实现前，Codex 首轮只需完成仓库审计并返回：

1. 当前仓库目录树摘要；
2. 现有四类 perturbation 的代码位置；
3. 当前 DPO trainer 入口及 log-probability reduction 方式；
4. 当前 diagnostic preference 入口；
5. 当前 Reasoning@k evaluator 入口；
6. 数据 JSONL 的实际 schema；
7. checkpoint 命名和加载方式；
8. 与本规范建议路径的映射；
9. P0 实现文件列表；
10. 预计需要作者确认的变量。

对无法从代码确定的事项，只列出问题，不要自行猜测。
