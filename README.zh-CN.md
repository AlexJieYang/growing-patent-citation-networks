# 专利网络增长与传递闭包

[English](README.md)

本仓库对应研究 **Transitive closure and target reuse in growing patent citation networks**。包含六本 Python Notebook 和同等计算内容的命令行入口，使用 pandas、NumPy、SciPy、NetworkX 和 Matplotlib。

## 安装与合成示例

使用 Python 3.11 或以上版本，在仓库目录创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 用 `.venv\Scripts\Activate.ps1` 激活；Linux/macOS 用 `source .venv/bin/activate`。随后运行：

```bash
python -m pip install -r requirements.txt
python run_all.py --demo
python -m pytest -q
```

示例不需要真实专利数据，会执行整个流程并生成十幅组图，保存在 `outputs/demo_physics/`。图上有 `SYNTHETIC DATA` 标记。示例使用较小重复次数和 300 dpi；真实研究默认仍为 30 次增长模拟、199 个零模型状态和 600 dpi。

## 真实数据

输入是已有研究层 Parquet 表及索引化的 NPY 矩阵，字段见 [data/README.md](data/README.md)。本仓库不包含大型数据，也不自动下载 USPTO 原始文件。

可以直接传入路径：

```powershell
python run_all.py --source-root "C:\data\PatentsView"
```

也可创建本地配置：

```bash
python -c "from shutil import copyfile; copyfile('config.json', 'config.local.json')"
```

修改 `config.local.json` 中的 `source_root`，再运行：

```bash
python run_all.py --config config.local.json
```

若需保留多次结果，使用不同运行名：

```bash
python run_all.py --config config.local.json --run-name comparison_2025
```

每次完整运行会重新建立对应的 `outputs/<run_name>/`，原始输入须位于其外。程序先检查所需文件是否存在；实际配置保存于 `outputs/`，不把本地路径写回公共 `config.json`。

## Notebook

运行 `jupyter lab`，按 **00 → 01 → 02 → 03 → 04 → 05** 从头执行。00 优先读取 `config.local.json`，否则读取 `config.json`；也可在其中设置 `SOURCE_ROOT`、`RUN_NAME`。后续读取活动配置。从合成示例切换真实数据时重新从 00 开始。若显式设置了 `PRE_CONFIG` 环境变量，后续配置加载优先使用该变量。

| 文件 | 内容 |
|---|---|
| `00_Data_Preparation.ipynb` | CPC 筛选、分块读取引用、向量对齐、有效截止日 |
| `01_Network_Structure.ipynb` | 到达过程、度、连通分量、聚类、核层与闭包 |
| `02_Attachment_Opportunities.ipynb` | 历史风险集合、富集率、来源专利 bootstrap |
| `03_Diffusion_and_Connectivity.ipynb` | 扩散、谱熵、节点移除与边插入 |
| `04_Growth_and_Null_Models.ipynb` | 三个嵌套增长机制与时间约束拓扑零模型 |
| `05_Publication_Figures.ipynb` | 正文五幅、SI 五幅组图及独立图注 |

## 研究口径

默认比较 neural 与 laser 两个 CPC 题域，授权年份为 1980—2025；请求截止日为 2025-12-31，实际截止日受来源覆盖和元数据日期约束。社群使用 2000 年以前的参考文献并保持冻结。增长窗口统一为 2023—2025，三个机制依次加入度与年龄、语义成本、社群成本。具体定义见 [METHODS.md](METHODS.md)。

论文的语义表征为标题与摘要拼接文本的 768 维 PatentSBERTa 向量，方法文献 DOI 为 10.1016/j.techfore.2024.123536。程序读取预计算向量，不训练语言模型，按 `patent_id` 对齐并归一化。专利记录来自 USPTO，经 PatentsView 获取。

保留面向工作站的有界设置：每领域最多 30,000 节点，含引用边界最多 600,000 节点、保留边最多 200 万、选中向量最多 3 GiB、增长候选对最多 1 亿。引用按 20 万行分块，数值线程为 4。各预算约束不同计算对象，并非总内存上限。结果均在代码同目录的 `outputs/` 子文件夹中。

## 源码检查与提交

Notebook、`prelib/` 和 `config.json` 保持同级。所有 Notebook 均不保存执行输出或执行次数。运行后提交源码前，可执行：

```bash
python tools/clean_notebooks.py
python tools/check_sources.py
```

GitHub Actions 仅检查 Python 语法、JSON、Notebook 结构及空输出状态，不执行 Notebook、研究测试或模拟。已有科学测试可在配置好依赖后主动运行 `python -m pytest -q`。`.gitignore` 排除本地输入、配置、生成结果和缓存。作者与引用信息见 `CITATION.cff`。

## 许可

研究代码采用 [MIT 许可](LICENSE)。外部专利数据、嵌入向量和其他第三方材料仍适用各自的许可与获取条件。
