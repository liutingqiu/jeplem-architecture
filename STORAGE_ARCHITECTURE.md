# 分层存储架构 — 代码/探索数据 → GitHub,训练大数据 → OSS/COS

> 性质：存储决策文档。本机磁盘小 + 老电脑报废(无免费外部存储),故按分层方案落地。
> 日期：2026-08-12

## 一、决策

| 数据类型 | 存储 | 理由 |
|---|---|---|
| **代码** | GitHub 私库 | 小、版本重要、免费 |
| **探索数据**（checkpoint/status/learned rules 报告） | GitHub 私库（gitignored 运行态不入库） | 中等大小、审计需要 |
| **训练大数据**（未来模型训练语料） | **阿里云 OSS / 腾讯云 COS** | GB~TB 级,GitHub 超限 |

## 二、Git 规则（探索数据如何入库）

- **入库**（GitHub）：代码、实验报告、learned-rules 快照、判据结果
- **不入库**（gitignored 运行态）：
  - `*-checkpoint.json`（探索状态,持续膨胀）
  - `*-status.json`（实时监测,持续重写）
  - `*.log`（运行日志）
  - `STOP` 信号文件

见 `.gitignore` 的 `Runtime state files` 段 + 根目录运行态段。

## 三、训练数据接入接口（未来 OSS/COS）

- **选型**：阿里云 OSS 或腾讯云 COS 标准存储（~0.1 元/GB/月）
- **成本**：1TB ≈ 100-120 元/月；上传免费,下行流量 ~0.5 元/GB
- **接入**：训练数据用 SDK(oss2 / cos-python)写入 bucket,训练时拉取;
  探索数据继续 GitHub 不占云额度
- **时机**：探索积累到 GB 级 / 开始真实训练时启用;当前 63MB 用 GitHub 免费额度足够

## 四、当前状态

- 三个工作区共 ~63MB,最大单文件 2.4MB —— GitHub 免费额度内
- 探索 500 轮进行中,checkpoint 预计增长到几十 MB(仍在 GitHub 额度内)
- 老电脑报废:无免费外部存储,训练数据必须走云
