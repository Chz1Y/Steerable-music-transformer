# Steerable Rhythmic Complexity in Autoregressive Music Generation 🎹🤖

[![Paper Accepted](https://img.shields.io/badge/Paper-EI_Conference-success.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*[English version below | 中文版本见下]*

---

## 📖 About The Project (English)

Welcome! This repository contains the official implementation of the paper **"Steerable Rhythmic Complexity in Autoregressive Music Generation: A Bar-Level Conditional Transformer Approach"** (Accepted by EI Conference). 

Ever wondered how to give a neural network a "gearbox" and a "brake pedal" for composing music? Controllable music generation typically suffers from feature entanglement—where increasing rhythmic speed inevitably destroys harmonic stability. 

This project introduces a lightweight, highly accurate conditioning mechanism based on the **REMI+ syntax**. By explicitly injecting complexity tags (`[Level_1]` to `[Level_10]`) at the bar level and strictly enforcing `[EOS]` stopping mechanisms, we trained a micro-Transformer (4 layers, 8 heads) from scratch on 349 J.S. Bach four-part chorales. The result? A model that allows you to intuitively "steer" the rhythmic busyness of the generated music while keeping the classical harmonic integrity rock solid.

### 🌟 Key Highlights
* **Zero Gear-Slipping:** Achieved a profoundly strong linear correlation (Pearson *r* = 0.893) between target complexity and actual note density.
* **Feature Decoupling:** Proved via Pitch Shannon Entropy that rhythmic complexity can be manipulated independently of harmonic noise.
* **Micro-Architecture:** Proves that with high-purity data and precise temporal-window chunking, a massive LLM is not required to achieve deterministic control.

### 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: Run `build_dataset.py` to chunk MIDI files and inject tags.
3. Train the model: Run `train_micro_model.py`.
4. Generate & Evaluate: Run `generate_samples.py` followed by `evaluate_and_plot.py` to reproduce the academic figures.

### 👨‍💻 Author
**Ziyi Chen** Electrical and Computer Engineering (ECE) & Computer Science researcher. Passionate about the intersection of Robotics, Computer Science, and the arts. 

---

## 📖 项目简介 (Chinese)

欢迎来到本仓库！这里是论文 **《自回归音乐生成中的可控节奏复杂度：基于小节级条件 Transformer 的方法》**（已受 EI 会议录用）的官方开源代码。

在这个项目中，我们尝试为神经网络装上“变速箱”和“刹车踏板”。传统的受控音乐生成往往面临特征耦合的困境——节奏一加快，和弦就崩塌。

本项目提出了一种基于 **REMI+ 语法**的轻量级、高精度控制机制。通过在小节级别显式注入复杂度标签（`[Level_1]` 至 `[Level_10]`）并严格引入 `[EOS]` 截断机制，我们在 349 首巴赫四声部合唱曲上从零训练了一个微缩版 Transformer（4 层，8 个注意力头）。最终，模型实现了一个直观的功能：你可以像换挡一样精准控制生成音乐的“节奏紧凑度”，同时完美保持古典音乐的和声稳定性。

### 🌟 核心亮点
* **精准的控制力**：目标复杂度与实际生成音符密度之间实现了极强的正向线性相关（Pearson *r* = 0.893）。
* **特征解耦**：通过音高香农熵（Pitch Shannon Entropy）验证，证明了系统可以在不增加和声噪音的前提下，独立提升节奏复杂度。
* **严密的控制变量实验**：内置了完整的消融实验（Ablation Study）脚本，用以对比无条件基线模型，证明控制机制的不可或缺性。

### 🚀 快速复现
1. 安装依赖包：`pip install -r requirements.txt`
2. 构建数据集：运行 `build_dataset.py` 进行时间窗口切片与标签注入。
3. 启动训练：运行 `train_micro_model.py`。
4. 生成与出图：依次运行 `generate_samples.py` 与 `evaluate_and_plot.py`，一键复现论文中的四张学术级量化图表。

---
*Note: The raw MIDI dataset used in this research is programmatically sourced from the open-source `music21` corpus. Sample outputs can be found in the `audio_demos/` folder.*
