# VisThoughts

**VisThoughts** 在开源评测框架 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 之上，验证视觉思维链（Visual Chain-of-Thought）方法：通过两阶段注意力引导（关键词挖掘 → 注意力头筛选 → 区域增强推理）提升多模态模型在视觉问答上的表现。

Python 包名仍为 `vlmeval`（兼容原框架接口与评测脚本）。

## 方法概览

1. **Stage-1 / Head Mining**：从问题中抽取视觉定位关键词，并在少量样本上挖掘与关键词区域对齐的注意力头。
2. **Stage-2 / Inference**：用筛选出的注意力头引导全量推理，并可关闭调试落盘以做大规模评测。

## License

本项目遵循 Apache License 2.0。上游版权归属 VLMEvalKit Authors；VisThoughts 的修改部分版权归本仓库贡献者所有。详见 [LICENSE](LICENSE)。

## 致谢

感谢 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 与 [OpenCompass](https://github.com/open-compass) 提供的开源评测基础设施。
