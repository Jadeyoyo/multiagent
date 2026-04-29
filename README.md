# Forging Intelligent Maintenance Multi-Agent Prototype

这是一个面向“锻造智能维护”的多智能体协同闭环原型工程。它严格遵循如下设计原则：

1. **智能体不直接诊断原始信号**：底层 PHM 模型池负责处理设备信号，智能体只进行场景理解、模型编排、知识检索、结果解释、验证和反馈更新。
2. **所有模型调用经过统一模型池接口**：模型版本、适用任务、输入模态、标签空间、可靠性、延迟和未知故障检测能力都记录在算法画像中。
3. **所有维护建议附带证据链**：输出包含模型版本、关键输出、知识来源、验证结果和人工复核标记。
4. **高风险动作需要人工复核**：系统只给出建议，不直接执行停机、参数修改或危险动作。

## 目录结构

```text
forge_multiagent_maintenance/
├─ main.py
├─ requirements.txt
├─ config/
│  └─ model_profiles.json
├─ knowledge/
│  └─ forging_knowledge.json
├─ examples/
│  ├─ sample_event_high_risk.json
│  ├─ sample_event_low_risk.json
│  ├─ sample_signal_abnormal.csv
│  └─ sample_signal_normal.csv
├─ outputs/
└─ forge_maint/
   ├─ schemas.py
   ├─ utils.py
   ├─ model_pool.py
   ├─ knowledge_base.py
   ├─ digital_twin.py
   ├─ controller.py
   └─ agents/
      ├─ scenario_understanding.py
      ├─ model_orchestration.py
      ├─ knowledge_retrieval.py
      ├─ fusion_explanation.py
      ├─ validation.py
      └─ update.py
```

## 运行方式

该原型只依赖 Python 标准库。建议 Python 3.9+。

```bash
cd forge_multiagent_maintenance
python main.py --event examples/sample_event_high_risk.json
```

低风险样例：

```bash
python main.py --event examples/sample_event_low_risk.json
```

带反馈更新的运行方式：

```bash
python main.py --event examples/sample_event_high_risk.json --feedback "专家确认需要检查导轨和液压系统" --actual-fault-label "bearing_or_guiding_system_fault" --maintenance-result "已完成检查和紧固"
```

输出结果会保存到 `outputs/`，其中包括完整证据链和反馈记录。

## 如何替换为真实 PHM 模型

在 `forge_maint/model_pool.py` 中新增一个继承 `BasePHMModel` 的类：

```python
class MyDeepPHMModel(BasePHMModel):
    def predict(self, scenario, signal):
        # 1. 读取 signal 或多模态数据
        # 2. 调用你的 CNN/Transformer/Domain Adaptation 模型
        # 3. 返回 ModelOutput
        ...
```

然后在 `MODEL_CLASS_REGISTRY` 中注册：

```python
MODEL_CLASS_REGISTRY = {
    "my_deep_phm_model": MyDeepPHMModel,
    ...
}
```

最后在 `config/model_profiles.json` 中新增对应算法画像。

## 六类智能体对应关系

| 论文角色 | 代码文件 | 职责 |
|---|---|---|
| 场景理解智能体 | `scenario_understanding.py` | 将原始事件转换为结构化场景 `S_t` |
| 模型编排智能体 | `model_orchestration.py` | 基于算法画像选择单模型或多模型调用计划 |
| 知识检索智能体 | `knowledge_retrieval.py` | 从维修手册、FMEA、历史案例和算法画像中检索证据 |
| 结果融合与解释智能体 | `fusion_explanation.py` | 融合多模型输出，生成解释、冲突和不确定性说明 |
| 验证智能体 | `validation.py` + `digital_twin.py` | 检查维护建议是否违反物理/工艺/安全约束 |
| 更新智能体 | `update.py` | 记录人工反馈、实际维修结果并生成更新触发项 |

## 适合写入论文的方法描述

该代码可以作为“Agent-assisted closed-loop intelligent maintenance framework”的工程化原型。论文中可以强调它不是用大语言模型替代底层故障诊断模型，而是将智能体作为高层认知、知识增强和工具编排模块；底层诊断仍由 PHM 模型池完成，智能体通过受约束工具调用完成场景理解、模型选择、证据检索、解释、验证和持续更新。
