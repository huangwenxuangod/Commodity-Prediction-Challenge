# 🚀 hwx - Kaggle竞赛评估框架

## 📖 概述

`hwx` 是一个专为Kaggle竞赛设计的评估框架，提供了完整的gRPC通信、数据处理和竞赛网关功能。该框架基于Kaggle的官方评估系统构建，支持多种数据格式和高效的推理服务。

## 🏗️ 架构设计

```
hwx/
├── core/                           # 核心功能模块
│   ├── base_gateway.py            # 基础网关实现
│   ├── relay.py                   # gRPC通信中继
│   ├── templates.py               # 竞赛模板类
│   ├── kaggle_evaluation.proto   # Protocol Buffers定义
│   └── generated/                 # 自动生成的protobuf代码
├── mitsui_gateway.py              # 三井商品预测竞赛网关
├── mitsui_inference_server.py     # 三井竞赛推理服务器
├── learning/                      # 学习模块（小白指南）
├── requirements.txt               # 核心依赖包
└── README.md                      # 本文档
```

## 🔧 核心功能

### 1. gRPC通信框架
- 高效的客户端-服务器通信
- 支持多种数据类型（DataFrame、numpy数组、polars等）
- 自动重试和错误处理
- 可配置的超时和连接参数

### 2. 竞赛网关系统
- 标准化的数据加载接口
- 自动数据验证和错误处理
- 支持文件共享和批量处理
- 竞赛特定的验证逻辑

### 3. 推理服务器
- 灵活的端点定义
- 自动服务管理
- 本地测试支持
- 性能监控和优化

## 🚀 快速开始

### 环境准备

#### 方法1：使用conda（推荐）

1. **创建conda环境**:
```bash
cd hwx
conda env create -f environment.yml
```

2. **激活环境**:
```bash
conda activate hwx-env
```

#### 方法2：使用pip

1. **安装依赖包**:
```bash
cd hwx
pip install -r requirements.txt
```

2. **验证安装**:
```python
import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.relay
import kaggle_evaluation.core.templates
print("✅ hwx核心包安装成功！")
```

### 基本使用

#### 创建竞赛网关

```python
from kaggle_evaluation.core.templates import Gateway
import pandas as pd
import polars as pl

class MyCompetitionGateway(Gateway):
    def __init__(self, data_paths=None):
        super().__init__(data_paths)
        self.row_id_column_name = 'id'
        self.set_response_timeout_seconds(300)
    
    def unpack_data_paths(self):
        if not self.data_paths:
            self.data_dir = '/kaggle/input/my-competition/'
        else:
            self.data_dir = self.data_paths[0]
    
    def generate_data_batches(self):
        # 实现数据批处理逻辑
        test_data = pl.read_csv(f"{self.data_dir}/test.csv")
        for batch in test_data.iter_chunks():
            yield batch, batch['id']
    
    def competition_specific_validation(self, predictions, row_ids, data_batch):
        # 实现竞赛特定的验证逻辑
        assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
        assert len(predictions) == len(row_ids)
```

#### 创建推理服务器

```python
from kaggle_evaluation.core.templates import InferenceServer

def predict_function(data_batch):
    # 实现你的预测逻辑
    # 返回预测结果
    return predictions

# 创建服务器
server = InferenceServer(predict_function)

# 启动服务
if __name__ == "__main__":
    server.serve()
```

## 📊 支持的数据类型

### 输入数据类型
- `pandas.DataFrame` / `pandas.Series`
- `polars.DataFrame` / `polars.Series`
- `numpy.ndarray`
- `list`, `tuple`, `dict`
- 基本Python类型（int, float, str, bool）

### 输出数据类型
- 所有输入类型都支持作为输出
- 自动类型转换和验证
- 支持嵌套数据结构

## 🔍 高级功能

### 1. 文件共享
```python
# 共享大文件以提高性能
self.share_files(['/path/to/large/file.csv'])
```

### 2. 自定义验证
```python
def competition_specific_validation(self, predictions, row_ids, data_batch):
    # 检查预测格式
    assert predictions.shape[1] == expected_columns
    
    # 检查数值范围
    assert predictions.min().min() >= 0
    assert predictions.max().max() <= 1
```

### 3. 错误处理
```python
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType

try:
    # 你的代码
    pass
except Exception as e:
    raise GatewayRuntimeError(
        GatewayRuntimeErrorType.SERVER_RAISED_EXCEPTION,
        f"处理失败: {str(e)}"
    )
```

## 🧪 测试和调试

### 本地测试
```python
# 使用本地数据测试
gateway = MyCompetitionGateway(['/local/path/to/data'])
gateway.run_local_gateway(['/local/path/to/data'])
```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
kaggle_evaluation.core.relay.set_log_level(logging.DEBUG)
```

## 📈 性能优化

### 1. 批量处理
- 使用适当的数据批次大小
- 避免单条记录处理

### 2. 内存管理
- 及时释放不需要的数据
- 使用流式处理处理大文件

### 3. 连接优化
- 调整gRPC参数
- 使用连接池

## 🚨 常见问题

### Q: 如何解决gRPC连接问题？
A: 检查端口配置、防火墙设置，确保服务正在运行。

### Q: 如何处理大数据集？
A: 使用`share_files`方法或分批处理数据。

### Q: 如何自定义错误消息？
A: 使用`GatewayRuntimeError`类抛出标准化的错误。

### Q: 如何优化推理性能？
A: 使用模型缓存、批处理、异步处理等技术。

## 🔗 相关资源

- [Kaggle竞赛平台](https://www.kaggle.com/)
- [gRPC官方文档](https://grpc.io/docs/)
- [Protocol Buffers文档](https://developers.google.com/protocol-buffers)
- [polars文档](https://pola.rs/)

## 🤝 贡献指南

欢迎贡献代码和提出建议！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 支持

如有问题或建议，请：
- 提交Issue
- 联系维护者
- 查看文档

---

**记住**: 这是一个专业的竞赛框架，设计用于生产环境。请确保在生产使用前充分测试！ 🚀
