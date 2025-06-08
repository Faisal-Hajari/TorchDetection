# TorchDetection

**"Explicit PyTorch" for Object Detection**

A PyTorch-native component architecture for building detection models with explicit dependency injection. No config magic, no hidden registries—just clear, composable PyTorch code.

## 🎯 Core Philosophy: "Batteries Included, Assembly Required"

TorchDetection provides all the components you need to build modern detection models, but **you** explicitly assemble them. If it's important, it's visible in your code.

### Design Principles

- **Explicit Over Implicit**: All dependencies passed as constructor parameters
- **Composition Over Inheritance**: Maximum 2-level inheritance (Interface → Implementation)  
- **Single Responsibility**: Each component has one clear purpose
- **PyTorch-Native**: Feels like natural PyTorch code, no framework magic

### What This Means in Practice

```python
# ✅ TorchDetection way - explicit and debuggable
from torchdetection.nn.backbones import ResNet50  # Complete backbone
from torchdetection.nn.necks import YOLOv8Neck  # Complete neck implementation
from torchdetection.nn.heads import YOLOHead  # Head using components
from torchdetection.nn.models.yolo import YOLOv8

# Backbones are complete feature extractors
backbone = ResNet50(pretrained=True)

# Necks are complete implementations using atomic components internally
neck = YOLOv8Neck(in_channels=[512, 1024, 2048])  # Uses FPN, SPPF, C3K2 components

# Heads compose conv, attention, normalization components
head = YOLOHead(num_classes=80, anchors=anchors)

# Models compose backbone + neck + head
model = YOLOv8(backbone=backbone, neck=neck, head=head)

# Easy to debug - inspect any component
print(backbone)  # Shows actual PyTorch modules
features = backbone(x)  # Examine intermediate outputs
print(neck.sppf)  # Inspect SPPF component used in neck  
print(neck.fpn)   # Inspect FPN component used in neck

# Example: Neck takes backbone features and returns head features
backbone_features = backbone(x)  # {"stage1": [...], "stage2": [...], "stage3": [...]}
neck_features = neck(backbone_features)  # {"P3": [...], "P4": [...], "P5": [...]}
predictions = head(neck_features)  # {"boxes": [...], "scores": [...], "classes": [...]}
```

```python
# ❌ What we avoid - config magic
cfg = load_config("yolo.yaml")  
model = build_detector(cfg)  # What does this create? 🤷‍♂️
```

## 🏗️ Architecture Overview

```
torchdetection/
├── nn/
│   ├── components/           # Atomic building blocks
│   │   ├── conv.py          # ConvBlock, DWConv, DepthwiseConv
│   │   ├── attention.py     # SE, CBAM, ECA, C2PSA modules
│   │   ├── residual.py      # ResidualBlock, BottleneckCSP, C3K2
│   │   ├── pooling.py       # SPPF, SPP, AdaptivePool, MaxPool
│   │   ├── activation.py    # SiLU, Mish, Swish, custom activations
│   │   ├── normalization.py # BatchNorm, GroupNorm, LayerNorm
│   │   ├── transformer.py   # TransformerBlock, MHA, EfficientRep
│   │   ├── fusion.py        # FPN, PANet, BiFPN, feature fusion blocks
│   │   ├── yolo_blocks.py   # Focus, C3, C2f, specialized YOLO components
│   │   └── utils.py         # Concat, Upsample, utility components
│   ├── backbones/           # Complete feature extraction models
│   │   ├── resnet.py        # ResNet18, ResNet50, ResNet101
│   │   ├── darknet.py       # DarkNet19, DarkNet53
│   │   ├── csp_darknet.py   # CSPDarkNet variants
│   │   ├── yolov8_backbone.py  # YOLOv8 backbone architecture
│   │   ├── yolov11_backbone.py # YOLOv11 backbone with C3K2
│   │   └── efficientnet.py  # EfficientNet variants
│   ├── necks/               # Complete neck implementations using components
│   │   ├── yolov8_neck.py   # YOLOv8 neck (uses FPN, SPPF, C3K2 components)
│   │   ├── yolov11_neck.py  # YOLOv11 neck (uses enhanced components)
│   │   ├── fpn_neck.py      # Generic FPN neck (uses FPN, Conv components)
│   │   ├── bifpn_neck.py    # BiFPN neck (uses BiFPN components)
│   │   └── detr_neck.py     # DETR encoder neck (uses transformer components)
│   ├── heads/               # Task-specific output heads using components
│   │   ├── yolo_head.py     # YOLO detection heads (anchor/anchor-free)
│   │   ├── detr_head.py     # DETR detection + classification head
│   │   ├── retinanet_head.py # RetinaNet classification + regression
│   │   └── rcnn_head.py     # R-CNN RoI classification + regression
│   └── models/              # Complete detection models (backbone+neck+head)
│       ├── yolo/            # YOLO family compositions
│       │   ├── yolov3.py    # DarkNet53 + YOLO neck + YOLO head
│       │   ├── yolov8.py    # YOLOv8 backbone + PANet + anchor-free head
│       │   └── yolov11.py   # YOLOv11 backbone + enhanced neck + head
│       ├── detr/            # DETR family compositions
│       │   ├── detr.py      # ResNet + Transformer + DETR head
│       │   └── deformable_detr.py # ResNet + Deformable Transformer
│       ├── rcnn/            # R-CNN family compositions
│       │   ├── faster_rcnn.py # ResNet + FPN + RPN + RoI head
│       │   └── mask_rcnn.py # Faster R-CNN + mask head
│       └── retinanet/       # Single-stage detector compositions
│           └── retinanet.py # ResNet + FPN + RetinaNet head
├── losses/                  # Loss functions
│   ├── yolo_loss.py        # YOLO-specific losses
│   ├── detr_loss.py        # DETR Hungarian matching
│   ├── focal_loss.py       # Focal loss variants
│   └── iou_loss.py         # IoU-based losses (GIoU, DIoU, CIoU)
├── data/                   # Dataset handling (COCO format focus)
│   ├── coco.py            # COCO dataset implementation
│   ├── transforms.py      # Augmentation pipeline
│   └── collate.py         # Batch collation utilities
├── training/              # Training infrastructure
│   ├── trainer.py         # Main training loop
│   ├── optimizer.py       # Optimizer configurations
│   └── scheduler.py       # Learning rate scheduling
└── utils/                 # Core utilities
    ├── bbox.py           # Bounding box operations
    ├── nms.py            # Non-maximum suppression
    ├── metrics.py        # Evaluation metrics (mAP, etc.)
    └── visualization.py  # Plotting and debugging tools
```

## 🚀 Quick Start Examples

### Building YOLOv8 from Components

```python
import torch
from torchdetection.nn.backbones import YOLOv8Backbone
from torchdetection.nn.necks import YOLOv8Neck
from torchdetection.nn.heads import YOLOHead
from torchdetection.nn.models.yolo import YOLOv8
from torchdetection.losses import YOLOLoss

# Complete backbone (internally composed from nn.components)
backbone = YOLOv8Backbone(depth_multiple=0.33, width_multiple=0.5)  # YOLOv8s

# Complete neck implementation - internally uses SPPF, FPN, C3K2 components  
neck = YOLOv8Neck(in_channels=[256, 512, 1024])

# Head that uses components internally  
head = YOLOHead(num_classes=80, anchors=None)

# Final model composition
model = YOLOv8(backbone=backbone, neck=neck, head=head)

# Define loss - explicit, no magic
criterion = YOLOLoss(
    num_classes=80,
    box_loss_gain=7.5,
    cls_loss_gain=0.5,
    obj_loss_gain=1.0
)
```

### Training Loop

```python
from torchdetection.data import COCODataset
from torchdetection.training import Trainer

# Load COCO dataset (our primary format)
dataset = COCODataset(
    root="./data/coco", 
    split="train2017",
    transforms=transforms
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize trainer with explicit dependencies
trainer = Trainer(
    model=model,
    criterion=criterion, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    train_loader=dataloader,
    device="cuda"
)

# Train model
trainer.train(epochs=100)
```

### Building Other Architectures (Future)

```python
# DETR example - same explicit composition pattern
from torchdetection.nn.backbones import ResNet50
from torchdetection.nn.necks import DETRNeck
from torchdetection.nn.heads import DETRHead
from torchdetection.nn.models.detr import DETR
from torchdetection.losses import DETRLoss

backbone = ResNet50(pretrained=True)
neck = DETRNeck(hidden_dim=256, num_layers=6)
head = DETRHead(num_classes=80, num_queries=100, hidden_dim=256)
model = DETR(backbone=backbone, neck=neck, head=head)
criterion = DETRLoss(num_classes=80, loss_weights={'bbox': 5, 'giou': 2})

# RetinaNet example  
from torchdetection.nn.backbones import ResNet50
from torchdetection.nn.necks import FPN
from torchdetection.nn.heads import RetinaNetHead
from torchdetection.nn.models.retinanet import RetinaNet
from torchdetection.losses import FocalLoss

backbone = ResNet50(pretrained=True)
neck = FPN(in_channels=[512, 1024, 2048])
head = RetinaNetHead(num_classes=80, num_anchors=9)
model = RetinaNet(backbone=backbone, neck=neck, head=head)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

## 🧩 Component Design Philosophy

### Interface Contracts

All components follow standard PyTorch patterns with clear input/output contracts:

```python
# Backbone interface
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:
        x: Input image [B, 3, H, W]
    Returns:
        {"stage1": [B, C1, H/4, W/4], "stage2": [B, C2, H/8, W/8], ...}
    """

# Neck interface  
def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Args:
        features: Output from backbone {"stage1": [...], "stage2": [...], ...}
    Returns:
        Dict of feature maps for detection heads {"P3": [...], "P4": [...], "P5": [...]}
    """

# Head interface
def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Args:
        features: Output from neck {"P3": [...], "P4": [...], "P5": [...]}
    Returns:
        {"boxes": ..., "scores": ..., "classes": ...}
    """
```

### Component Principles

1. **Single Responsibility**: Each component does one thing well
2. **Explicit Dependencies**: No hidden state or global configs
3. **Type Hints**: Every parameter and return value is typed
4. **Documentation**: Clear docstrings with examples
5. **Testability**: Components work in isolation

### Loss Function Design

```python
# Flexible loss composition - build your own
class CustomYOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = IoULoss(loss_type='ciou')
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0) 
        self.obj_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Your custom loss logic here
        return bbox_loss + cls_loss + obj_loss
```

## 📊 Supported Models & Features

### Current Support (v0.1.0)
- **YOLO Family**: YOLOv3, YOLOv5, YOLOv8, YOLOv11
- **Components**: 50+ reusable building blocks (C3K2, SPPF, C2PSA, FPN, etc.)
- **Losses**: YOLO loss, Focal loss, IoU variants (GIoU, DIoU, CIoU)
- **Data**: COCO format datasets with built-in augmentations
- **Training**: Full training pipeline with AMP support, gradient clipping

### Architecture Roadmap

| Version | Models | Key Features |
|---------|--------|-------------|
| **v0.1.0** | YOLO (v3,v5,v8,v11) | Core infrastructure, component system |
| **v0.2.0** | + DETR family | Transformer-based detection, Hungarian matching |
| **v0.3.0** | + RetinaNet, FCOS | Single-stage detectors, anchor-free methods |
| **v0.4.0** | + R-CNN family | Two-stage detectors, instance segmentation |
| **v0.5.0** | + EfficientDet, RT-DETR | Advanced architectures, real-time transformers |

## 🛠️ Installation & Setup

```bash
# Install from source (replace with actual repo URL)
git clone https://github.com/your-username/TorchDetection.git
cd TorchDetection
pip install -e .

# Core dependencies
pip install torch>=1.12.0 torchvision>=0.13.0
pip install numpy opencv-python pillow pyyaml tqdm

# Optional dependencies  
pip install wandb tensorboard  # Experiment tracking
pip install matplotlib seaborn  # Visualization
```

### Quick Setup for COCO Training

```bash
# Download COCO dataset
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip *.zip

# Verify installation
cd ../../
python -c "import torchdetection; print('TorchDetection installed successfully!')"
```

## 🧪 Testing & Validation

```python
# Test individual components
from torchdetection.nn.components import C3K2, SPPF
import torch

# Test C3K2 component
x = torch.randn(1, 256, 32, 32)
c3k2 = C3K2(in_channels=256, out_channels=256, num_blocks=2)
out = c3k2(x)
print(f"C3K2 output shape: {out.shape}")

# Test complete model
from torchdetection.nn.models.yolo import YOLOv8
model = YOLOv8.from_preset('yolov8s', num_classes=80)
x = torch.randn(1, 3, 640, 640)
predictions = model(x)
print(f"Predictions: {predictions.keys()}")
```

### Model Presets

```python
# Easy model creation with validated presets
model = YOLOv8.from_preset('yolov8n', num_classes=80)    # nano
model = YOLOv8.from_preset('yolov8s', num_classes=80)    # small  
model = YOLOv8.from_preset('yolov8m', num_classes=80)    # medium
model = YOLOv8.from_preset('yolov8l', num_classes=80)    # large
model = YOLOv8.from_preset('yolov8x', num_classes=80)    # xlarge

# Or YOLOv11 with latest features
model = YOLOv11.from_preset('yolov11s', num_classes=80)
```

## Why This Design?

### What You Get

- **🔍 Debuggability**: Every component is inspectable and testable
- **🔧 Flexibility**: Mix and match any components  
- **📖 Readability**: Code clearly shows what's happening
- **🚀 Performance**: Pure PyTorch, no framework overhead
- **🧪 Testability**: Each component works in isolation
- **📚 Learning**: Understand exactly how models work

### What You Don't Get

- ❌ Config file magic
- ❌ Hidden registries or auto-discovery  
- ❌ "Everything works out of the box" convenience
- ❌ Framework-specific abstractions
- ❌ Automatic hyperparameter tuning

**Trade-off**: A bit more explicit coding for much better understanding and control.

## 🔄 Inference & Export

```python
# Inference with post-processing
import torch
from torchdetection.nn.models.yolo import YOLOv8
from torchdetection.utils import non_max_suppression, letterbox

model = YOLOv8.from_preset('yolov8s', num_classes=80)
model.eval()

# Load and preprocess image
image = letterbox(image, size=640)  # Resize with padding
x = torch.from_numpy(image).unsqueeze(0).float() / 255.0

# Inference
with torch.no_grad():
    predictions = model(x)
    
# Post-processing
detections = non_max_suppression(
    predictions, 
    conf_threshold=0.25, 
    iou_threshold=0.45
)
```

### Model Export

```python
# Export to ONNX
torch.onnx.export(
    model, 
    torch.randn(1, 3, 640, 640),
    "yolov8s.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Export to TorchScript
scripted = torch.jit.script(model)
scripted.save("yolov8s.pt")
```

## 🤝 Contributing

We follow strict architectural principles:

1. **Explicit Over Implicit**: If it's important, make it visible
2. **PyTorch Patterns**: Use standard PyTorch conventions
3. **Component Isolation**: Each piece should work independently  
4. **Interface Consistency**: Follow established contracts
5. **Documentation**: Include examples and type hints
6. **Type Safety**: Full type hints for all public APIs

### Adding New Components

```python
# Follow this pattern for new components
class NewComponent(nn.Module):
    """Brief description.
    
    Args:
        param1: Description with type info
        param2: Description with type info
        
    Example:
        >>> comp = NewComponent(param1=value1)
        >>> output = comp(input_tensor)
    """
    def __init__(self, param1: int, param2: str):
        super().__init__()
        # Explicit initialization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clear input/output contract
        pass
```

## 🔗 Resources & References

- **Research Papers**: YOLOv8, YOLOv11, DETR, RetinaNet implementation details
- **Documentation**: Full API docs and component guides  
- **Examples**: Complete training scripts and model configurations
- **Benchmarks**: Performance comparisons and optimization guides

**TorchDetection**: Where explicitness meets performance. Build detection models the PyTorch way. 🚀
