import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 简单的GPU测试
    x = torch.rand(5, 3).cuda()
    print(f"张量在GPU上: {x.is_cuda}")
