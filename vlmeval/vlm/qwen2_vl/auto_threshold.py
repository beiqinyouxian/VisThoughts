
import cv2
import numpy as np

# 固定阈值法
def constant_threshold(image, threshold=0.5):
    # 直接使用输入的2D数组
    # 固定阈值法
    _, binary = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return binary

# Otsu 自动阈值法，效果尚可，但是对小目标的分割效果较差
def auto_otsu(image):
    # 转换为8位整数
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Otsu自动阈值法
    _, binary_uint8 = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 转回0-1浮点数
    binary = binary_uint8 / 255.0
    return binary

# 自适应阈值法，效果非常差，不建议使用
def adaptive_threshold(image):
    # 转换为8位整数
    image_uint8 = (image * 255).astype(np.uint8)
    
    # 自适应阈值法
    binary_uint8 = cv2.adaptiveThreshold(image_uint8, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    
    # 转回0-1浮点数
    binary = binary_uint8 / 255.0
    return binary

# 局部Otsu阈值法，不可用
def local_otsu(image, block_size=64):
    h, w = image.shape
    result = np.zeros_like(image)
    
    # 转换为8位整数
    image_uint8 = (image * 255).astype(np.uint8)
    result_uint8 = np.zeros_like(image_uint8)
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # 计算不越界的块大小
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            
            block = image_uint8[y:y_end, x:x_end]
            if block.size == 0:
                continue
                
            # 对每个块应用Otsu阈值
            _, block_binary = cv2.threshold(
                block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            result_uint8[y:y_end, x:x_end] = block_binary
    
    # 转回0-1浮点数
    result = result_uint8 / 255.0
    return result