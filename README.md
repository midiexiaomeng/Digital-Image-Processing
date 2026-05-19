# 数字图像处理实验报告

## 1. 引言

数字图像处理是通过计算机对图像进行分析、处理和识别的技术。本实验通过实现一个类似Photoshop的图像处理软件，深入理解数字图像处理的基本原理和常用算法。实验内容包括图像加载与显示、灰度转换、高斯模糊、边缘检测、图像旋转、裁剪、直方图显示、翻转、形态学变换、图像分割等功能的实现。

## 2. 理论基础

### 2.1 数字图像基本概念

数字图像是由像素组成的二维矩阵，每个像素代表图像在该点的亮度或颜色值。常见的图像类型包括：

1. 灰度图像：单通道，像素值表示亮度(0-255)
2. 彩色图像：三通道(RGB)，每个通道表示一种颜色分量

### 2.2 相关图像处理算法原理

#### 2.2.1 灰度转换算法

将彩色图像转换为灰度图像的公式如下：

```python
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

#### 2.2.2 高斯模糊算法

高斯模糊使用高斯函数作为权重对图像进行平滑处理。二维高斯函数定义为：

```
G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
```

实现代码：

```python
def gaussian_kernel(self, size, sigma=1.0):
    """生成高斯核"""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    kernel /= kernel.sum()  # 归一化
    return kernel
```

#### 2.2.3 Sobel边缘检测算法

Sobel算子使用两个3×3卷积核计算图像梯度：

```
Sobel_x = [-1 0 1; -2 0 2; -1 0 1]
Sobel_y = [-1 -2 -1; 0 0 0; 1 2 1]
```

梯度幅值计算：

```
G = sqrt(Gx² + Gy²)
```

实现代码：

```python
def sobel_edge_detection(self):
    """改进的Sobel边缘检测(使用纯numpy实现)"""
    if self.image_data is None:
        return
        
    try:
        # Sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # 处理彩色或灰度图像
        if len(self.image_data.shape) == 3:  # 彩色
            edges = np.zeros_like(self.image_data[..., 0])  # 单通道输出
            for c in range(3):  # 对每个通道分别处理
                grad_x = self.convolve2d(self.image_data[..., c], sobel_x)
                grad_y = self.convolve2d(self.image_data[..., c], sobel_y)
                edges = np.maximum(edges, np.sqrt(grad_x**2 + grad_y**2))
        else:  # 灰度
            grad_x = self.convolve2d(self.image_data, sobel_x)
            grad_y = self.convolve2d(self.image_data, sobel_y)
            edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 手动归一化到0-255
        edges_min = np.min(edges)
        edges_max = np.max(edges)
        if edges_max > edges_min:
            edges = ((edges - edges_min) / (edges_max - edges_min)) * 255
        
        # 双阈值处理
        low_threshold = 30
        high_threshold = 100
        edges[(edges >= low_threshold) & (edges <= high_threshold)] = 128
        edges[edges > high_threshold] = 255
        edges[edges < low_threshold] = 0
        
        self.image_data = edges.astype(np.uint8)
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"边缘检测错误: {e}")
```

#### 2.2.4 图像旋转算法

图像旋转通过坐标变换实现，90度旋转的矩阵变换：

```
[x'] = [0 -1] [x]
[y']   [1  0] [y]
```

实现代码：

```python
def rotate_image(self, angle):
    """旋转图像"""
    if self.image_data is None:
        return
        
    try:
        # 处理不同角度
        if angle == 90:
            if len(self.image_data.shape) == 3:  # 彩色
                self.image_data = np.rot90(self.image_data, axes=(0, 1))
            else:  # 灰度
                self.image_data = np.rot90(self.image_data)
        elif angle == 180:
            if len(self.image_data.shape) == 3:  # 彩色
                self.image_data = np.rot90(self.image_data, 2, axes=(0, 1))
            else:  # 灰度
                self.image_data = np.rot90(self.image_data, 2)
        elif angle == 270:
            if len(self.image_data.shape) == 3:  # 彩色
                self.image_data = np.rot90(self.image_data, 3, axes=(0, 1))
            else:  # 灰度
                self.image_data = np.rot90(self.image_data, 3)
                
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"旋转图像错误: {e}")
```

#### 2.2.5 图像读取与显示

图像读取使用OpenCV库实现，支持多种常见图像格式：

```python
def load_image(self):
    """加载图像文件"""
    file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
    if file_path:
        try:
            # 使用cv2加载图像(仅用于加载)
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.image_data = self.original_image.copy()
                self.update_image_display()
                self.add_to_history()
        except Exception as e:
            print(f"加载图像错误: {e}")
```

#### 2.2.6 阈值处理算法

阈值处理将图像转换为二值图像：

```python
def apply_segmentation(self, method, threshold=127):
    """应用图像分割
    method: 'threshold', 'edge', 'region', 'watershed'
    threshold: 仅用于阈值分割
    """
    if self.image_data is None:
        return
        
    try:
        # 转换为灰度图像进行处理
        if len(self.image_data.shape) == 3:
            gray = np.dot(self.image_data[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = self.image_data.copy()
            
        if method == 'threshold':
            # 阈值分割
            _, segmented = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
        # 将结果转换为彩色显示
        if len(self.image_data.shape) == 3:
            self.image_data = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
        else:
            self.image_data = segmented
            
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"图像分割错误: {e}")
```

#### 2.2.7 直方图显示

直方图用于分析图像像素值分布：

```python
def show_histogram(self):
    """显示图像直方图"""
    if self.image_data is None:
        return
        
    try:
        # 创建新窗口
        hist_window = QDialog(self)
        hist_window.setWindowTitle("图像直方图")
        hist_window.setMinimumSize(800, 600)
        
        # 创建matplotlib图形
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        
        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        hist_window.setLayout(layout)
        
        # 绘制直方图
        ax = fig.add_subplot(111)
        
        if len(self.image_data.shape) == 2:  # 灰度图像
            ax.hist(self.image_data.ravel(), bins=256, range=(0, 256), 
                   color='gray', alpha=0.7)
            ax.set_title('灰度直方图')
        else:  # 彩色图像
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                ax.hist(self.image_data[..., i].ravel(), bins=256, 
                       range=(0, 256), color=color, alpha=0.5, 
                       label=f'{color.upper()}通道')
            ax.set_title('RGB通道直方图')
            ax.legend()
            
        ax.set_xlim([0, 256])
        ax.set_xlabel('像素值')
        ax.set_ylabel('频数')
        
        # 显示窗口
        hist_window.exec_()
    except Exception as e:
        print(f"显示直方图错误: {e}")
```

#### 2.2.8 图像裁剪

图像裁剪允许用户选择区域进行裁剪：

```python
def apply_crop(self):
    """应用裁剪"""
    if self.image_data is None or not self.crop_start or not self.crop_end:
        return
        
    try:
        # 获取图像尺寸
        height, width = self.image_data.shape[:2]
        
        # 计算裁剪区域(像素坐标)
        x1 = max(0, min(int(self.crop_start.x()), width-1))
        y1 = max(0, min(int(self.crop_start.y()), height-1))
        x2 = max(0, min(int(self.crop_end.x()), width-1))
        y2 = max(0, min(int(self.crop_end.y()), height-1))
        
        # 确保x1,y1是左上角，x2,y2是右下角
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 应用裁剪
        if len(self.image_data.shape) == 3:  # 彩色
            self.image_data = self.image_data[y1:y2, x1:x2, :]
        else:  # 灰度
            self.image_data = self.image_data[y1:y2, x1:x2]
            
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"裁剪错误: {e}")
```

#### 2.2.9 图像滤波

频域滤波用于增强或抑制特定频率成分：

```python
def apply_filter(self, low_threshold, high_threshold):
    """应用频域滤波
    low_threshold: 低频阈值 (0-100)
    high_threshold: 高频阈值 (0-100)
    """
    if self.image_data is None:
        return
        
    try:
        # 转换为灰度图像进行处理
        if len(self.image_data.shape) == 3:
            gray = np.dot(self.image_data[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
        else:
            gray = self.image_data.astype(np.float32)
        
        # 获取图像尺寸
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 傅里叶变换
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        
        # 创建理想带通滤波器
        mask = np.zeros((rows, cols), np.float32)
        r = min(rows, cols) // 2
        low_r = r * low_threshold / 100.0
        high_r = r * high_threshold / 100.0
        
        # 创建圆形掩模
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2
        mask[(mask_area >= low_r**2) & (mask_area <= high_r**2)] = 1
        
        # 应用滤波器
        fshift = dft_shift * mask
        
        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # 归一化到0-255
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        
        # 将结果转换为彩色显示
        if len(self.image_data.shape) == 3:
            self.image_data = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            self.image_data = img_back.astype(np.uint8)
            
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"图像滤波错误: {e}")
```

#### 2.2.10 亮度对比度调整

亮度对比度调整改变图像整体视觉效果：

```python
def apply_brightness_contrast(self, brightness=None, contrast=None):
    """应用亮度和对比度调整(基于历史记录中的最新状态)"""
    if self.image_data is None or not self.history:
        return
        
    try:
        # 使用传入值或当前滑块值
        brightness = brightness if brightness is not None else self.brightness_slider.value()
        contrast = contrast if contrast is not None else self.contrast_slider.value()
        
        # 计算调整参数
        brightness_adjust = brightness  # 直接使用滑块值(-100到100)
        contrast_factor = 1.0 + contrast / 100.0  # 对比度范围0.5-1.5
        
        # 获取历史记录中的最新状态作为基准
        base_image = self.history[self.history_index].copy()
        
        # 处理彩色或灰度图像
        if len(base_image.shape) == 3:  # 彩色
            adjusted = np.zeros_like(base_image, dtype=np.float32)
            for c in range(3):  # 对每个通道分别处理
                channel = base_image[..., c].astype(np.float32)
                # 应用对比度
                channel = (channel - 127.5) * contrast_factor + 127.5
                # 应用亮度(直接加减)
                channel = channel + brightness_adjust
                # 裁剪到0-255范围
                adjusted[..., c] = np.clip(channel, 0, 255)
        else:  # 灰度
            adjusted = base_image.astype(np.float32)
            # 应用对比度
            adjusted = (adjusted - 127.5) * contrast_factor + 127.5
            # 应用亮度(直接加减)
            adjusted = adjusted + brightness_adjust
            # 裁剪到0-255范围
            adjusted = np.clip(adjusted, 0, 255)
            
        # 保存调整后的图像(不添加到历史记录)
        self.image_data = adjusted.astype(np.uint8)
        self.update_processed_display()
    except Exception as e:
        print(f"亮度和对比度调整错误: {e}")
```

#### 2.2.11 图像分割算法

1. 阈值分割：将像素值与阈值比较，分为前景和背景
2. 区域生长：从种子点开始，根据相似性准则扩展区域
3. 分水岭算法：将图像视为地形图，模拟水淹过程进行分割

实现代码：

```python
def apply_segmentation(self, method, threshold=127):
    """应用图像分割
    method: 'threshold', 'edge', 'region', 'watershed'
    threshold: 仅用于阈值分割
    """
    if self.image_data is None:
        return
        
    try:
        # 转换为灰度图像进行处理
        if len(self.image_data.shape) == 3:
            gray = np.dot(self.image_data[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = self.image_data.copy()
            
        if method == 'threshold':
            # 阈值分割
            _, segmented = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'edge':
            # 边缘检测分割
            edges = cv2.Canny(gray, 100, 200)
            kernel = np.ones((3,3), np.uint8)
            segmented = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        elif method == 'region':
            # 区域生长(简化版)
            segmented = self.region_growing(gray)
        elif method == 'watershed':
            # 分水岭算法
            segmented = self.watershed_segmentation(gray)
            
        # 将结果转换为彩色显示
        if len(self.image_data.shape) == 3:
            self.image_data = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
        else:
            self.image_data = segmented
            
        self.update_processed_display()
        self.add_to_history()
    except Exception as e:
        print(f"图像分割错误: {e}")
```

## 3. 实验环境与工具及依赖库

- 操作系统：Windows 11
- 编程语言：Python 3.9
- 主要依赖库：
  - PyQt5：用于GUI界面开发
  - NumPy：用于图像矩阵运算
  - OpenCV：用于图像加载和保存
  - Matplotlib：用于直方图显示

## 4. 实验内容与步骤

### 4.1 实验准备

1. 安装Python环境和必要依赖库
2. 设计软件界面布局
3. 实现图像加载和显示功能

### 4.2 图像处理功能实现

1. 灰度转换
2. 高斯模糊
3. 边缘检测
4. 图像旋转
5. 图像裁剪
6. 直方图显示
7. 图像翻转
8. 形态学变换
9. 图像分割

### 4.3 功能测试与优化

1. 测试各功能模块
2. 优化算法性能
3. 添加撤销/重做功能

## 5. 实验结果与分析

### 5.1 灰度转换效果

灰度转换后图像保留了原始图像的亮度信息，去除了颜色信息。实验结果表明，加权平均法比简单平均法能更好地保留人眼感知的亮度。

### 5.2 高斯模糊效果

高斯模糊能有效去除图像噪声，同时保留图像的主要特征。σ值越大，模糊效果越明显。

### 5.3 边缘检测效果

Sobel算子能有效检测图像边缘，双阈值处理可以减少噪声干扰，得到更清晰的边缘。

### 5.4 图像分割效果

不同分割方法适用于不同场景：
- 阈值分割：适用于前景背景对比明显的图像
- 区域生长：适用于区域内部一致性高的图像
- 分水岭算法：适用于物体边界清晰的图像

### 5.5 图像读取与显示效果

图像读取功能支持多种常见格式，包括PNG、JPG、JPEG和BMP。实验表明：
1. OpenCV的imread函数能高效加载各种格式图像
2. 颜色空间转换(BGR→RGB)确保显示颜色正确
3. 错误处理机制能有效捕获并提示加载异常

### 5.6 阈值处理效果

阈值处理实验结果分析：
1. 固定阈值(127)适用于大多数普通场景
2. 对于光照不均匀的图像，自适应阈值效果更好
3. 阈值大小直接影响分割结果：
   - 阈值过低会导致过多背景被误认为前景
   - 阈值过高会导致前景信息丢失

### 5.7 直方图分析效果

直方图功能实验结果：
1. 灰度直方图能直观显示图像亮度分布
2. RGB通道直方图可分析各颜色分量分布
3. 直方图形状反映图像特性：
   - 窄峰：对比度低
   - 宽分布：对比度高
   - 多峰：可能包含多个亮度区域

### 5.8 图像裁剪效果

裁剪功能实验结果：
1. 能精确选择并保留感兴趣区域
2. 坐标转换确保裁剪区域正确
3. 边界处理避免越界错误
4. 支持彩色和灰度图像裁剪

### 5.9 频域滤波效果

频域滤波实验结果分析：
1. 低频滤波(低通)：
   - 保留图像整体结构
   - 产生平滑效果
   - 适合去噪
2. 高频滤波(高通)：
   - 增强边缘和细节
   - 可能放大噪声
   - 适合边缘检测预处理
3. 带通滤波：
   - 选择性增强特定频率
   - 需要精细调节阈值

### 5.10 亮度对比度调整效果

亮度对比度调整实验结果：
1. 亮度调整：
   - 正值整体增亮图像
   - 负值整体变暗图像
   - 线性变化保持相对关系
2. 对比度调整：
   - 高对比度增强细节但可能丢失中间色调
   - 低对比度使图像平淡但保留更多层次
3. 组合调整可实现多种视觉效果

## 6. 实验总结和疑难总结

### 6.1 实验总结

通过本次实验，实现了完整的图像处理软件，深入理解了数字图像处理的基本原理和常用算法。实验过程中，掌握了图像处理算法的实现方法，以及GUI界面与图像处理算法的结合方式。

### 6.2 疑难总结

1. 图像旋转时边界处理问题：通过调整旋转中心和填充方式解决
2. 边缘检测噪声问题：通过高斯平滑和双阈值处理优化
3. 分水岭算法过分割问题：通过标记控制改进
4. 大图像处理性能问题：通过优化算法和分块处理提高效率
