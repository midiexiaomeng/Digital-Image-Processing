import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, 
                            QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QSlider, QComboBox, QGraphicsView, QGraphicsScene,
                            QSplitter, QToolBar, QStatusBar, QGroupBox,
                            QDialog, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPen, QColor, QPainter, QBrush
from PyQt5.QtCore import Qt, QSize, QRectF, QPointF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import struct
import zlib
import cv2  # 仅用于图像分割

class ImageEditorPS(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理软件 (PS风格)")
        self.setGeometry(100, 100, 1400, 800)
        
        # 初始化UI
        self.initUI()
        
        # 图像数据
        self.image_data = None    # 当前图像数据 (numpy数组)
        self.original_image = None  # 原始图像数据
        self.history = []        # 历史记录
        self.history_index = -1  # 当前历史记录索引
        
        # 裁剪相关状态
        self.crop_rect = None    # 裁剪区域
        self.crop_start = None   # 裁剪起始点
        self.crop_end = None     # 裁剪结束点
        self.is_cropping = False # 是否正在裁剪
        
    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 图像显示区域
        image_area = QWidget()
        image_layout = QHBoxLayout()
        image_area.setLayout(image_layout)
        
        # 原图视图
        self.original_view = QGraphicsView()
        self.original_scene = QGraphicsScene()
        self.original_view.setScene(self.original_scene)
        
        # 处理后视图
        self.processed_view = QGraphicsView()
        self.processed_scene = QGraphicsScene()
        self.processed_view.setScene(self.processed_scene)
        
        # 使用QSplitter创建可调整的视图
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.original_view)
        splitter.addWidget(self.processed_view)
        splitter.setSizes([600, 600])
        
        image_layout.addWidget(splitter)
        main_layout.addWidget(image_area, 70)
        
        # 控制面板
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 30)
        
        # 文件操作按钮组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("保存图像")
        self.save_btn.clicked.connect(self.save_image)
        file_layout.addWidget(self.save_btn)
        
        file_group.setLayout(file_layout)
        control_panel.addWidget(file_group)
        
        # 图像处理功能按钮组
        process_group = QGroupBox("图像处理")
        process_layout = QVBoxLayout()
        
        # 灰度转换
        self.grayscale_btn = QPushButton("灰度转换")
        self.grayscale_btn.clicked.connect(self.convert_to_grayscale)
        process_layout.addWidget(self.grayscale_btn)
        
        # 高斯模糊
        self.gaussian_btn = QPushButton("高斯模糊")
        self.gaussian_btn.clicked.connect(self.apply_gaussian_blur)
        process_layout.addWidget(self.gaussian_btn)
        
        # 边缘检测
        self.edge_btn = QPushButton("边缘检测")
        self.edge_btn.clicked.connect(self.detect_edges)
        process_layout.addWidget(self.edge_btn)
        
        # 旋转图像
        self.rotate_btn = QPushButton("旋转图像")
        self.rotate_btn.clicked.connect(self.show_rotate_dialog)
        process_layout.addWidget(self.rotate_btn)
        
        # 裁剪图像
        self.crop_btn = QPushButton("裁剪图像")
        self.crop_btn.clicked.connect(self.start_crop)
        process_layout.addWidget(self.crop_btn)
        
        # 直方图
        self.histogram_btn = QPushButton("直方图")
        self.histogram_btn.clicked.connect(self.show_histogram)
        process_layout.addWidget(self.histogram_btn)
        
        # 阈值处理
        self.threshold_btn = QPushButton("阈值处理")
        self.threshold_btn.clicked.connect(self.show_threshold_dialog)
        process_layout.addWidget(self.threshold_btn)
        
        # 形态学变换
        self.morphology_btn = QPushButton("形态学变换")
        self.morphology_btn.clicked.connect(self.show_morphology_dialog)
        process_layout.addWidget(self.morphology_btn)
        
        # 翻转图像
        self.flip_btn = QPushButton("翻转图像")
        self.flip_btn.clicked.connect(self.show_flip_dialog)
        process_layout.addWidget(self.flip_btn)
        
        # 图像分割
        self.segmentation_btn = QPushButton("图像分割")
        self.segmentation_btn.clicked.connect(self.show_segmentation_dialog)
        process_layout.addWidget(self.segmentation_btn)
        
        # 图像滤波
        self.filter_btn = QPushButton("图像滤波")
        self.filter_btn.clicked.connect(self.show_filter_dialog)
        process_layout.addWidget(self.filter_btn)
        
        process_group.setLayout(process_layout)
        control_panel.addWidget(process_group)
        
        # 撤销重做按钮
        undo_redo_group = QGroupBox("撤销/重做")
        undo_redo_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.clicked.connect(self.undo_action)
        undo_redo_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("重做")
        self.redo_btn.clicked.connect(self.redo_action)
        undo_redo_layout.addWidget(self.redo_btn)
        
        undo_redo_group.setLayout(undo_redo_layout)
        control_panel.addWidget(undo_redo_group)
        
        # 亮度和对比度调整
        bc_group = QGroupBox("亮度和对比度")
        bc_layout = QVBoxLayout()
        
        # 亮度滑块
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("亮度: 0")
        bc_layout.addWidget(self.brightness_label)
        bc_layout.addWidget(self.brightness_slider)
        
        # 对比度滑块
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_label = QLabel("对比度: 0")
        bc_layout.addWidget(self.contrast_label)
        bc_layout.addWidget(self.contrast_slider)
        
        # 应用按钮
        self.apply_bc_btn = QPushButton("应用")
        self.apply_bc_btn.clicked.connect(self.adjust_brightness_contrast)
        bc_layout.addWidget(self.apply_bc_btn)
        
        bc_group.setLayout(bc_layout)
        control_panel.addWidget(bc_group)
        
        # 添加拉伸使布局更美观
        control_panel.addStretch()
        
    # [图像处理功能实现...]
    def save_image(self):
        """保存图像文件(仅使用numpy)"""
        if self.image_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", 
                        "PNG图像 (*.png);;BMP图像 (*.bmp)")
        if file_path:
            try:
                if file_path.lower().endswith('.bmp'):
                    # BMP保存实现
                    height, width = self.image_data.shape[:2]
                    if len(self.image_data.shape) == 3:  # 彩色图像
                        # 转换为BGR格式
                        bgr_data = self.image_data[:,:,::-1].copy()
                        bytes_per_pixel = 3
                    else:  # 灰度图像
                        bgr_data = np.stack([self.image_data]*3, axis=2)
                        bytes_per_pixel = 3
                    
                    # 计算行大小(每行必须是4字节对齐)
                    row_size = (width * bytes_per_pixel + 3) & ~3
                    padding = row_size - width * bytes_per_pixel
                    
                    # 创建文件头
                    file_size = 54 + row_size * height
                    header = bytearray([
                        ord('B'), ord('M'),  # 签名
                        *file_size.to_bytes(4, 'little'),  # 文件大小
                        0, 0, 0, 0,  # 保留
                        54, 0, 0, 0,  # 像素数据偏移
                        40, 0, 0, 0,  # DIB头大小
                        *width.to_bytes(4, 'little'),  # 宽度
                        *height.to_bytes(4, 'little'),  # 高度
                        1, 0,  # 颜色平面数
                        24, 0,  # 每像素位数(24位)
                        0, 0, 0, 0,  # 压缩方法
                        *(row_size * height).to_bytes(4, 'little'),  # 图像大小
                        0, 0, 0, 0,  # 水平分辨率
                        0, 0, 0, 0,  # 垂直分辨率
                        0, 0, 0, 0,  # 调色板颜色数
                        0, 0, 0, 0   # 重要颜色数
                    ])
                    
                    # 写入文件
                    with open(file_path, 'wb') as f:
                        f.write(header)
                        # 从下到上写入行数据
                        for row in reversed(range(height)):
                            f.write(bgr_data[row].tobytes())
                            if padding:
                                f.write(bytes([0] * padding))
                                
                elif file_path.lower().endswith('.png'):
                    # 简化PNG保存实现
                    height, width = self.image_data.shape[:2]
                    if len(self.image_data.shape) == 3:  # 彩色图像
                        channels = 3
                    else:  # 灰度图像
                        channels = 1
                    
                    # PNG签名
                    signature = bytes([137, 80, 78, 71, 13, 10, 26, 10])
                    
                    # IHDR块
                    ihdr_data = bytearray()
                    ihdr_data.extend(width.to_bytes(4, 'big'))  # 宽度
                    ihdr_data.extend(height.to_bytes(4, 'big'))  # 高度
                    ihdr_data.append(8)  # 位深度
                    ihdr_data.append(2 if channels == 3 else 0)  # 颜色类型(RGB/灰度)
                    ihdr_data.extend([0, 0, 0])  # 压缩/过滤/交错方法
                    
                    ihdr_chunk = bytearray(b'IHDR')
                    ihdr_chunk.extend(ihdr_data)
                    ihdr_crc = zlib.crc32(ihdr_chunk[4:]).to_bytes(4, 'big')
                    ihdr_chunk = (len(ihdr_data).to_bytes(4, 'big') + ihdr_chunk + ihdr_crc)
                    
                    # IDAT块(简化版，不压缩)
                    pixel_data = bytearray()
                    for row in range(height):
                        pixel_data.append(0)  # 过滤器类型(无过滤)
                        if channels == 3:
                            pixel_data.extend(self.image_data[row].tobytes())
                        else:
                            pixel_data.extend(self.image_data[row].tobytes())
                    
                    idat_chunk = bytearray(b'IDAT')
                    idat_chunk.extend(zlib.compress(pixel_data))
                    idat_crc = zlib.crc32(idat_chunk[4:]).to_bytes(4, 'big')
                    idat_chunk = (len(idat_chunk)-4).to_bytes(4, 'big') + idat_chunk + idat_crc
                    
                    # IEND块
                    iend_chunk = bytes([0,0,0,0, 73,69,78,68, 174,66,96,130])
                    
                    # 写入文件
                    with open(file_path, 'wb') as f:
                        f.write(signature)
                        f.write(ihdr_chunk)
                        f.write(idat_chunk)
                        f.write(iend_chunk)
                
            except Exception as e:
                print(f"Error saving image: {e}")

    def load_image(self):
        """加载图像文件(仅使用numpy)"""
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                # 读取文件二进制数据
                with open(file_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint8)
                
                # 简单文件类型检测
                if file_path.lower().endswith('.bmp'):
                    # BMP文件处理
                    offset = int.from_bytes(data[10:14], byteorder='little')
                    width = int.from_bytes(data[18:22], byteorder='little')
                    height = int.from_bytes(data[22:26], byteorder='little')
                    bpp = int.from_bytes(data[28:30], byteorder='little')
                    
                    # 仅支持24位和32位BMP
                    if bpp not in [24, 32]:
                        raise ValueError("仅支持24/32位BMP")
                    
                    # 读取像素数据
                    pixel_data = data[offset:]
                    if bpp == 24:
                        self.original_image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                            (height, width, 3))[:,:,::-1]  # BGR转RGB
                    else:  # 32位
                        self.original_image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                            (height, width, 4))[:,:,:3]  # 忽略alpha通道
                
                elif file_path.lower().endswith('.png'):
                    # 简化PNG解码 - 仅处理8位RGB PNG
                    # 检查PNG签名
                    if not np.array_equal(data[:8], np.array([137,80,78,71,13,10,26,10], dtype=np.uint8)):
                        raise ValueError("无效的PNG文件")
                    
                    # 查找IHDR块
                    ihdr_pos = 8
                    while ihdr_pos < len(data)-12:
                        if np.array_equal(data[ihdr_pos+4:ihdr_pos+8], np.array([73,72,68,82], dtype=np.uint8)):
                            break
                        ihdr_pos += 1
                    
                    # 获取图像尺寸和格式
                    width = int.from_bytes(data[ihdr_pos+8:ihdr_pos+12], byteorder='big')
                    height = int.from_bytes(data[ihdr_pos+12:ihdr_pos+16], byteorder='big')
                    color_type = data[ihdr_pos+17]
                    
                    # 仅支持RGB PNG
                    if color_type != 2:
                        raise ValueError("仅支持RGB PNG")
                    
                    # 查找IDAT块
                    idat_pos = ihdr_pos + 8 + int.from_bytes(data[ihdr_pos:ihdr_pos+4], byteorder='big') + 4
                    while idat_pos < len(data)-12:
                        if np.array_equal(data[idat_pos+4:idat_pos+8], np.array([73,68,65,84], dtype=np.uint8)):
                            break
                        idat_pos += 1
                    
                    # 解压IDAT数据
                    idat_size = int.from_bytes(data[idat_pos:idat_pos+4], byteorder='big')
                    compressed_data = bytes(data[idat_pos+8:idat_pos+8+idat_size])
                    raw_data = zlib.decompress(compressed_data)
                    
                    # 处理扫描线过滤器(简化版，只处理无过滤和Sub过滤)
                    stride = width * 3
                    pixel_data = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    for i in range(height):
                        filter_type = raw_data[i * (stride + 1)]
                        scanline = raw_data[i * (stride + 1) + 1 : (i + 1) * (stride + 1)]
                        
                        if filter_type == 0:  # None
                            pixel_data[i] = np.frombuffer(scanline, dtype=np.uint8).reshape(width, 3)
                        elif filter_type == 1:  # Sub
                            scanline = np.frombuffer(scanline, dtype=np.uint8)
                            row = np.zeros(stride, dtype=np.uint8)
                            for j in range(stride):
                                if j < 3:
                                    row[j] = scanline[j]
                                else:
                                    row[j] = (scanline[j] + row[j-3]) % 256
                            pixel_data[i] = row.reshape(width, 3)
                    
                    self.original_image = pixel_data
                
                else:
                    raise ValueError("不支持的图像格式")
                
                self.image_data = self.original_image.copy()
                self.history = [self.image_data.copy()]
                self.history_index = 0
                self.update_image_views()
                
            except Exception as e:
                print(f"Error loading image: {e}")
                try:
                    # 尝试使用cv2作为后备方案
                    self.original_image = cv2.imread(file_path)
                    if self.original_image is not None:
                        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                        self.image_data = self.original_image.copy()
                        self.history = [self.image_data.copy()]
                        self.history_index = 0
                        self.update_image_views()
                        return
                except:
                    pass
                
                # 如果所有方法都失败，创建空白图像
                self.original_image = np.zeros((100, 100, 3), dtype=np.uint8)
                self.image_data = self.original_image.copy()
                self.history = [self.image_data.copy()]
                self.history_index = 0
                self.update_image_views()
    
    def convert_to_grayscale(self):
        """将图像转换为灰度"""
        if self.image_data is None:
            return
            
        if len(self.image_data.shape) == 2:  # 已经是灰度图
            return
            
        # 创建副本进行处理，保留原图
        img_copy = self.image_data.copy()
        
        # 使用加权平均法转换灰度
        r, g, b = img_copy[:,:,0], img_copy[:,:,1], img_copy[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        self.image_data = gray.astype(np.uint8)
        self.add_to_history()
        self.update_image_views()
    
    def apply_gaussian_blur(self, size=5, sigma=1.0):
        """应用高斯模糊"""
        if self.image_data is None:
            return
            
        # 验证参数
        size = max(3, min(15, int(size)))  # 限制核大小在3-15之间
        sigma = max(0.5, min(5.0, float(sigma)))  # 限制sigma在0.5-5.0之间
        
        # 创建高斯核
        kernel = self.create_gaussian_kernel(size, sigma)
        
        # 应用卷积
        if len(self.image_data.shape) == 3:  # 彩色图像
            blurred = np.zeros_like(self.image_data, dtype=np.float32)
            for i in range(3):
                blurred[:,:,i] = self._convolve2d(self.image_data[:,:,i].astype(np.float32), kernel)
            blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        else:  # 灰度图像
            blurred = self.convolve2d(self.image_data, kernel)
            
        self.image_data = blurred
        self.add_to_history()
        self.update_image_views()
    
    def create_gaussian_kernel(self, size, sigma):
        """创建高斯核"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i,j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
        
        # 确保核值非负且总和为1
        kernel = np.abs(kernel)
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def convolve2d(self, image, kernel):
        """2D卷积实现"""
        if len(image.shape) == 3:  # 彩色图像
            return np.stack([self._convolve2d(image[:,:,i], kernel) 
                           for i in range(3)], axis=2)
        return self._convolve2d(image, kernel)
    
    def _convolve2d(self, image, kernel):
        """单通道2D卷积"""
        # 获取图像和核的尺寸
        ih, iw = image.shape
        kh, kw = kernel.shape
        
        # 计算填充大小
        pad_h = kh // 2
        pad_w = kw // 2
        
        # 添加边缘填充
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 
                       mode='edge')
        
        # 初始化输出(使用浮点提高精度)
        output = np.zeros_like(image, dtype=np.float32)
        
        # 执行卷积
        for y in range(ih):
            for x in range(iw):
                output[y,x] = np.sum(padded[y:y+kh, x:x+kw].astype(np.float32) * kernel)
        
        # 裁剪到0-255范围并转换为uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
    def detect_edges(self):
        """使用Sobel算子检测边缘"""
        if self.image_data is None:
            return
            
        # 创建图像副本
        img_copy = self.image_data.copy()
        
        # Sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        if len(img_copy.shape) == 3:  # 彩色图像
            # 对每个通道单独计算边缘
            edges = np.zeros_like(img_copy)
            for i in range(3):
                grad_x = self._convolve2d(img_copy[:,:,i].astype(np.float32), sobel_x)
                grad_y = self._convolve2d(img_copy[:,:,i].astype(np.float32), sobel_y)
                edges[:,:,i] = np.sqrt(grad_x**2 + grad_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        else:  # 灰度图像
            grad_x = self._convolve2d(img_copy.astype(np.float32), sobel_x)
            grad_y = self._convolve2d(img_copy.astype(np.float32), sobel_y)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
            
        self.image_data = edges
        self.add_to_history()
        self.update_image_views()
    
    def rotate_image(self, angle):
        """旋转图像"""
        if self.image_data is None:
            return
            
        # 创建图像副本
        img_copy = self.image_data.copy()
        
        # 执行旋转
        if angle == 90:
            if len(img_copy.shape) == 3:  # 彩色图像
                rotated = np.rot90(img_copy, axes=(1,0))
            else:  # 灰度图像
                rotated = np.rot90(img_copy)
        elif angle == 180:
            if len(img_copy.shape) == 3:  # 彩色图像
                rotated = np.rot90(img_copy, 2, axes=(1,0))
            else:  # 灰度图像
                rotated = np.rot90(img_copy, 2)
        elif angle == 270:
            if len(img_copy.shape) == 3:  # 彩色图像
                rotated = np.rot90(img_copy, 3, axes=(1,0))
            else:  # 灰度图像
                rotated = np.rot90(img_copy, 3)
        else:
            rotated = img_copy  # 不处理其他角度
            
        self.image_data = rotated
        self.add_to_history()
        self.update_image_views()
    
    def show_rotate_dialog(self):
        """显示旋转对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("旋转图像")
        layout = QVBoxLayout()
        
        combo = QComboBox()
        combo.addItems(["90度", "180度", "270度"])
        layout.addWidget(combo)
        
        btn = QPushButton("确定")
        btn.clicked.connect(lambda: self.rotate_image(90 * (combo.currentIndex() + 1)))
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def start_crop(self):
        """开始裁剪模式"""
        if self.image_data is None:
            return
            
        self.is_cropping = True
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        self.update_image_views()
        
    def apply_crop(self):
        """应用裁剪"""
        if not self.crop_rect or self.image_data is None:
            return
            
        # 获取裁剪区域坐标
        x = int(self.crop_rect.x())
        y = int(self.crop_rect.y())
        w = int(self.crop_rect.width())
        h = int(self.crop_rect.height())
        
        # 使用numpy数组切片进行裁剪
        if len(self.image_data.shape) == 3:  # 彩色图像
            self.image_data = self.image_data[y:y+h, x:x+w, :]
        else:  # 灰度图像
            self.image_data = self.image_data[y:y+h, x:x+w]
            
        self.is_cropping = False
        self.add_to_history()
        self.update_image_views()
        
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if self.is_cropping and event.button() == Qt.LeftButton:
            # 获取点击位置相对于图像视图的坐标
            pos = self.processed_view.mapFromGlobal(event.globalPos())
            scene_pos = self.processed_view.mapToScene(pos)
            
            # 转换为图像坐标
            img_width = self.image_data.shape[1]
            img_height = self.image_data.shape[0]
            view_width = self.processed_view.width()
            view_height = self.processed_view.height()
            
            scale_x = img_width / view_width
            scale_y = img_height / view_height
            
            x = int(scene_pos.x() * scale_x)
            y = int(scene_pos.y() * scale_y)
            
            # 确保坐标在图像范围内
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            
            self.crop_start = QPointF(x, y)
            self.crop_end = QPointF(x, y)
            self.update_image_views()
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.is_cropping and self.crop_start:
            # 获取当前位置相对于图像视图的坐标
            pos = self.processed_view.mapFromGlobal(event.globalPos())
            scene_pos = self.processed_view.mapToScene(pos)
            
            # 转换为图像坐标
            img_width = self.image_data.shape[1]
            img_height = self.image_data.shape[0]
            view_width = self.processed_view.width()
            view_height = self.processed_view.height()
            
            scale_x = img_width / view_width
            scale_y = img_height / view_height
            
            x = int(scene_pos.x() * scale_x)
            y = int(scene_pos.y() * scale_y)
            
            # 确保坐标在图像范围内
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            
            self.crop_end = QPointF(x, y)
            self.update_image_views()
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if self.is_cropping and event.button() == Qt.LeftButton:
            # 创建裁剪矩形
            x1 = min(self.crop_start.x(), self.crop_end.x())
            y1 = min(self.crop_start.y(), self.crop_end.y())
            x2 = max(self.crop_start.x(), self.crop_end.x())
            y2 = max(self.crop_start.y(), self.crop_end.y())
            
            self.crop_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            self.apply_crop()

    def update_image_views(self):
        """更新图像显示"""
        # 清空场景
        self.original_scene.clear()
        self.processed_scene.clear()
        
        if self.image_data is None:
            return
            
        def create_scaled_pixmap(image_data):
            """创建缩放后的pixmap，保持原尺寸"""
            # 确保图像数据是连续的
            image_data = np.ascontiguousarray(image_data)
            
            if len(image_data.shape) == 2:  # 灰度图像
                qimage = QImage(image_data.data, 
                               image_data.shape[1], 
                               image_data.shape[0], 
                               image_data.shape[1], 
                               QImage.Format_Grayscale8)
            else:  # 彩色图像
                # 确保是3通道RGB格式
                if image_data.shape[2] == 4:  # 如果有alpha通道，去掉
                    image_data = image_data[:,:,:3]
                qimage = QImage(image_data.data, 
                               image_data.shape[1], 
                               image_data.shape[0], 
                               image_data.shape[1] * 3, 
                               QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            # 保持原图尺寸显示
            return pixmap.scaled(
                self.original_view.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # 显示原图和处理后图像
        try:
            original_pixmap = create_scaled_pixmap(self.original_image)
            processed_pixmap = create_scaled_pixmap(self.image_data)
            
            self.original_scene.addPixmap(original_pixmap)
            self.processed_scene.addPixmap(processed_pixmap)
        except Exception as e:
            print(f"Error updating image views: {e}")
        
        # 如果正在裁剪，显示裁剪矩形
        if self.is_cropping and self.crop_start and self.crop_end:
            pen = QPen(Qt.DashLine)
            pen.setColor(Qt.red)
            pen.setWidth(2)
            self.processed_scene.addRect(QRectF(self.crop_start, self.crop_end), pen)

    def add_to_history(self):
        """添加当前状态到历史记录"""
        if self.image_data is None:
            return
            
        # 如果当前不是最新状态，删除之后的所有历史
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
            
        # 添加新状态
        self.history.append(self.image_data.copy())
        self.history_index = len(self.history) - 1
        
        # 更新按钮状态
        self.update_undo_redo_buttons()

    def undo_action(self):
        """撤销操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.image_data = self.history[self.history_index].copy()
            self.update_image_views()
            self.update_undo_redo_buttons()

    def redo_action(self):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.image_data = self.history[self.history_index].copy()
            self.update_image_views()
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """更新撤销/重做按钮状态"""
        self.undo_btn.setEnabled(self.history_index > 0)
        self.redo_btn.setEnabled(self.history_index < len(self.history) - 1)
        
    def adjust_brightness_contrast(self):
        """调整亮度和对比度"""
        if self.image_data is None:
            return
            
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        
        # 更新标签显示
        self.brightness_label.setText(f"亮度: {brightness}")
        self.contrast_label.setText(f"对比度: {contrast}")
        
        # 创建图像副本
        img_copy = self.original_image.copy() if self.original_image is not None else self.image_data.copy()
        
        # 调整亮度
        if brightness != 0:
            if brightness > 0:
                img_copy = np.clip(img_copy.astype(np.int32) + brightness, 0, 255).astype(np.uint8)
            else:
                img_copy = np.clip(img_copy.astype(np.int32) + brightness, 0, 255).astype(np.uint8)
        
        # 调整对比度
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img_copy = np.clip(factor * (img_copy.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
            
        self.image_data = img_copy
        self.add_to_history()
        self.update_image_views()

    def show_threshold_dialog(self):
        """显示阈值处理对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("阈值处理")
        layout = QVBoxLayout()
        
        # 阈值类型选择
        type_combo = QComboBox()
        type_combo.addItems(["二进制", "反二进制", "截断", "归零", "反归零"])
        layout.addWidget(QLabel("阈值类型:"))
        layout.addWidget(type_combo)
        
        # 阈值滑块
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(0, 255)
        threshold_slider.setValue(127)
        threshold_label = QLabel("阈值: 127")
        layout.addWidget(threshold_label)
        layout.addWidget(threshold_slider)
        
        def update_threshold(value):
            threshold_label.setText(f"阈值: {value}")
            self.apply_threshold(type_combo.currentIndex(), value)
            
        threshold_slider.valueChanged.connect(update_threshold)
        
        # 确定按钮
        btn = QPushButton("确定")
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_threshold(self, type_idx, threshold):
        """应用阈值处理"""
        if self.image_data is None:
            return
            
        # 如果是彩色图像，先转换为灰度
        if len(self.image_data.shape) == 3:
            self.convert_to_grayscale()
            
        # 根据类型应用阈值
        if type_idx == 0:  # 二进制
            self.image_data = np.where(self.image_data > threshold, 255, 0).astype(np.uint8)
        elif type_idx == 1:  # 反二进制
            self.image_data = np.where(self.image_data > threshold, 0, 255).astype(np.uint8)
        elif type_idx == 2:  # 截断
            self.image_data = np.where(self.image_data > threshold, threshold, self.image_data).astype(np.uint8)
        elif type_idx == 3:  # 归零
            self.image_data = np.where(self.image_data > threshold, self.image_data, 0).astype(np.uint8)
        elif type_idx == 4:  # 反归零
            self.image_data = np.where(self.image_data > threshold, 0, self.image_data).astype(np.uint8)
            
        self.add_to_history()
        self.update_image_views()

    def show_morphology_dialog(self):
        """显示形态学变换对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("形态学变换")
        layout = QVBoxLayout()
        
        # 操作类型选择
        op_combo = QComboBox()
        op_combo.addItems(["开运算", "闭运算", "膨胀", "腐蚀"])
        layout.addWidget(QLabel("操作类型:"))
        layout.addWidget(op_combo)
        
        # 核大小选择
        kernel_combo = QComboBox()
        kernel_combo.addItems(["3x3", "5x5", "7x7"])
        layout.addWidget(QLabel("核大小:"))
        layout.addWidget(kernel_combo)
        
        # 确定按钮
        btn = QPushButton("确定")
        btn.clicked.connect(lambda: self.apply_morphology(op_combo.currentIndex(), 
                           kernel_combo.currentIndex()))
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_morphology(self, op_idx, kernel_idx):
        """应用形态学变换"""
        if self.image_data is None:
            return
            
        # 如果是彩色图像，先转换为灰度
        if len(self.image_data.shape) == 3:
            self.convert_to_grayscale()
            
        # 创建核
        kernel_size = 3 + 2 * kernel_idx
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 应用形态学操作
        if op_idx == 0:  # 开运算
            self.image_data = cv2.morphologyEx(self.image_data, cv2.MORPH_OPEN, kernel)
        elif op_idx == 1:  # 闭运算
            self.image_data = cv2.morphologyEx(self.image_data, cv2.MORPH_CLOSE, kernel)
        elif op_idx == 2:  # 膨胀
            self.image_data = cv2.dilate(self.image_data, kernel)
        elif op_idx == 3:  # 腐蚀
            self.image_data = cv2.erode(self.image_data, kernel)
            
        self.add_to_history()
        self.update_image_views()

    def show_segmentation_dialog(self):
        """显示图像分割对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("图像分割")
        layout = QVBoxLayout()
        
        # 分割方法选择
        method_combo = QComboBox()
        method_combo.addItems(["阈值分割", "边缘检测", "区域生长", "K-means聚类"])
        layout.addWidget(QLabel("分割方法:"))
        layout.addWidget(method_combo)
        
        # 参数设置
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("参数:"))
        param_spin = QSpinBox()
        param_spin.setRange(0, 255)
        param_spin.setValue(128)
        param_layout.addWidget(param_spin)
        layout.addLayout(param_layout)
        
        # 确定按钮
        btn = QPushButton("确定")
        btn.clicked.connect(lambda: self.apply_segmentation(
            method_combo.currentIndex(), param_spin.value()))
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_segmentation(self, method_idx, param):
        """应用图像分割"""
        if self.image_data is None:
            return
            
        # 创建图像副本进行处理
        img_copy = self.image_data.copy()
        
        # 根据方法应用分割
        if method_idx == 0:  # 阈值分割
            if len(img_copy.shape) == 3:  # 彩色图像
                # 对每个通道单独处理
                for i in range(3):
                    _, img_copy[:,:,i] = cv2.threshold(img_copy[:,:,i], param, 255, cv2.THRESH_BINARY)
            else:  # 灰度图像
                _, img_copy = cv2.threshold(img_copy, param, 255, cv2.THRESH_BINARY)
                
        elif method_idx == 1:  # 边缘检测
            if len(img_copy.shape) == 3:  # 彩色图像
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            img_copy = cv2.Canny(img_copy, param, param*2)
            
        elif method_idx == 2:  # 区域生长
            if len(img_copy.shape) == 3:  # 彩色图像
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
            seed = (img_copy.shape[0]//2, img_copy.shape[1]//2)
            img_copy = self.region_growing(img_copy, seed, param)
            
        elif method_idx == 3:  # K-means聚类
            if len(img_copy.shape) == 3:  # 彩色图像
                # 彩色图像K-means
                pixel_values = img_copy.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                _, labels, centers = cv2.kmeans(pixel_values, param, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                img_copy = centers[labels.flatten()].reshape(img_copy.shape)
            else:  # 灰度图像
                pixel_values = img_copy.reshape((-1, 1)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                _, labels, centers = cv2.kmeans(pixel_values, param, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                img_copy = centers[labels.flatten()].reshape(img_copy.shape)
        
        self.image_data = img_copy
        self.add_to_history()
        self.update_image_views()

    def region_growing(self, img, seed, threshold):
        """区域生长算法实现"""
        # 创建标记矩阵
        h, w = img.shape
        region = np.zeros((h, w), dtype=np.uint8)
        
        # 获取种子点
        x, y = seed
        region[x, y] = 255
        seed_value = img[x, y]
        
        # 定义8邻域
        neighbors = [(-1,-1),(-1,0),(-1,1),
                    (0,-1),         (0,1),
                    (1,-1), (1,0), (1,1)]
        
        # 生长过程
        changed = True
        while changed:
            changed = False
            # 找到所有边界点
            border = np.where(region == 255)
            for i in range(len(border[0])):
                x, y = border[0][i], border[1][i]
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if region[nx, ny] == 0 and abs(int(img[nx, ny]) - int(seed_value)) <= threshold:
                            region[nx, ny] = 255
                            changed = True
        return region

    def show_filter_dialog(self):
        """显示图像滤波对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("图像滤波")
        layout = QVBoxLayout()
        
        # 滤波类型选择
        type_combo = QComboBox()
        type_combo.addItems(["低通滤波", "高通滤波", "带通滤波"])
        layout.addWidget(QLabel("滤波类型:"))
        layout.addWidget(type_combo)
        
        # 低频阈值滑块
        low_slider = QSlider(Qt.Horizontal)
        low_slider.setRange(0, 100)
        low_slider.setValue(30)
        low_label = QLabel("低频阈值: 30")
        layout.addWidget(low_label)
        layout.addWidget(low_slider)
        
        # 高频阈值滑块
        high_slider = QSlider(Qt.Horizontal)
        high_slider.setRange(0, 100)
        high_slider.setValue(70)
        high_label = QLabel("高频阈值: 70")
        layout.addWidget(high_label)
        layout.addWidget(high_slider)
        
        def update_filter():
            low = low_slider.value()
            high = high_slider.value()
            low_label.setText(f"低频阈值: {low}")
            high_label.setText(f"高频阈值: {high}")
            self.apply_filter(type_combo.currentIndex(), low, high)
            
        low_slider.valueChanged.connect(update_filter)
        high_slider.valueChanged.connect(update_filter)
        
        # 确定按钮
        btn = QPushButton("确定")
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_filter(self, type_idx, low, high):
        """应用图像滤波"""
        if self.image_data is None:
            return
            
        # 如果是彩色图像，先转换为灰度
        if len(self.image_data.shape) == 3:
            self.convert_to_grayscale()
            
        # 执行FFT
        f = np.fft.fft2(self.image_data)
        fshift = np.fft.fftshift(f)
        
        # 创建滤波器
        rows, cols = self.image_data.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows, cols), np.uint8)
        
        if type_idx == 0:  # 低通滤波
            r = low
            mask[crow-r:crow+r, ccol-r:ccol+r] = 1
        elif type_idx == 1:  # 高通滤波
            r = high
            mask = np.ones((rows, cols), np.uint8)
            mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        elif type_idx == 2:  # 带通滤波
            r_low = low
            r_high = high
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow-r_high:crow+r_high, ccol-r_high:ccol+r_high] = 1
            mask[crow-r_low:crow+r_low, ccol-r_low:ccol+r_low] = 0
            
        # 应用滤波器
        fshift = fshift * mask
        
        # 逆FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        self.image_data = np.abs(img_back).astype(np.uint8)
        
        self.add_to_history()
        self.update_image_views()

    def show_histogram(self):
        """显示图像直方图"""
        if self.image_data is None:
            return
            
        # 创建新窗口
        hist_window = QDialog(self)
        hist_window.setWindowTitle("图像直方图")
        hist_window.setGeometry(200, 200, 800, 600)
        
        # 创建图表
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # 计算直方图
        if len(self.image_data.shape) == 3:  # 彩色图像
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.image_data], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
            ax.legend(['Red', 'Green', 'Blue'])
        else:  # 灰度图像
            hist = cv2.calcHist([self.image_data], [0], None, [256], [0, 256])
            ax.plot(hist, color='k')
            
        ax.set_title('图像直方图')
        ax.set_xlabel('像素值')
        ax.set_ylabel('频数')
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        hist_window.setLayout(layout)
        hist_window.exec_()

    def show_flip_dialog(self):
        """显示翻转图像对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("翻转图像")
        layout = QVBoxLayout()
        
        # 翻转方式选择
        flip_combo = QComboBox()
        flip_combo.addItems(["水平翻转", "垂直翻转"])
        layout.addWidget(QLabel("翻转方式:"))
        layout.addWidget(flip_combo)
        
        # 确定按钮
        btn = QPushButton("确定")
        btn.clicked.connect(lambda: self.flip_image(flip_combo.currentIndex()))
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def flip_image(self, flip_type):
        """翻转图像"""
        if self.image_data is None:
            return
            
        # 0=水平翻转, 1=垂直翻转
        if flip_type == 0:
            self.image_data = np.fliplr(self.image_data)
        else:
            self.image_data = np.flipud(self.image_data)
            
        self.add_to_history()
        self.update_image_views()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ImageEditorPS()
    editor.show()
    sys.exit(app.exec_())
