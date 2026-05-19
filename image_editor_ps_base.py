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
        
        # 亮度/对比度调整状态
        self.cumulative_brightness = 0
        self.cumulative_contrast = 0
        
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
        self.gaussian_blur_btn = QPushButton("高斯模糊")
        self.gaussian_blur_btn.clicked.connect(self.show_gaussian_blur_dialog)
        process_layout.addWidget(self.gaussian_blur_btn)
        
        # 边缘检测
        self.edge_detect_btn = QPushButton("边缘检测")
        self.edge_detect_btn.clicked.connect(self.sobel_edge_detection)
        process_layout.addWidget(self.edge_detect_btn)
        
        # 图像旋转
        self.rotate_btn = QPushButton("旋转图像")
        self.rotate_btn.clicked.connect(self.show_rotate_dialog)
        process_layout.addWidget(self.rotate_btn)
        
        # 图像裁剪
        self.crop_btn = QPushButton("裁剪图像")
        self.crop_btn.clicked.connect(self.start_cropping)
        process_layout.addWidget(self.crop_btn)
        
        # 直方图
        self.histogram_btn = QPushButton("显示直方图")
        self.histogram_btn.clicked.connect(self.show_histogram)
        process_layout.addWidget(self.histogram_btn)
        
        # 图像翻转
        self.flip_btn = QPushButton("翻转图像")
        self.flip_btn.clicked.connect(self.show_flip_dialog)
        process_layout.addWidget(self.flip_btn)
        
        # 形态学变换
        self.morphology_btn = QPushButton("形态学变换")
        self.morphology_btn.clicked.connect(self.show_morphology_dialog)
        process_layout.addWidget(self.morphology_btn)
        
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
        
        # 添加拉伸使布局更美观
        control_panel.addStretch()
        
        # 亮度和对比度调整
        adjust_group = QGroupBox("亮度和对比度")
        adjust_layout = QVBoxLayout()
        
        # 亮度调整
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("亮度:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_value = QLabel("0")
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_value)
        adjust_layout.addLayout(brightness_layout)
        
        # 对比度调整
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("对比度:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_value = QLabel("0")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_value)
        adjust_layout.addLayout(contrast_layout)
        
        # 应用按钮
        self.apply_adjust_btn = QPushButton("应用调整")
        self.apply_adjust_btn.clicked.connect(self.apply_brightness_contrast)
        adjust_layout.addWidget(self.apply_adjust_btn)
        
        adjust_group.setLayout(adjust_layout)
        control_panel.addWidget(adjust_group)
        
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
                for c in range(3):
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

    def apply_gaussian_blur(self, kernel_size=3, sigma=1.0):
        """应用高斯模糊"""
        if self.image_data is None:
            return
            
        try:
            # 生成高斯核
            kernel = self.gaussian_kernel(kernel_size, sigma)
            
            # 处理彩色或灰度图像
            if len(self.image_data.shape) == 3:  # 彩色
                blurred = np.zeros_like(self.image_data)
                for c in range(3):  # 对每个通道分别处理
                    blurred[..., c] = self.convolve2d(self.image_data[..., c], kernel)
            else:  # 灰度
                blurred = self.convolve2d(self.image_data, kernel)
                
            self.image_data = blurred.astype(np.uint8)
            self.update_processed_display()
            self.add_to_history()
        except Exception as e:
            print(f"高斯模糊错误: {e}")

    def convolve2d(self, image, kernel):
        """二维卷积实现"""
        kernel = np.flipud(np.fliplr(kernel))  # 翻转核
        kh, kw = kernel.shape
        ih, iw = image.shape
        
        # 输出图像
        output = np.zeros_like(image)
        
        # 填充图像
        pad = kh // 2
        padded = np.pad(image, pad, mode='reflect')
        
        # 卷积运算
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
                
        return output

    def show_gaussian_blur_dialog(self):
        """显示高斯模糊参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("高斯模糊参数")
        layout = QVBoxLayout()
        
        # 核大小选择
        size_label = QLabel("核大小 (3-15 奇数):")
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(3, 15)
        self.kernel_size_spin.setSingleStep(2)
        self.kernel_size_spin.setValue(3)
        layout.addWidget(size_label)
        layout.addWidget(self.kernel_size_spin)
        
        # Sigma值选择
        sigma_label = QLabel("Sigma值 (0.1-5.0):")
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)
        layout.addWidget(sigma_label)
        layout.addWidget(self.sigma_spin)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_gaussian_blur_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

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

    def show_edge_detect_dialog(self):
        """显示边缘检测参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("边缘检测参数")
        layout = QVBoxLayout()
        
        # 阈值选择
        threshold_label = QLabel("阈值 (0-255):")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(50)
        layout.addWidget(threshold_label)
        layout.addWidget(self.threshold_spin)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_edge_detect_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

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

    def show_rotate_dialog(self):
        """显示旋转参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("旋转图像")
        layout = QVBoxLayout()
        
        # 角度选择
        angle_label = QLabel("选择旋转角度:")
        self.angle_combo = QComboBox()
        self.angle_combo.addItems(["90°", "180°", "270°"])
        layout.addWidget(angle_label)
        layout.addWidget(self.angle_combo)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_rotate_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def start_cropping(self):
        """开始裁剪模式"""
        if self.image_data is None:
            return
            
        self.is_cropping = True
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        self.update_image_display()
        
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if self.is_cropping and event.button() == Qt.LeftButton:
            pos = self.original_view.mapFromGlobal(event.globalPos())
            scene_pos = self.original_view.mapToScene(pos)
            self.crop_start = scene_pos
            self.crop_end = scene_pos
            self.update_image_display()
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.is_cropping and self.crop_start:
            pos = self.original_view.mapFromGlobal(event.globalPos())
            scene_pos = self.original_view.mapToScene(pos)
            self.crop_end = scene_pos
            self.update_image_display()
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if self.is_cropping and event.button() == Qt.LeftButton and self.crop_start:
            pos = self.original_view.mapFromGlobal(event.globalPos())
            scene_pos = self.original_view.mapToScene(pos)
            self.crop_end = scene_pos
            
            # 确保起点和终点形成有效矩形
            if self.crop_start != self.crop_end:
                self.apply_crop()
                
            self.is_cropping = False
            self.update_image_display()
            
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

    def update_image_display(self):
        """更新图像显示"""
        if self.image_data is not None:
            # 确保数据是连续的
            if not self.image_data.flags['C_CONTIGUOUS']:
                self.image_data = np.ascontiguousarray(self.image_data)
            
            # 转换为QImage显示
            if len(self.image_data.shape) == 2:  # 灰度图像
                height, width = self.image_data.shape
                bytes_per_line = width
                img_bytes = self.image_data.tobytes()
                qimage = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # 彩色图像
                height, width, channel = self.image_data.shape
                bytes_per_line = 3 * width
                img_bytes = self.image_data.tobytes()
            self.original_scene.addPixmap(pixmap)
            self.processed_scene.clear()
            self.processed_scene.addPixmap(pixmap)
            
            # 如果正在裁剪，绘制裁剪矩形
            if self.is_cropping and self.crop_start and self.crop_end:
                # 计算矩形区域
                rect = QRectF(self.crop_start, self.crop_end).normalized()
                
                # 创建虚线笔
                pen = QPen(Qt.white, 1, Qt.DashLine)
                pen.setDashPattern([4, 4])
                
                # 绘制矩形
                self.original_scene.addRect(rect, pen)
                self.processed_scene.addRect(rect, pen)

    def on_rotate_apply(self, dialog):
        """应用旋转参数"""
        angle_text = self.angle_combo.currentText()
        angle = int(angle_text.replace("°", ""))
        self.rotate_image(angle)
        dialog.close()

    def on_edge_detect_apply(self, dialog):
        """应用边缘检测参数"""
        threshold = self.threshold_spin.value()
        self.sobel_edge_detection(threshold)
        dialog.close()

    def on_gaussian_blur_apply(self, dialog):
        """应用高斯模糊参数"""
        kernel_size = self.kernel_size_spin.value()
        sigma = self.sigma_spin.value()
        self.apply_gaussian_blur(kernel_size, sigma)
        dialog.close()

    def convert_to_grayscale(self):
        """将图像转换为灰度图"""
        if self.image_data is not None:
            try:
                # 手动实现灰度转换 (不使用cv2)
                if len(self.image_data.shape) == 3:  # 彩色图像
                    # 使用加权平均法 (ITU-R BT.601标准)
                    gray_data = np.dot(self.image_data[...,:3], [0.299, 0.587, 0.114])
                    gray_data = gray_data.astype(np.uint8)
                    self.image_data = gray_data
                else:  # 已经是灰度图
                    return
                
                self.update_processed_display()
                self.add_to_history()
            except Exception as e:
                print(f"灰度转换错误: {e}")

    def update_processed_display(self):
        """更新处理后图像显示"""
        if self.image_data is not None:
            # 确保数据是连续的
            if not self.image_data.flags['C_CONTIGUOUS']:
                self.image_data = np.ascontiguousarray(self.image_data)
            
            # 转换为QImage显示
            if len(self.image_data.shape) == 2:  # 灰度图像
                height, width = self.image_data.shape
                bytes_per_line = width
                img_bytes = self.image_data.tobytes()
                qimage = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # 彩色图像
                height, width, channel = self.image_data.shape
                bytes_per_line = 3 * width
                img_bytes = self.image_data.tobytes()
                qimage = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.processed_scene.clear()
            self.processed_scene.addPixmap(pixmap)

    def save_image(self):
        """保存图像文件"""
        if self.image_data is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", 
                        "PNG图像 (*.png);;BMP图像 (*.bmp);;JPEG图像 (*.jpg *.jpeg)")
        if file_path:
            try:
                # 使用cv2保存图像(仅用于保存)
                cv2.imwrite(file_path, cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"保存图像错误: {e}")

    def update_image_display(self):
        """更新图像显示"""
        if self.image_data is not None:
            # 转换为QImage显示
            if len(self.image_data.shape) == 2:  # 灰度图像
                height, width = self.image_data.shape
                bytes_per_line = width
                # 确保数据是连续的并转换为bytes
                img_bytes = np.ascontiguousarray(self.image_data).tobytes()
                qimage = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # 彩色图像
                height, width, channel = self.image_data.shape
                bytes_per_line = 3 * width
                # 确保数据是连续的并转换为bytes
                img_bytes = np.ascontiguousarray(self.image_data).tobytes()
                qimage = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            
            # 清除场景并添加新图像
            self.original_scene.clear()
            self.original_scene.addPixmap(pixmap)
            self.processed_scene.clear()
            self.processed_scene.addPixmap(pixmap)

    def add_to_history(self):
        """添加当前状态到历史记录"""
        if self.image_data is not None:
            # 只保留当前位置之前的历史记录
            self.history = self.history[:self.history_index + 1]
            # 添加新状态
            self.history.append(self.image_data.copy())
            self.history_index = len(self.history) - 1
            # 更新按钮状态
            self.update_undo_redo_buttons()

    def show_filter_dialog(self):
        """显示图像滤波参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("图像滤波")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        # 低频阈值
        low_label = QLabel("低频阈值 (0-100):")
        self.low_freq_slider = QSlider(Qt.Horizontal)
        self.low_freq_slider.setRange(0, 100)
        self.low_freq_slider.setValue(30)
        self.low_freq_value = QLabel("30")
        low_layout = QHBoxLayout()
        low_layout.addWidget(low_label)
        low_layout.addWidget(self.low_freq_slider)
        low_layout.addWidget(self.low_freq_value)
        layout.addLayout(low_layout)
        
        # 高频阈值
        high_label = QLabel("高频阈值 (0-100):")
        self.high_freq_slider = QSlider(Qt.Horizontal)
        self.high_freq_slider.setRange(0, 100)
        self.high_freq_slider.setValue(70)
        self.high_freq_value = QLabel("70")
        high_layout = QHBoxLayout()
        high_layout.addWidget(high_label)
        high_layout.addWidget(self.high_freq_slider)
        high_layout.addWidget(self.high_freq_value)
        layout.addLayout(high_layout)
        
        # 连接滑块值变化信号
        self.low_freq_slider.valueChanged.connect(
            lambda v: self.low_freq_value.setText(str(v)))
        self.high_freq_slider.valueChanged.connect(
            lambda v: self.high_freq_value.setText(str(v)))
        
        # 连接亮度/对比度滑块信号
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_value.setText(str(v)))
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_value.setText(str(v)))
        
        # 实时应用调整
        self.brightness_slider.valueChanged.connect(
            lambda v: self.apply_brightness_contrast(v, None))
        self.contrast_slider.valueChanged.connect(
            lambda v: self.apply_brightness_contrast(None, v))
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_filter_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

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

    def on_filter_apply(self, dialog):
        """应用滤波参数"""
        low_threshold = self.low_freq_slider.value()
        high_threshold = self.high_freq_slider.value()
        self.apply_filter(low_threshold, high_threshold)
        dialog.close()

    def show_segmentation_dialog(self):
        """显示图像分割参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("图像分割")
        layout = QVBoxLayout()
        
        # 分割方法选择
        method_label = QLabel("选择分割方法:")
        self.seg_method_combo = QComboBox()
        self.seg_method_combo.addItems(["阈值分割", "边缘检测分割", "区域生长", "分水岭"])
        layout.addWidget(method_label)
        layout.addWidget(self.seg_method_combo)
        
        # 阈值选择(仅当选择阈值分割时显示)
        self.threshold_label = QLabel("阈值 (0-255):")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(127)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_spin)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_segmentation_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

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

    def region_growing(self, image):
        """区域生长算法(简化版)"""
        # 随机选择种子点
        height, width = image.shape
        seed = (np.random.randint(0, height), np.random.randint(0, width))
        
        # 创建标记图像
        markers = np.zeros_like(image)
        markers[seed] = 255
        
        # 简单区域生长
        threshold = 20
        for i in range(1, height-1):
            for j in range(1, width-1):
                if markers[i,j] == 255:  # 如果是种子点
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if abs(int(image[i+di,j+dj]) - int(image[i,j])) < threshold:
                                markers[i+di,j+dj] = 255
        return markers

    def watershed_segmentation(self, image):
        """分水岭算法"""
        # 应用Otsu阈值
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 去除噪声
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # 找到未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), markers)
        image[markers == -1] = 255  # 标记边界
        
        return image

    def on_segmentation_apply(self, dialog):
        """应用图像分割参数"""
        method_text = self.seg_method_combo.currentText()
        threshold = self.threshold_spin.value()
        
        # 映射方法类型
        method_mapping = {
            "阈值分割": "threshold",
            "边缘检测分割": "edge",
            "区域生长": "region",
            "分水岭": "watershed"
        }
        
        self.apply_segmentation(method_mapping[method_text], threshold)
        dialog.close()

    def show_morphology_dialog(self):
        """显示形态学变换参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("形态学变换")
        layout = QVBoxLayout()
        
        # 操作类型选择
        op_label = QLabel("选择操作类型:")
        self.morph_op_combo = QComboBox()
        self.morph_op_combo.addItems(["腐蚀", "膨胀", "开运算", "闭运算"])
        layout.addWidget(op_label)
        layout.addWidget(self.morph_op_combo)
        
        # 核大小选择
        kernel_label = QLabel("选择核大小:")
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItems(["3x3", "5x5", "7x7"])
        layout.addWidget(kernel_label)
        layout.addWidget(self.kernel_size_combo)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_morphology_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_morphology(self, operation, kernel_size):
        """应用形态学变换
        operation: 'erode', 'dilate', 'open', 'close'
        kernel_size: 3, 5, 7
        """
        if self.image_data is None:
            return
            
        try:
            # 创建结构元素(核)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # 处理彩色或灰度图像
            if len(self.image_data.shape) == 3:  # 彩色
                result = np.zeros_like(self.image_data)
                for c in range(3):  # 对每个通道分别处理
                    channel = self.image_data[..., c]
                    if operation == 'erode':
                        result[..., c] = self.erode(channel, kernel)
                    elif operation == 'dilate':
                        result[..., c] = self.dilate(channel, kernel)
                    elif operation == 'open':
                        result[..., c] = self.dilate(self.erode(channel, kernel), kernel)
                    elif operation == 'close':
                        result[..., c] = self.erode(self.dilate(channel, kernel), kernel)
            else:  # 灰度
                if operation == 'erode':
                    result = self.erode(self.image_data, kernel)
                elif operation == 'dilate':
                    result = self.dilate(self.image_data, kernel)
                elif operation == 'open':
                    result = self.dilate(self.erode(self.image_data, kernel), kernel)
                elif operation == 'close':
                    result = self.erode(self.dilate(self.image_data, kernel), kernel)
                    
            self.image_data = result
            self.update_processed_display()
            self.add_to_history()
        except Exception as e:
            print(f"形态学变换错误: {e}")

    def erode(self, image, kernel):
        """腐蚀操作"""
        output = np.zeros_like(image)
        k_height, k_width = kernel.shape
        pad_height = k_height // 2
        pad_width = k_width // 2
        
        # 填充图像
        padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                    mode='constant', constant_values=255)
        
        # 腐蚀操作
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+k_height, j:j+k_width]
                output[i, j] = np.min(region * kernel)
                
        return output

    def dilate(self, image, kernel):
        """膨胀操作"""
        output = np.zeros_like(image)
        k_height, k_width = kernel.shape
        pad_height = k_height // 2
        pad_width = k_width // 2
        
        # 填充图像
        padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                    mode='constant', constant_values=0)
        
        # 膨胀操作
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+k_height, j:j+k_width]
                output[i, j] = np.max(region * kernel)
                
        return output

    def on_morphology_apply(self, dialog):
        """应用形态学变换参数"""
        op_text = self.morph_op_combo.currentText()
        kernel_text = self.kernel_size_combo.currentText()
        
        # 映射操作类型
        op_mapping = {
            "腐蚀": "erode",
            "膨胀": "dilate", 
            "开运算": "open",
            "闭运算": "close"
        }
        
        # 获取核大小
        kernel_size = int(kernel_text.split('x')[0])
        
        self.apply_morphology(op_mapping[op_text], kernel_size)
        dialog.close()

    def show_flip_dialog(self):
        """显示翻转参数对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("翻转图像")
        layout = QVBoxLayout()
        
        # 翻转方式选择
        flip_label = QLabel("选择翻转方式:")
        self.flip_combo = QComboBox()
        self.flip_combo.addItems(["水平翻转", "垂直翻转"])
        layout.addWidget(flip_label)
        layout.addWidget(self.flip_combo)
        
        # 确认按钮
        btn = QPushButton("应用")
        btn.clicked.connect(lambda: self.on_flip_apply(dialog))
        layout.addWidget(btn)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def flip_image(self, direction):
        """翻转图像
        direction: 'horizontal' 或 'vertical'
        """
        if self.image_data is None:
            return
            
        try:
            print(f"\n翻转前图像形状: {self.image_data.shape}")
            print(f"翻转前图像数据类型: {self.image_data.dtype}")
            print(f"翻转前图像值范围: {np.min(self.image_data)}-{np.max(self.image_data)}")
            
            if direction == 'horizontal':
                if len(self.image_data.shape) == 3:  # 彩色
                    self.image_data = np.flip(self.image_data, axis=1)
                else:  # 灰度
                    self.image_data = np.flip(self.image_data, axis=1)
            elif direction == 'vertical':
                if len(self.image_data.shape) == 3:  # 彩色
                    self.image_data = np.flip(self.image_data, axis=0)
                else:  # 灰度
                    self.image_data = np.flip(self.image_data, axis=0)
            
            print(f"\n翻转后图像形状: {self.image_data.shape}")
            print(f"翻转后图像数据类型: {self.image_data.dtype}")
            print(f"翻转后图像值范围: {np.min(self.image_data)}-{np.max(self.image_data)}")
            
            # 确保数据是连续的
            if not self.image_data.flags['C_CONTIGUOUS']:
                self.image_data = np.ascontiguousarray(self.image_data)
            
            self.update_processed_display()
            self.add_to_history()
        except Exception as e:
            print(f"翻转图像错误: {e}")

    def on_flip_apply(self, dialog):
        """应用翻转参数"""
        flip_text = self.flip_combo.currentText()
        if flip_text == "水平翻转":
            self.flip_image('horizontal')
        else:
            self.flip_image('vertical')
        dialog.close()

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

    def update_undo_redo_buttons(self):
        """更新撤销/重做按钮状态"""
        self.undo_btn.setEnabled(self.history_index > 0)
        self.redo_btn.setEnabled(self.history_index < len(self.history) - 1)

    def undo_action(self):
        """撤销操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.image_data = self.history[self.history_index].copy()
            self.update_image_display()
            self.update_undo_redo_buttons()

    def redo_action(self):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.image_data = self.history[self.history_index].copy()
            self.update_image_display()
            self.update_undo_redo_buttons()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ImageEditorPS()
    editor.show()
    sys.exit(app.exec_())
