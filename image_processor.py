import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
                            QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QSlider, QComboBox, QGraphicsView, 
                            QGraphicsScene, QSplitter, QToolBar, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理软件")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化UI
        self.initUI()
        
        # 图像数据
        self.image_data = None
        self.original_image = None
        
    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 使用QSplitter创建可调整的视图
        splitter = QSplitter(Qt.Horizontal)
        
        # 原图视图
        self.original_view = QGraphicsView()
        self.original_scene = QGraphicsScene()
        self.original_view.setScene(self.original_scene)
        
        # 处理后视图
        self.processed_view = QGraphicsView()
        self.processed_scene = QGraphicsScene()
        self.processed_view.setScene(self.processed_scene)
        
        splitter.addWidget(self.original_view)
        splitter.addWidget(self.processed_view)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter, 80)
        
        # 控制面板
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 20)
        
        # 功能按钮
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        control_panel.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("保存图像")
        self.save_btn.clicked.connect(self.save_image)
        control_panel.addWidget(self.save_btn)
        
        # 图像处理功能按钮
        self.grayscale_btn = QPushButton("灰度转换")
        self.grayscale_btn.clicked.connect(self.convert_to_grayscale)
        control_panel.addWidget(self.grayscale_btn)
        
        self.gaussian_btn = QPushButton("高斯模糊")
        self.gaussian_btn.clicked.connect(self.apply_gaussian_blur)
        control_panel.addWidget(self.gaussian_btn)
        
        self.edge_btn = QPushButton("边缘检测")
        self.edge_btn.clicked.connect(self.detect_edges)
        control_panel.addWidget(self.edge_btn)
        
        self.rotate_btn = QPushButton("旋转图像")
        self.rotate_btn.clicked.connect(self.rotate_image)
        control_panel.addWidget(self.rotate_btn)
        
        self.crop_btn = QPushButton("裁剪图像")
        self.crop_btn.clicked.connect(self.crop_image)
        control_panel.addWidget(self.crop_btn)
        
        self.histogram_btn = QPushButton("直方图")
        self.histogram_btn.clicked.connect(self.show_histogram)
        control_panel.addWidget(self.histogram_btn)
        
        self.brightness_btn = QPushButton("亮度调整")
        self.brightness_btn.clicked.connect(self.adjust_brightness)
        control_panel.addWidget(self.brightness_btn)
        
        self.contrast_btn = QPushButton("对比度调整")
        self.contrast_btn.clicked.connect(self.adjust_contrast)
        control_panel.addWidget(self.contrast_btn)
        
        self.threshold_btn = QPushButton("阈值处理")
        self.threshold_btn.clicked.connect(self.apply_threshold)
        control_panel.addWidget(self.threshold_btn)
        
        self.morphology_btn = QPushButton("形态学变换")
        self.morphology_btn.clicked.connect(self.apply_morphology)
        control_panel.addWidget(self.morphology_btn)
        
        self.flip_btn = QPushButton("翻转图像")
        self.flip_btn.clicked.connect(self.flip_image)
        control_panel.addWidget(self.flip_btn)
        
        self.segmentation_btn = QPushButton("图像分割")
        self.segmentation_btn.clicked.connect(self.segment_image)
        control_panel.addWidget(self.segmentation_btn)
        
        self.filter_btn = QPushButton("图像滤波")
        self.filter_btn.clicked.connect(self.apply_filter)
        control_panel.addWidget(self.filter_btn)
        
        # 添加拉伸使布局更美观
        control_panel.addStretch()
        
    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.png *.jpg *.bmp)")
        if file_path:
            # 使用Pillow加载图像并转换为numpy数组
            from PIL import Image
            img = Image.open(file_path)
            self.original_image = np.array(img)
            self.image_data = self.original_image.copy()
            self.update_image_views()
    
    def save_image(self):
        """保存处理后的图像"""
        if self.image_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg)")
            if file_path:
                from PIL import Image
                Image.fromarray(self.image_data).save(file_path)
    
    def update_image_views(self):
        """更新图像视图"""
        if self.image_data is None:
            return
            
        # 清除场景
        self.original_scene.clear()
        self.processed_scene.clear()
        
        # 显示原图
        if len(self.original_image.shape) == 2:  # 灰度图像
            height, width = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            q_img = QImage(img_bytes, width, height, width, QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.original_image.shape
            # 确保颜色通道顺序正确 (RGB -> BGR for QImage)
            bgr_image = self.original_image[:, :, ::-1].copy()
            img_bytes = np.ascontiguousarray(bgr_image).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.original_scene.addPixmap(pixmap)
        
        # 显示处理后图像
        if len(self.image_data.shape) == 2:  # 灰度图像
            height, width = self.image_data.shape
            img_bytes = np.ascontiguousarray(self.image_data).data
            q_img = QImage(img_bytes, width, height, width, QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.image_data.shape
            # 确保颜色通道顺序正确 (RGB -> BGR for QImage)
            bgr_image = self.image_data[:, :, ::-1].copy()
            img_bytes = np.ascontiguousarray(bgr_image).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.processed_scene.addPixmap(pixmap)
    
    # 以下是图像处理函数的占位符，将逐步实现
    def convert_to_grayscale(self):
        """将图像转换为灰度"""
        if self.image_data is None:
            return
            
        # 检查是否已经是灰度图像
        if len(self.image_data.shape) == 2:
            return
            
        # 手动实现灰度转换 (使用亮度公式: 0.299*R + 0.587*G + 0.114*B)
        r, g, b = self.image_data[:,:,0], self.image_data[:,:,1], self.image_data[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        self.image_data = gray.astype(np.uint8)
        
        # 更新显示
        self.update_image_views()
        
    def apply_gaussian_blur(self):
        """应用高斯模糊"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QSpinBox, QDoubleSpinBox, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建高斯模糊对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("高斯模糊")
        layout = QVBoxLayout()
        
        # 核大小设置
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("核大小 (奇数):"))
        
        kernel_spin = QSpinBox()
        kernel_spin.setRange(3, 15)
        kernel_spin.setSingleStep(2)
        kernel_spin.setValue(5)
        kernel_layout.addWidget(kernel_spin)
        layout.addLayout(kernel_layout)
        
        # Sigma值设置
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma值:"))
        
        sigma_spin = QDoubleSpinBox()
        sigma_spin.setRange(0.1, 10.0)
        sigma_spin.setSingleStep(0.1)
        sigma_spin.setValue(1.0)
        sigma_layout.addWidget(sigma_spin)
        layout.addLayout(sigma_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            ksize = kernel_spin.value()
            sigma = sigma_spin.value()
            
            # 生成高斯核
            def gaussian_kernel(size, sigma):
                """生成2D高斯核"""
                kernel = np.zeros((size, size))
                center = size // 2
                
                for i in range(size):
                    for j in range(size):
                        x, y = i - center, j - center
                        kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
                
                return kernel / np.sum(kernel)
            
            kernel = gaussian_kernel(ksize, sigma)
            
            # 应用高斯模糊
            def apply_convolution(image, kernel):
                """应用卷积运算"""
                h, w = image.shape
                kh, kw = kernel.shape
                pad = kh // 2
                
                # 添加边界填充
                padded = np.pad(image, pad, mode='reflect')
                
                # 执行卷积
                result = np.zeros_like(image)
                for i in range(h):
                    for j in range(w):
                        result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
                
                return np.clip(result, 0, 255).astype(np.uint8)
            
            # 处理图像
            if len(self.image_data.shape) == 3:  # 彩色图像
                # 对每个通道分别处理
                for c in range(3):
                    self.image_data[:,:,c] = apply_convolution(self.image_data[:,:,c], kernel)
            else:  # 灰度图像
                self.image_data = apply_convolution(self.image_data, kernel)
                
            # 更新显示
            self.update_image_views()
        
    def detect_edges(self):
        """边缘检测"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QSlider, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建边缘检测对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("边缘检测")
        layout = QVBoxLayout()
        
        # 阈值设置
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("边缘阈值:"))
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(50)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(25)
        
        value_label = QLabel("50")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            threshold = slider.value()
            
            # 确保图像是灰度图像
            if len(self.image_data.shape) == 3:
                self.convert_to_grayscale()
                
            # 定义Sobel算子
            sobel_x = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
            
            sobel_y = np.array([[-1, -2, -1],
                               [0,  0,  0],
                               [1,  2,  1]])
            
            # 应用卷积计算梯度
            def apply_convolution(image, kernel):
                """应用卷积运算"""
                h, w = image.shape
                kh, kw = kernel.shape
                pad = kh // 2
                
                # 添加边界填充
                padded = np.pad(image, pad, mode='reflect')
                
                # 执行卷积
                result = np.zeros_like(image)
                for i in range(h):
                    for j in range(w):
                        result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
                
                return result
            
            # 计算x和y方向的梯度
            grad_x = apply_convolution(self.image_data, sobel_x)
            grad_y = apply_convolution(self.image_data, sobel_y)
            
            # 计算梯度幅值
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 应用阈值
            self.image_data = np.where(grad_magnitude > threshold, 255, 0).astype(np.uint8)
            
            # 更新显示
            self.update_image_views()
        
    def rotate_image(self):
        """旋转图像"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QSpinBox, QComboBox, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建旋转对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("旋转图像")
        layout = QVBoxLayout()
        
        # 角度设置
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("旋转角度 (度):"))
        
        angle_spin = QSpinBox()
        angle_spin.setRange(-360, 360)
        angle_spin.setValue(90)
        angle_layout.addWidget(angle_spin)
        layout.addLayout(angle_layout)
        
        # 插值方法选择
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("插值方法:"))
        
        interp_combo = QComboBox()
        interp_combo.addItems(["最近邻", "双线性"])
        interp_layout.addWidget(interp_combo)
        layout.addLayout(interp_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            angle = angle_spin.value()
            interp_method = interp_combo.currentIndex()
            
            # 将角度转换为弧度
            theta = np.radians(angle)
            
            # 获取图像中心
            h, w = self.image_data.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # 创建旋转矩阵
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # 创建结果图像
            if len(self.image_data.shape) == 3:  # 彩色图像
                rotated = np.zeros_like(self.image_data)
            else:  # 灰度图像
                rotated = np.zeros_like(self.image_data)
            
            # 遍历每个像素并应用旋转
            for y in range(h):
                for x in range(w):
                    # 计算相对于中心的位置
                    x_centered = x - center_x
                    y_centered = y - center_y
                    
                    # 应用旋转
                    x_rot = int(cos_theta * x_centered - sin_theta * y_centered + center_x)
                    y_rot = int(sin_theta * x_centered + cos_theta * y_centered + center_y)
                    
                    # 检查旋转后的坐标是否在图像范围内
                    if 0 <= x_rot < w and 0 <= y_rot < h:
                        if interp_method == 0:  # 最近邻插值
                            rotated[y, x] = self.image_data[y_rot, x_rot]
                        else:  # 双线性插值
                            # 获取四个最近的像素
                            x1, y1 = int(np.floor(x_rot)), int(np.floor(y_rot))
                            x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                            
                            # 计算权重
                            x_weight = x_rot - x1
                            y_weight = y_rot - y1
                            
                            # 应用双线性插值
                            if len(self.image_data.shape) == 3:  # 彩色图像
                                for c in range(3):
                                    top = (1 - x_weight) * self.image_data[y1, x1, c] + x_weight * self.image_data[y1, x2, c]
                                    bottom = (1 - x_weight) * self.image_data[y2, x1, c] + x_weight * self.image_data[y2, x2, c]
                                    rotated[y, x, c] = (1 - y_weight) * top + y_weight * bottom
                            else:  # 灰度图像
                                top = (1 - x_weight) * self.image_data[y1, x1] + x_weight * self.image_data[y1, x2]
                                bottom = (1 - x_weight) * self.image_data[y2, x1] + x_weight * self.image_data[y2, x2]
                                rotated[y, x] = (1 - y_weight) * top + y_weight * bottom
            
            self.image_data = rotated.astype(np.uint8)
            self.update_image_views()
        
    def crop_image(self):
        """裁剪图像"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QPushButton, QGraphicsRectItem)
        from PyQt5.QtCore import Qt, QRectF, QPointF
        from PyQt5.QtGui import QPen, QColor
        
        # 创建裁剪对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("裁剪图像")
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout()
        
        # 创建预览区域
        preview_label = QLabel("在原图上拖动鼠标选择裁剪区域")
        layout.addWidget(preview_label)
        
        # 创建图形视图用于交互选择
        crop_view = QGraphicsView()
        crop_scene = QGraphicsScene()
        crop_view.setScene(crop_scene)
        
        # 显示原图
        if len(self.original_image.shape) == 2:  # 灰度图像
            height, width = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            q_img = QImage(img_bytes, width, height, width, QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        pixmap_item = crop_scene.addPixmap(pixmap)
        crop_view.fitInView(pixmap_item, Qt.KeepAspectRatio)
        
        # 添加选择矩形
        selection_rect = QGraphicsRectItem()
        selection_rect.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
        crop_scene.addItem(selection_rect)
        
        # 选择区域变量
        start_pos = None
        crop_rect = None
        
        def mouse_press(event):
            nonlocal start_pos
            start_pos = crop_view.mapToScene(event.pos())
            selection_rect.setRect(QRectF(start_pos, QSizeF(0, 0)))
            
        def mouse_move(event):
            if start_pos is None:
                return
            end_pos = crop_view.mapToScene(event.pos())
            selection_rect.setRect(QRectF(start_pos, end_pos).normalized())
            
        def mouse_release(event):
            nonlocal crop_rect
            if start_pos is None:
                return
            end_pos = crop_view.mapToScene(event.pos())
            crop_rect = QRectF(start_pos, end_pos).normalized()
            
        crop_view.mousePressEvent = mouse_press
        crop_view.mouseMoveEvent = mouse_move
        crop_view.mouseReleaseEvent = mouse_release
        
        layout.addWidget(crop_view)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted and crop_rect is not None:
            # 获取裁剪区域坐标
            x1 = int(max(0, crop_rect.left()))
            y1 = int(max(0, crop_rect.top()))
            x2 = int(min(self.image_data.shape[1], crop_rect.right()))
            y2 = int(min(self.image_data.shape[0], crop_rect.bottom()))
            
            # 执行裁剪
            if len(self.image_data.shape) == 3:  # 彩色图像
                self.image_data = self.image_data[y1:y2, x1:x2, :]
            else:  # 灰度图像
                self.image_data = self.image_data[y1:y2, x1:x2]
                
            # 更新显示
            self.update_image_views()
        
    def show_histogram(self):
        """显示直方图"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel)
        from PyQt5.QtCore import Qt
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        # 创建直方图对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("图像直方图")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout()
        
        # 创建matplotlib图形
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # 计算直方图
        if len(self.image_data.shape) == 3:  # 彩色图像
            colors = ('r', 'g', 'b')
            channel_names = ('Red', 'Green', 'Blue')
            for i, color in enumerate(colors):
                hist, bins = np.histogram(self.image_data[:,:,i], bins=256, range=(0, 256))
                ax.plot(bins[:-1], hist, color=color, label=channel_names[i])
        else:  # 灰度图像
            hist, bins = np.histogram(self.image_data, bins=256, range=(0, 256))
            ax.plot(bins[:-1], hist, color='black', label='Intensity')
            
        # 设置图表属性
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Image Histogram')
        ax.legend()
        ax.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 绘制图表
        canvas.draw()
        
        # 确定按钮
        ok_button = QPushButton("关闭")
        ok_button.clicked.connect(dialog.close)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def adjust_brightness(self):
        """调整亮度"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import QSlider, QDialog, QVBoxLayout, QLabel
        from PyQt5.QtCore import Qt
        
        # 创建亮度调整对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("调整亮度")
        layout = QVBoxLayout()
        
        # 创建滑块 (-100 到 +100)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(20)
        
        # 显示当前值的标签
        value_label = QLabel("亮度调整值: 0")
        
        # 滑块值改变时更新标签
        def update_label(value):
            value_label.setText(f"亮度调整值: {value}")
            
        slider.valueChanged.connect(update_label)
        
        # 添加控件到对话框
        layout.addWidget(QLabel("拖动滑块调整亮度:"))
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取调整值并应用
            value = slider.value()
            
            # 调整亮度 (value范围-100到100)
            if len(self.image_data.shape) == 3:  # 彩色图像
                self.image_data = np.clip(self.image_data.astype(np.int16) + value, 0, 255).astype(np.uint8)
            else:  # 灰度图像
                self.image_data = np.clip(self.image_data.astype(np.int16) + value, 0, 255).astype(np.uint8)
                
            # 更新显示
            self.update_image_views()
        
    def adjust_contrast(self):
        """调整对比度"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import QSlider, QDialog, QVBoxLayout, QLabel
        from PyQt5.QtCore import Qt
        
        # 创建对比度调整对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("调整对比度")
        layout = QVBoxLayout()
        
        # 创建滑块 (50% 到 150%)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(50, 150)
        slider.setValue(100)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        
        # 显示当前值的标签
        value_label = QLabel("对比度调整值: 100%")
        
        # 滑块值改变时更新标签
        def update_label(value):
            value_label.setText(f"对比度调整值: {value}%")
            
        slider.valueChanged.connect(update_label)
        
        # 添加控件到对话框
        layout.addWidget(QLabel("拖动滑块调整对比度:"))
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取调整值并应用
            contrast = slider.value() / 100.0  # 转换为比例
            
            # 调整对比度
            if len(self.image_data.shape) == 3:  # 彩色图像
                # 对每个通道分别处理
                for c in range(3):
                    self.image_data[:,:,c] = np.clip(contrast * (self.image_data[:,:,c].astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
            else:  # 灰度图像
                self.image_data = np.clip(contrast * (self.image_data.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
                
            # 更新显示
            self.update_image_views()
        
    def apply_threshold(self):
        """应用阈值处理"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                                   QLabel, QComboBox, QSlider, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建阈值处理对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("阈值处理")
        layout = QVBoxLayout()
        
        # 阈值类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("阈值类型:"))
        
        type_combo = QComboBox()
        type_combo.addItems(["二值化", "反二值化", "截断", "阈值化为0", "反阈值化为0"])
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # 阈值大小滑块
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("阈值大小:"))
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(127)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(25)
        
        value_label = QLabel("127")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            threshold_type = type_combo.currentIndex()
            threshold_value = slider.value()
            
            # 确保图像是灰度图像
            if len(self.image_data.shape) == 3:
                self.convert_to_grayscale()
                
            # 应用阈值处理
            if threshold_type == 0:  # 二值化
                self.image_data = np.where(self.image_data > threshold_value, 255, 0)
            elif threshold_type == 1:  # 反二值化
                self.image_data = np.where(self.image_data > threshold_value, 0, 255)
            elif threshold_type == 2:  # 截断
                self.image_data = np.where(self.image_data > threshold_value, threshold_value, self.image_data)
            elif threshold_type == 3:  # 阈值化为0
                self.image_data = np.where(self.image_data > threshold_value, self.image_data, 0)
            else:  # 反阈值化为0
                self.image_data = np.where(self.image_data > threshold_value, 0, self.image_data)
                
            # 更新显示
            self.update_image_views()
        
    def apply_morphology(self):
        """应用形态学变换"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QComboBox, QSpinBox, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建形态学变换对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("形态学变换")
        layout = QVBoxLayout()
        
        # 操作类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("操作类型:"))
        
        type_combo = QComboBox()
        type_combo.addItems(["膨胀", "腐蚀", "开运算", "闭运算"])
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # 结构元素大小设置
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("结构元素大小 (奇数):"))
        
        size_spin = QSpinBox()
        size_spin.setRange(3, 15)
        size_spin.setSingleStep(2)
        size_spin.setValue(3)
        size_layout.addWidget(size_spin)
        layout.addLayout(size_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            operation = type_combo.currentIndex()
            ksize = size_spin.value()
            
            # 确保图像是灰度图像
            if len(self.image_data.shape) == 3:
                self.convert_to_grayscale()
                
            # 创建结构元素
            kernel = np.ones((ksize, ksize), np.uint8)
            
            # 应用形态学操作
            if operation == 0:  # 膨胀
                def dilate(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    
                    # 添加边界填充
                    padded = np.pad(image, pad, mode='constant', constant_values=0)
                    
                    # 执行膨胀
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.max(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                self.image_data = dilate(self.image_data, kernel)
                
            elif operation == 1:  # 腐蚀
                def erode(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    
                    # 添加边界填充
                    padded = np.pad(image, pad, mode='constant', constant_values=255)
                    
                    # 执行腐蚀
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.min(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                self.image_data = erode(self.image_data, kernel)
                
            elif operation == 2:  # 开运算 (先腐蚀后膨胀)
                def erode(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    padded = np.pad(image, pad, mode='constant', constant_values=255)
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.min(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                def dilate(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    padded = np.pad(image, pad, mode='constant', constant_values=0)
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.max(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                temp = erode(self.image_data, kernel)
                self.image_data = dilate(temp, kernel)
                
            else:  # 闭运算 (先膨胀后腐蚀)
                def dilate(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    padded = np.pad(image, pad, mode='constant', constant_values=0)
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.max(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                def erode(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad = kh // 2
                    padded = np.pad(image, pad, mode='constant', constant_values=255)
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.min(padded[i:i+kh, j:j+kw] * kernel)
                    return result
                
                temp = dilate(self.image_data, kernel)
                self.image_data = erode(temp, kernel)
                
            # 更新显示
            self.update_image_views()
        
    def flip_image(self):
        """翻转图像"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import QInputDialog
        
        # 让用户选择翻转方式
        items = ["水平翻转", "垂直翻转"]
        item, ok = QInputDialog.getItem(self, "选择翻转方式", 
                                      "请选择翻转方向:", items, 0, False)
        if not ok:
            return
            
        # 执行翻转
        if item == "水平翻转":
            self.image_data = np.fliplr(self.image_data)
        else:  # 垂直翻转
            self.image_data = np.flipud(self.image_data)
            
        # 更新显示
        self.update_image_views()
        
    def segment_image(self):
        """图像分割"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QSpinBox, QPushButton, 
                                    QGraphicsView, QGraphicsScene)
        from PyQt5.QtCore import Qt, QPointF
        from PyQt5.QtGui import QPen, QColor
        
        # 创建图像分割对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("图像分割")
        dialog.setMinimumSize(600, 500)
        layout = QVBoxLayout()
        
        # 创建预览区域
        preview_label = QLabel("在原图上点击选择种子点")
        layout.addWidget(preview_label)
        
        # 创建图形视图用于交互选择
        seg_view = QGraphicsView()
        seg_scene = QGraphicsScene()
        seg_view.setScene(seg_scene)
        
        # 显示原图
        if len(self.original_image.shape) == 2:  # 灰度图像
            height, width = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            q_img = QImage(img_bytes, width, height, width, QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        pixmap_item = seg_scene.addPixmap(pixmap)
        seg_view.fitInView(pixmap_item, Qt.KeepAspectRatio)
        
        # 阈值设置
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("相似度阈值:"))
        
        threshold_spin = QSpinBox()
        threshold_spin.setRange(1, 100)
        threshold_spin.setValue(20)
        threshold_layout.addWidget(threshold_spin)
        layout.addLayout(threshold_layout)
        
        # 种子点变量
        seed_point = None
        
        def mouse_press(event):
            nonlocal seed_point
            pos = seg_view.mapToScene(event.pos())
            x = int(pos.x())
            y = int(pos.y())
            
            # 确保点在图像范围内
            if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
                seed_point = (y, x)  # numpy数组是(y,x)顺序
                
                # 清除之前的标记
                seg_scene.clear()
                seg_scene.addPixmap(pixmap)
                
                # 添加新标记
                pen = QPen(QColor(255, 0, 0))
                pen.setWidth(3)
                seg_scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, pen)
        
        seg_view.mousePressEvent = mouse_press
        layout.addWidget(seg_view)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted and seed_point is not None:
            # 获取参数
            threshold = threshold_spin.value()
            
            # 确保图像是灰度图像
            if len(self.image_data.shape) == 3:
                self.convert_to_grayscale()
                
            # 区域生长算法
            def region_growing(image, seed, threshold):
                """基于区域生长的图像分割"""
                h, w = image.shape
                segmented = np.zeros_like(image)
                
                # 获取种子点像素值
                seed_value = image[seed[0], seed[1]]
                
                # 创建待处理队列
                queue = [seed]
                segmented[seed[0], seed[1]] = 255
                
                # 4邻域偏移量
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                
                while queue:
                    current = queue.pop(0)
                    
                    for dy, dx in neighbors:
                        y = current[0] + dy
                        x = current[1] + dx
                        
                        # 检查边界
                        if 0 <= y < h and 0 <= x < w:
                            # 检查是否已处理
                            if segmented[y, x] == 0:
                                # 检查像素值差异
                                if abs(int(image[y, x]) - int(seed_value)) <= threshold:
                                    segmented[y, x] = 255
                                    queue.append((y, x))
                
                return segmented
            
            # 应用区域生长
            self.image_data = region_growing(self.image_data, seed_point, threshold)
            
            # 更新显示
            self.update_image_views()
        
    def apply_filter(self):
        """应用图像滤波"""
        if self.image_data is None:
            return
            
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QComboBox, QSpinBox, QPushButton)
        from PyQt5.QtCore import Qt
        
        # 创建滤波对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("图像滤波")
        layout = QVBoxLayout()
        
        # 滤波类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("滤波类型:"))
        
        type_combo = QComboBox()
        type_combo.addItems(["均值滤波", "中值滤波"])
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # 核大小设置
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("核大小 (奇数):"))
        
        size_spin = QSpinBox()
        size_spin.setRange(3, 15)
        size_spin.setSingleStep(2)
        size_spin.setValue(3)
        size_layout.addWidget(size_spin)
        layout.addLayout(size_layout)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            # 获取参数
            filter_type = type_combo.currentIndex()
            ksize = size_spin.value()
            
            # 应用滤波
            if filter_type == 0:  # 均值滤波
                def mean_filter(image, ksize):
                    """应用均值滤波"""
                    h, w = image.shape
                    pad = ksize // 2
                    
                    # 添加边界填充
                    padded = np.pad(image, pad, mode='reflect')
                    
                    # 执行滤波
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.mean(padded[i:i+ksize, j:j+ksize])
                    return result.astype(np.uint8)
                
                if len(self.image_data.shape) == 3:  # 彩色图像
                    # 对每个通道分别处理
                    for c in range(3):
                        self.image_data[:,:,c] = mean_filter(self.image_data[:,:,c], ksize)
                else:  # 灰度图像
                    self.image_data = mean_filter(self.image_data, ksize)
                    
            else:  # 中值滤波
                def median_filter(image, ksize):
                    """应用中值滤波"""
                    h, w = image.shape
                    pad = ksize // 2
                    
                    # 添加边界填充
                    padded = np.pad(image, pad, mode='reflect')
                    
                    # 执行滤波
                    result = np.zeros_like(image)
                    for i in range(h):
                        for j in range(w):
                            result[i, j] = np.median(padded[i:i+ksize, j:j+ksize])
                    return result.astype(np.uint8)
                
                if len(self.image_data.shape) == 3:  # 彩色图像
                    # 对每个通道分别处理
                    for c in range(3):
                        self.image_data[:,:,c] = median_filter(self.image_data[:,:,c], ksize)
                else:  # 灰度图像
                    self.image_data = median_filter(self.image_data, ksize)
            
            # 更新显示
            self.update_image_views()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
