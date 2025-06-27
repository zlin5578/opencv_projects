import numpy as np
import cv2


def compute_vanishing_point(lines, img_width, img_height):
    """
    计算消失点（车道线的交汇点）

    Args:
        lines: HoughLinesP 检测到的线段
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        vp_x, vp_y: 消失点坐标
    """
    if lines is None or len(lines) == 0:
        return img_width // 2, int(img_height * 0.6)

    # 收集所有有效的交点
    intersections = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]

            # 计算两条线的斜率
            denom1 = x2 - x1
            denom2 = x4 - x3

            if abs(denom1) < 1e-5 or abs(denom2) < 1e-5:
                continue

            slope1 = (y2 - y1) / denom1
            slope2 = (y4 - y3) / denom2

            # 过滤平行线和斜率相近的线
            if abs(slope1 - slope2) < 0.1:
                continue

            # 计算交点
            # 直线方程: y - y1 = slope1 * (x - x1)
            # 直线方程: y - y3 = slope2 * (x - x3)
            denom = slope1 - slope2
            if abs(denom) < 1e-5:
                continue

            x = (slope1 * x1 - slope2 * x3 + y3 - y1) / denom
            y = slope1 * (x - x1) + y1

            # 过滤不合理的交点
            if 0 <= x <= img_width and 0 <= y <= img_height * 0.8:
                intersections.append((x, y))

    if not intersections:
        return img_width // 2, int(img_height * 0.6)

    # 使用中位数来获得稳定的消失点
    intersections = np.array(intersections)
    vp_x = int(np.median(intersections[:, 0]))
    vp_y = int(np.median(intersections[:, 1]))

    # 确保消失点在合理范围内
    vp_x = max(img_width // 4, min(3 * img_width // 4, vp_x))
    vp_y = max(img_height // 3, min(2 * img_height // 3, vp_y))

    return vp_x, vp_y


def get_trapezoid_roi(image, top_margin_ratio=0.1, bottom_margin_ratio=0.15, debug=False):
    """
    自适应梯形ROI，通过检测消失点动态调整形状

    Args:
        image: 输入图像（可以是原图或边缘图）
        top_margin_ratio: 消失点左右扩展的比例
        bottom_margin_ratio: 底部左右缩进的比例
        debug: 是否保存中间结果图像

    Returns:
        masked_image: 应用ROI后的图像
        roi: ROI多边形点
    """
    h, w = image.shape[:2]

    # 如果输入是彩色图像，先转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
    else:
        # 假设输入已经是边缘图
        edges = image

    # 步骤1: 在下半部分图像中检测线段
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(coarse_mask, (0, h // 2), (w, h), 255, -1)
    edges_coarse = cv2.bitwise_and(edges, coarse_mask)

    # 步骤2: 使用HoughLinesP检测线段
    lines = cv2.HoughLinesP(
        edges_coarse,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )

    # 步骤3: 计算消失点
    vp_x, vp_y = compute_vanishing_point(lines, w, h)

    # 步骤4: 构造梯形ROI
    top_margin = int(w * top_margin_ratio)

    # 四个角点
    left_top = [max(0, vp_x - top_margin), vp_y]
    right_top = [min(w, vp_x + top_margin), vp_y]
    left_down = [int(w * bottom_margin_ratio), h]
    right_down = [int(w * (1 - bottom_margin_ratio)), h]

    roi_points = np.array([
        left_down,
        left_top,
        right_top,
        right_down
    ], dtype=np.int32)

    # 步骤5: 创建掩码并应用
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi_points, 255)

    # 应用掩码
    if len(image.shape) == 3:
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_image = cv2.bitwise_and(image, mask_3channel)
    else:
        masked_image = cv2.bitwise_and(image, mask)


    return masked_image, roi_points


def get_simple_trapezoid_roi(image, top_width_ratio=0.1, top_height_ratio=0.55, bottom_width_ratio=0.8):
    """
    简单的固定梯形ROI（当自适应方法失败时的后备方案）
    """
    height, width = image.shape[:2]

    top_width = int(width * top_width_ratio)
    bottom_width = int(width * bottom_width_ratio)
    top_y = int(height * top_height_ratio)

    roi = np.array([
        [(width - bottom_width) // 2, height],  # Bottom-left
        [(width - top_width) // 2, top_y],  # Top-left
        [(width + top_width) // 2, top_y],  # Top-right
        [(width + bottom_width) // 2, height]  # Bottom-right
    ], dtype=np.int32)

    mask = np.zeros_like(image)
    if len(image.shape) == 2:
        cv2.fillPoly(mask, [roi], 255)
    else:
        cv2.fillPoly(mask, [roi], (255, 255, 255))

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, roi


# 用于保持与原代码兼容的包装函数
def get_trapezoid_roi_with_fallback(image, adaptive=True, **kwargs):
    """
    带有后备机制的ROI函数

    Args:
        image: 输入图像
        adaptive: 是否使用自适应方法
        **kwargs: 传递给具体ROI函数的参数
    """
    if adaptive:
        try:
            return get_trapezoid_roi(image, **kwargs)
        except Exception as e:
            print(f"Adaptive ROI failed: {e}, falling back to simple ROI")
            return get_simple_trapezoid_roi(image)
    else:
        return get_simple_trapezoid_roi(image, **kwargs)