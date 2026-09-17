# import streamlit as st
# import cv2
# import numpy as np
# import pydicom
# import plistlib
# from PIL import Image
# import requests
# import base64
# import json

# # ==============================================================================
# # 0. CẤU HÌNH
# # ==============================================================================
# # Thay bằng URL thật của bạn
# # MODAL_API_URL = "https://teddy2003--mammogram-ensemble-backend-api-inference.modal.run"
# MODAL_API_URL = "https://agkbskwbhs--mammogram-ensemble-backend-api-inference.modal.run"
# NORM_MEAN = [0.2512, 0.2741, 0.1900]
# NORM_STD = [0.2775, 0.2605, 0.2203]
# # ==============================================================================
# # 1. CÁC HÀM XỬ LÝ ẢNH (ĐƯA VỀ FRONTEND ĐỂ ĐỒNG BỘ MASK)
# # ==============================================================================

# def read_image_data(uploaded_file):
#     if uploaded_file.name.lower().endswith('.dcm'):
#         dcm_data = pydicom.dcmread(uploaded_file)
#         pixel_array = dcm_data.pixel_array.astype(np.float32)
#         pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)
#         return pixel_array
#     else:
#         return np.array(Image.open(uploaded_file).convert("L"))

# def create_mask_from_xml(xml_file, img_shape):
#     try:
#         plist = plistlib.load(xml_file)
#         images = plist["Images"]
#         image0 = images[0]
#         rois = image0["ROIs"]
#         h, w = img_shape
#         mask = np.zeros((h, w), dtype=np.uint8)
#         mass_rois = [roi for roi in rois if roi.get("Name") == "Mass"]
#         if not mass_rois: return mask, False
#         for roi in mass_rois:
#             point_px_list = roi.get("Point_px", [])
#             points = []
#             for s in point_px_list:
#                 s = s.strip("()")
#                 x_str, y_str, *_ = s.split(",")
#                 points.append([float(x_str), float(y_str)])
#             points = np.array(points, dtype=np.float32)
#             if points.shape[0] >= 3:
#                 pts_int = points.reshape((-1, 1, 2)).astype(np.int32)
#                 cv2.fillPoly(mask, [pts_int], 1)
#         return mask, True
#     except: return np.zeros(img_shape, dtype=np.uint8), False

# def smart_crop_and_square(img, mask=None):
#     h, w = img.shape
#     _, bin_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
#     coords = cv2.findNonZero(bin_img)
#     if coords is None: return img, mask
    
#     x, y, w_box, h_box = cv2.boundingRect(coords)
#     crop_img = img[y:y+h_box, x:x+w_box]
    
#     crop_mask = None
#     if mask is not None:
#         crop_mask = mask[y:y+h_box, x:x+w_box] # Cắt mask y hệt cắt ảnh
    
#     h_c, w_c = crop_img.shape[:2]
#     square_size = max(h_c, w_c)
    
#     final_img = np.zeros((square_size, square_size), dtype=np.uint8)
#     final_mask = np.zeros((square_size, square_size), dtype=np.uint8) if mask is not None else None
    
#     left_sum = np.sum(crop_img[:, :w_c//2])
#     right_sum = np.sum(crop_img[:, w_c//2:])
#     y_pos = 0
#     x_pos = 0 if left_sum > right_sum else square_size - w_c
        
#     final_img[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_img
#     if mask is not None:
#         final_mask[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_mask
        
#     return final_img, final_mask

# def generate_3_channels(img_gray):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
#     ch2 = clahe.apply(img_gray)
#     table = np.array([((i / 255.0) ** 1.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     ch3 = cv2.LUT(ch2, table)
#     return cv2.merge((img_gray, ch2, ch3))

# def resize_final(img_3ch, mask=None, target_size=(640, 640)):
#     img_resized = cv2.resize(img_3ch, target_size, interpolation=cv2.INTER_AREA)
#     mask_binary = None
#     if mask is not None:
#         mask_temp = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
#         _, mask_binary = cv2.threshold(mask_temp, 0.5, 1, cv2.THRESH_BINARY)
#     return img_resized, mask_binary

# def remove_small_objects(pred_mask_np, min_size=100):
#     mask_uint8 = pred_mask_np.astype(np.uint8)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
#     new_mask = np.zeros_like(mask_uint8)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] >= min_size:
#             new_mask[labels == i] = 1
#     return new_mask

# def calculate_metrics(gt, pred):
#     eps = 1e-6
#     intersection = (gt * pred).sum()
#     dice = (2. * intersection + eps) / (gt.sum() + pred.sum() + eps)
#     union = gt.sum() + pred.sum() - intersection
#     iou = (intersection + eps) / (union + eps)
#     return dice, iou

# def create_result_overlay(bg_img, gt, pred):
#     if len(bg_img.shape) == 2: bg = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2RGB)
#     else: bg = bg_img.copy()
#     overlay = bg.copy()
    
#     if gt is None:
#         overlay[pred==1] = [0, 0, 255] # Red
#     else:
#         overlay[np.logical_and(gt==1, pred==1)] = [255, 215, 0] # TP: Vàng
#         overlay[np.logical_and(gt==1, pred==0)] = [0, 255, 0]   # FN: Xanh lá
#         overlay[np.logical_and(gt==0, pred==1)] = [255, 0, 0]   # FP: Đỏ
#     return cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)

# # ==============================================================================
# # 3. GIAO DIỆN CHÍNH
# # ==============================================================================
# st.set_page_config(page_title="End-to-End Mammogram Detection", layout="wide")

# st.markdown("""
#     <style>
#         /* 1. Ẩn Header mặc định (giữ lại nút Deploy/Menu) */
#         header[data-testid="stHeader"] {
#             background-color: transparent !important;
#             z-index: 99999; /* Đảm bảo nút Deploy nằm trên cùng để bấm được */
#         }
        
#         /* 2. Điều chỉnh container chính */
#         .block-container {
#             padding-top: 5rem !important; 
#         }

#         /* 3. Tạo Header cố định */
#         .fixed-header {
#             position: fixed;
#             top: 0;
#             left: 14;
#             width: 100%;
#             height: 3.75rem;
#             background-color: white;
#             z-index: 99990; /* Thấp hơn nút Deploy một chút */
            
#             display: flex;
#             align-items: center;
            
#             /* --- SỬA Ở ĐÂY ĐỂ DỜI CHỮ QUA PHẢI --- */
#             justify_content: flex-end; /* Đẩy toàn bộ nội dung sang phía bên phải */
#             padding-right: 14rem;      /* Tạo khoảng cách 14rem (khoảng 220px) tính từ lề phải vào */
#                                        /* Con số này chừa chỗ cho nút Deploy và Menu */
            
#             border-bottom: 1px solid #f0f0f0;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#         }
        
#         /* Style chữ */
#         .header-text {
#             font-family: 'Source Sans Pro', sans-serif;
#             font-weight: 700;
#             font-size: 2rem;
#             color: #31333F;
#             margin: 0;
#             padding-top: 0.2rem;
#             white-space: nowrap; /* Đảm bảo chữ không bị xuống dòng nếu màn hình nhỏ */
#         }

#         /* Sidebar */
#         section[data-testid="stSidebar"] {
#             top: 0rem !important;
#             height: 100vh !important;
#         }
#     </style>

#     <div class="fixed-header">
#         <h1 class="header-text"> Hệ thống Chuẩn đoán Khối u Tuyến vú</h1>
#     </div>
# """, unsafe_allow_html=True)

# with st.sidebar:
#     st.header("1. Cấu hình")
#     st.success("Backend: Modal.com (GPU T4)")
#     st.header("2. Hậu xử lý")
#     pixel_thresh = st.slider("Pixel Threshold", 0.1, 0.9, 0.3, 0.05)
#     min_area = st.slider("Min Area Size (px)", 0, 500, 100, 10)
#     st.header("3. Hiển thị")
#     view_mode = st.radio("Kênh ảnh:", ("Grayscale", "CLAHE", "Combo", "Merged"))

# col_up1, col_up2 = st.columns(2)
# with col_up1: img_file = st.file_uploader("1. Ảnh Mammogram", type=["dcm", "png", "jpg"])
# with col_up2: mask_file = st.file_uploader("2. Mask (Optional)", type=["xml", "dcm", "png"])

# if img_file:
#     file_id = f"{img_file.name}_{mask_file.name if mask_file else 'None'}"
    
#     if 'curr_id' not in st.session_state or st.session_state.curr_id != file_id:
#         with st.spinner("🚀 Đang xử lý ảnh & gửi sang Cloud GPU..."):
#             try:
#                 # 1. Đọc Ảnh & Mask
#                 original_img = read_image_data(img_file)
#                 h_orig, w_orig = original_img.shape
                
#                 original_mask = None
#                 if mask_file:
#                     if mask_file.name.endswith('.xml'):
#                         original_mask, _ = create_mask_from_xml(mask_file, (h_orig, w_orig))
#                     elif mask_file.name.endswith('.dcm'):
#                         d = pydicom.dcmread(mask_file)
#                         original_mask = (d.pixel_array > 127).astype(np.uint8)
#                     else:
#                         original_mask = np.array(Image.open(mask_file).convert("L"))
#                         original_mask = (original_mask > 127).astype(np.uint8)
#                         # Resize thô nếu mask png lệch size
#                         if original_mask.shape != (h_orig, w_orig):
#                             original_mask = cv2.resize(original_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

#                 # 2. Preprocessing ĐỒNG BỘ (Quan trọng nhất)
#                 # Cắt cả ảnh và mask cùng 1 tọa độ
#                 square_img, square_mask = smart_crop_and_square(original_img, original_mask)
#                 img_3ch = generate_3_channels(square_img)
#                 # Resize cả 2 về 640x640
#                 input_img, input_mask = resize_final(img_3ch, square_mask, target_size=(640, 640))
                
#                 # 3. Gửi ảnh đã xử lý (640x640) sang Modal
#                 _, buffer = cv2.imencode('.png', input_img)
#                 img_b64 = base64.b64encode(buffer).decode('utf-8')
                
#                 payload = {"image": img_b64}
#                 resp = requests.post(MODAL_API_URL, json=payload, timeout=300)
                
#                 if resp.status_code == 200:
#                     data = resp.json()
#                     probs = np.array(data["probs"], dtype=np.float32)
                    
#                     st.session_state.probs = probs
#                     st.session_state.input_img = input_img # Ảnh 3 kênh
#                     st.session_state.input_mask = input_mask # Mask đã crop khớp
#                     st.session_state.original_shape = (h_orig, w_orig)
#                     st.session_state.squared_shape = square_img.shape
#                     st.session_state.curr_id = file_id
#                 else:
#                     st.error(f"Lỗi Server: {resp.text}"); st.stop()
#             except Exception as e:
#                 st.error(f"Lỗi: {e}"); st.stop()

#     # --- HIỂN THỊ ---
#     if 'probs' in st.session_state:
#         probs = st.session_state.probs
#         input_img = st.session_state.input_img
#         input_mask = st.session_state.input_mask
        
#         raw_pred = (probs > pixel_thresh).astype(np.uint8)
#         clean_pred = remove_small_objects(raw_pred, min_size=min_area)
        
#         if input_mask is not None:
#             dice, iou = calculate_metrics(input_mask, clean_pred)
#             st.markdown(f"### 📊 Dice: **{dice:.4f}** | IoU: **{iou:.4f}**")
#             c1, c2, c3 = st.columns(3)
            
#             # Logic hiển thị màu/xám
#             disp_img = input_img
#             if view_mode == "Grayscale": disp_img = input_img[:,:,0]
#             elif view_mode == "CLAHE": disp_img = input_img[:,:,1]
#             elif view_mode == "Combo": disp_img = input_img[:,:,2]
            
#             with c1: st.image(disp_img, caption=f"Input ({view_mode})", use_container_width=True)
#             with c2: st.image(input_mask*255, caption="Ground Truth (Đã Crop)", use_container_width=True)
#             with c3: 
#                 ov = create_result_overlay(disp_img, input_mask, clean_pred)
#                 st.image(ov, caption="So sánh", use_container_width=True)
#             # Chú thích màu
#             st.info("""
#                 **Giải thích màu Overlay:**
#                 - 🟡 **Vàng (TP):** Model dự đoán đúng.
#                 - 🟢 **Xanh lá (FN):** Vùng khối u thực tế bị model bỏ sót.
#                 - 🔴 **Đỏ (FP):** Vùng model dự đoán sai (Dương tính giả).
#                 """)
#         else:
#             st.info("Chế độ Dự đoán")
#             c1, c2 = st.columns(2)
#             disp_img = input_img
#             if view_mode == "Grayscale": disp_img = input_img[:,:,0]
#             # ... (Tương tự cho các mode khác nếu muốn)
#             with c1: st.image(disp_img, caption="Input", use_container_width=True)
#             with c2: 
#                 ov = create_result_overlay(disp_img, None, clean_pred)
#                 st.image(ov, caption="Dự đoán", use_container_width=True)
#         # --- PHẦN BỔ SUNG: EXPANDER CHI TIẾT ---
#         with st.expander("🔎 Chi tiết quy trình xử lý dữ liệu"):
#             st.markdown("#### 1. Thông tin ảnh đầu vào")
#             st.write(f"- **Kích thước gốc:** {st.session_state.original_shape}")
#             st.write(f"- **Kích thước sau khi cắt vuông (Smart Crop):** {st.session_state.squared_shape}")
#             st.write(f"- **Kích thước đầu vào Model (Resize):** {input_img.shape} (640x640)")
            
#             st.markdown("#### 2. Thông số Tiền xử lý (Preprocessing)")
#             st.write("- **CLAHE:** Clip Limit = 2.0, Tile Grid = (24, 24)")
#             st.write("- **Gamma Correction:** Gamma = 1.5 (Làm tối nền)")
#             st.write(f"- **Normalization Mean:** `{NORM_MEAN}`")
#             st.write(f"- **Normalization Std:** `{NORM_STD}`")
            
#             st.markdown("#### 3. Cấu trúc Tensor")
#             st.code(f"""
#             Input Tensor Shape: (1, 3, 640, 640)
#             - Channel 0: Grayscale Original
#             - Channel 1: CLAHE Enhanced
#             - Channel 2: Combo (Gamma + CLAHE)
#             """, language="text")
# __________________________________________________________________________________________________________
import streamlit as st
import cv2
import numpy as np
import pydicom
import plistlib
from PIL import Image
import requests
import base64
import json

# ==============================================================================
# 0. CẤU HÌNH
# ==============================================================================
# Thay bằng URL thật của bạn
# MODAL_API_URL = "https://teddy2003--mammogram-ensemble-backend-api-inference.modal.run"
MODAL_API_URL = "https://agkbskwbhs--mammogram-ensemble-backend-api-inference.modal.run"
NORM_MEAN = [0.2512, 0.2741, 0.1900]
NORM_STD = [0.2775, 0.2605, 0.2203]

# --- CẤU HÌNH BỎ BIÊN & LỌC NHIỄU (port từ "bỏ biên và nhiễu.py") ---
# Đơn vị: pixel, tính trên ảnh gốc. Có thể chỉnh lại trong sidebar.
DEFAULT_CUT_BORDER = 100
DILATE_PIXELS = 10

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ ẢNH (ĐƯA VỀ FRONTEND ĐỂ ĐỒNG BỘ MASK)
# ==============================================================================

def read_image_data(uploaded_file):
    if uploaded_file.name.lower().endswith('.dcm'):
        dcm_data = pydicom.dcmread(uploaded_file)
        pixel_array = dcm_data.pixel_array.astype(np.float32)
        pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)
        return pixel_array
    else:
        return np.array(Image.open(uploaded_file).convert("L"))

def create_mask_from_xml(xml_file, img_shape):
    try:
        plist = plistlib.load(xml_file)
        images = plist["Images"]
        image0 = images[0]
        rois = image0["ROIs"]
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mass_rois = [roi for roi in rois if roi.get("Name") == "Mass"]
        if not mass_rois: return mask, False
        for roi in mass_rois:
            point_px_list = roi.get("Point_px", [])
            points = []
            for s in point_px_list:
                s = s.strip("()")
                x_str, y_str, *_ = s.split(",")
                points.append([float(x_str), float(y_str)])
            points = np.array(points, dtype=np.float32)
            if points.shape[0] >= 3:
                pts_int = points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts_int], 1)
        return mask, True
    except: return np.zeros(img_shape, dtype=np.uint8), False

# ------------------------------------------------------------------
# [MỚI] BỎ BIÊN CỐ ĐỊNH + LỌC NHIỄU SCANNER
# Port từ process_fixed_border_and_noise() trong "bỏ biên và nhiễu.py"
# Hàm này KHÔNG đổi kích thước ảnh -> mask Ground Truth vẫn khớp tọa độ.
# ------------------------------------------------------------------
def remove_border_and_noise(img_gray, cut_top=100, cut_bottom=100, cut_left=100,
                            cut_right=100, dilate_pixels=DILATE_PIXELS, gt_mask=None):
    """
    Trả về: (ảnh xám đã làm sạch, mask GT đã làm sạch hoặc None, mask vùng vú, cờ OK)
    """
    h, w = img_gray.shape
    gray = img_gray.copy()

    # --- BƯỚC 1: TÔ ĐEN BIÊN CỐ ĐỊNH (FIXED BLACKOUT) ---
    ct, cb = int(min(cut_top, h)), int(min(cut_bottom, h))
    cl, cr = int(min(cut_left, w)), int(min(cut_right, w))
    if ct > 0: gray[0:ct, :] = 0
    if cb > 0: gray[h - cb:h, :] = 0
    if cl > 0: gray[:, 0:cl] = 0
    if cr > 0: gray[:, w - cr:w] = 0

    # --- BƯỚC 2: LỌC NHIỄU (MORPHOLOGY) ---
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Giữ contour lớn nhất (vùng vú), bỏ rác còn sót
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray, gt_mask, np.zeros_like(img_gray), False

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1000:   # ảnh lỗi
        return img_gray, gt_mask, np.zeros_like(img_gray), False

    mask_processed = np.zeros_like(gray)
    cv2.drawContours(mask_processed, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Nới rộng mask để không mất rìa da
    dilate_size = dilate_pixels + 5
    kernel_dilate = np.ones((dilate_size * 2 + 1, dilate_size * 2 + 1), np.uint8)
    mask_dilated = cv2.dilate(mask_processed, kernel_dilate, iterations=1)

    # --- BƯỚC 3: ÁP DỤNG LÊN ẢNH GỐC (và mask GT nếu có) ---
    cleaned_img = cv2.bitwise_and(img_gray, img_gray, mask=mask_dilated)

    cleaned_gt = None
    if gt_mask is not None:
        cleaned_gt = cv2.bitwise_and(gt_mask, gt_mask, mask=mask_dilated)

    return cleaned_img, cleaned_gt, mask_dilated, True

def smart_crop_and_square(img, mask=None):
    h, w = img.shape
    _, bin_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(bin_img)
    if coords is None: return img, mask

    x, y, w_box, h_box = cv2.boundingRect(coords)
    crop_img = img[y:y+h_box, x:x+w_box]

    crop_mask = None
    if mask is not None:
        crop_mask = mask[y:y+h_box, x:x+w_box] # Cắt mask y hệt cắt ảnh

    h_c, w_c = crop_img.shape[:2]
    square_size = max(h_c, w_c)

    final_img = np.zeros((square_size, square_size), dtype=np.uint8)
    final_mask = np.zeros((square_size, square_size), dtype=np.uint8) if mask is not None else None

    left_sum = np.sum(crop_img[:, :w_c//2])
    right_sum = np.sum(crop_img[:, w_c//2:])
    y_pos = 0
    x_pos = 0 if left_sum > right_sum else square_size - w_c

    final_img[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_img
    if mask is not None:
        final_mask[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_mask

    return final_img, final_mask

def generate_3_channels(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
    ch2 = clahe.apply(img_gray)
    table = np.array([((i / 255.0) ** 1.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
    ch3 = cv2.LUT(ch2, table)
    return cv2.merge((img_gray, ch2, ch3))

def resize_final(img_3ch, mask=None, target_size=(640, 640)):
    img_resized = cv2.resize(img_3ch, target_size, interpolation=cv2.INTER_AREA)
    mask_binary = None
    if mask is not None:
        mask_temp = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        _, mask_binary = cv2.threshold(mask_temp, 0.5, 1, cv2.THRESH_BINARY)
    return img_resized, mask_binary

def remove_small_objects(pred_mask_np, min_size=100):
    mask_uint8 = pred_mask_np.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    new_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1
    return new_mask

def calculate_metrics(gt, pred):
    eps = 1e-6
    intersection = (gt * pred).sum()
    dice = (2. * intersection + eps) / (gt.sum() + pred.sum() + eps)
    union = gt.sum() + pred.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return dice, iou

def create_result_overlay(bg_img, gt, pred):
    if len(bg_img.shape) == 2: bg = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2RGB)
    else: bg = bg_img.copy()
    overlay = bg.copy()

    if gt is None:
        overlay[pred==1] = [255, 0, 0] # Đỏ (st.image hiểu theo thứ tự RGB)
    else:
        overlay[np.logical_and(gt==1, pred==1)] = [255, 215, 0] # TP: Vàng
        overlay[np.logical_and(gt==1, pred==0)] = [0, 255, 0]   # FN: Xanh lá
        overlay[np.logical_and(gt==0, pred==1)] = [255, 0, 0]   # FP: Đỏ
    return cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
st.set_page_config(page_title="End-to-End Mammogram Detection", layout="wide")

st.markdown("""
    <style>
        /* 1. Ẩn Header mặc định (giữ lại nút Deploy/Menu) */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99999; /* Đảm bảo nút Deploy nằm trên cùng để bấm được */
        }

        /* 2. Điều chỉnh container chính */
        .block-container {
            padding-top: 5rem !important;
        }

        /* 3. Tạo Header cố định */
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3.75rem;
            background-color: white;
            z-index: 99990; /* Thấp hơn nút Deploy một chút */

            display: flex;
            align-items: center;

            /* --- SỬA Ở ĐÂY ĐỂ DỜI CHỮ QUA PHẢI --- */
            justify-content: flex-end; /* Đẩy toàn bộ nội dung sang phía bên phải */
            padding-right: 14rem;      /* Tạo khoảng cách 14rem (khoảng 220px) tính từ lề phải vào */
                                       /* Con số này chừa chỗ cho nút Deploy và Menu */

            border-bottom: 1px solid #f0f0f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Style chữ */
        .header-text {
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            color: #31333F;
            margin: 0;
            padding-top: 0.2rem;
            white-space: nowrap; /* Đảm bảo chữ không bị xuống dòng nếu màn hình nhỏ */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            top: 0rem !important;
            height: 100vh !important;
        }
    </style>

    <div class="fixed-header">
        <h1 class="header-text"> Hệ thống Chuẩn đoán Khối u Tuyến vú</h1>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Cấu hình")
    st.success("Backend: Modal.com (GPU T4)")

    st.header("2. Tiền xử lý")
    use_clean = st.checkbox("Bỏ biên & lọc nhiễu", value=True,
                            help="Tô đen biên cố định, lọc nhiễu scanner, giữ lại vùng vú lớn nhất.")
    # Cắt riêng từng cạnh: cạnh thành ngực (vú chạm mép ảnh) nên để 0 để không mất mô.
    cc1, cc2 = st.columns(2)
    with cc1:
        cut_top = st.number_input("Cắt trên (px)", 0, 1000, DEFAULT_CUT_BORDER, 10, disabled=not use_clean)
        cut_left = st.number_input("Cắt trái (px)", 0, 1000, DEFAULT_CUT_BORDER, 10, disabled=not use_clean)
    with cc2:
        cut_bottom = st.number_input("Cắt dưới (px)", 0, 1000, DEFAULT_CUT_BORDER, 10, disabled=not use_clean)
        cut_right = st.number_input("Cắt phải (px)", 0, 1000, DEFAULT_CUT_BORDER, 10, disabled=not use_clean)
    dilate_px = st.number_input("Nới rộng mask vú (px)", 0, 100,
                                DILATE_PIXELS, 5, disabled=not use_clean,
                                help="Giữ lại rìa da. Kernel thực tế = giá trị này + 5.")
    if use_clean:
        st.caption("💡 Vùng vú luôn chạm 1 mép ảnh (phía thành ngực). "
                   "Đặt cạnh đó về 0 để không cắt mất mô vú.")

    st.header("3. Hậu xử lý")
    pixel_thresh = st.slider("Pixel Threshold", 0.1, 0.9, 0.3, 0.05)
    min_area = st.slider("Min Area Size (px)", 0, 500, 100, 10)

    st.header("4. Hiển thị")
    view_mode = st.radio("Kênh ảnh:", ("Grayscale", "CLAHE", "Combo", "Merged"))

col_up1, col_up2 = st.columns(2)
with col_up1: img_file = st.file_uploader("1. Ảnh Mammogram", type=["dcm", "png", "jpg"])
with col_up2: mask_file = st.file_uploader("2. Mask (Optional)", type=["xml", "dcm", "png"])

if img_file:
    # Đổi tham số tiền xử lý -> phải gửi lại ảnh sang Modal
    file_id = (f"{img_file.name}_{mask_file.name if mask_file else 'None'}"
               f"_{use_clean}_{cut_top}_{cut_bottom}_{cut_left}_{cut_right}_{dilate_px}")

    if 'curr_id' not in st.session_state or st.session_state.curr_id != file_id:
        with st.spinner("🚀 Đang xử lý ảnh & gửi sang Cloud GPU..."):
            try:
                # 1. Đọc Ảnh & Mask
                original_img = read_image_data(img_file)
                h_orig, w_orig = original_img.shape

                original_mask = None
                if mask_file:
                    if mask_file.name.endswith('.xml'):
                        original_mask, _ = create_mask_from_xml(mask_file, (h_orig, w_orig))
                    elif mask_file.name.endswith('.dcm'):
                        d = pydicom.dcmread(mask_file)
                        original_mask = (d.pixel_array > 127).astype(np.uint8)
                    else:
                        original_mask = np.array(Image.open(mask_file).convert("L"))
                        original_mask = (original_mask > 127).astype(np.uint8)
                        # Resize thô nếu mask png lệch size
                        if original_mask.shape != (h_orig, w_orig):
                            original_mask = cv2.resize(original_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                # 1.5. [MỚI] BỎ BIÊN & LỌC NHIỄU (chạy TRƯỚC Smart Crop)
                proc_img = original_img
                proc_mask = original_mask
                clean_ok = False
                if use_clean:
                    proc_img, proc_mask, breast_mask, clean_ok = remove_border_and_noise(
                        original_img,
                        cut_top=cut_top, cut_bottom=cut_bottom,
                        cut_left=cut_left, cut_right=cut_right,
                        dilate_pixels=dilate_px, gt_mask=original_mask)
                    if not clean_ok:
                        st.warning("⚠️ Không tìm thấy vùng vú hợp lệ (diện tích < 1000px) sau khi bỏ biên. "
                                   "Đang dùng ảnh gốc — hãy thử giảm giá trị cắt biên.")
                        proc_img, proc_mask = original_img, original_mask

                # 2. Preprocessing ĐỒNG BỘ (Quan trọng nhất)
                # Cắt cả ảnh và mask cùng 1 tọa độ
                square_img, square_mask = smart_crop_and_square(proc_img, proc_mask)
                img_3ch = generate_3_channels(square_img)
                # Resize cả 2 về 640x640
                input_img, input_mask = resize_final(img_3ch, square_mask, target_size=(640, 640))

                # 3. Gửi ảnh đã xử lý (640x640) sang Modal
                _, buffer = cv2.imencode('.png', input_img)
                img_b64 = base64.b64encode(buffer).decode('utf-8')

                payload = {"image": img_b64}
                resp = requests.post(MODAL_API_URL, json=payload, timeout=300)

                if resp.status_code == 200:
                    data = resp.json()
                    if "probs" not in data:
                        st.error(f"Lỗi Backend: {data}"); st.stop()
                    probs = np.array(data["probs"], dtype=np.float32)

                    st.session_state.probs = probs
                    st.session_state.input_img = input_img # Ảnh 3 kênh
                    st.session_state.input_mask = input_mask # Mask đã crop khớp
                    st.session_state.original_shape = (h_orig, w_orig)
                    st.session_state.squared_shape = square_img.shape
                    st.session_state.original_img = original_img
                    st.session_state.cleaned_img = proc_img
                    st.session_state.used_clean = use_clean and clean_ok
                    st.session_state.curr_id = file_id
                else:
                    st.error(f"Lỗi Server: {resp.text}"); st.stop()
            except Exception as e:
                st.error(f"Lỗi: {e}"); st.stop()

    # --- HIỂN THỊ ---
    if 'probs' in st.session_state:
        probs = st.session_state.probs
        input_img = st.session_state.input_img
        input_mask = st.session_state.input_mask

        raw_pred = (probs > pixel_thresh).astype(np.uint8)
        clean_pred = remove_small_objects(raw_pred, min_size=min_area)

        if input_mask is not None:
            dice, iou = calculate_metrics(input_mask, clean_pred)
            st.markdown(f"### 📊 Dice: **{dice:.4f}** | IoU: **{iou:.4f}**")
            c1, c2, c3 = st.columns(3)

            # Logic hiển thị màu/xám
            disp_img = input_img
            if view_mode == "Grayscale": disp_img = input_img[:,:,0]
            elif view_mode == "CLAHE": disp_img = input_img[:,:,1]
            elif view_mode == "Combo": disp_img = input_img[:,:,2]

            with c1: st.image(disp_img, caption=f"Input ({view_mode})", use_container_width=True)
            with c2: st.image(input_mask*255, caption="Ground Truth (Đã Crop)", use_container_width=True)
            with c3:
                ov = create_result_overlay(disp_img, input_mask, clean_pred)
                st.image(ov, caption="So sánh", use_container_width=True)
            # Chú thích màu
            st.info("""
                **Giải thích màu Overlay:**
                - 🟡 **Vàng (TP):** Model dự đoán đúng.
                - 🟢 **Xanh lá (FN):** Vùng khối u thực tế bị model bỏ sót.
                - 🔴 **Đỏ (FP):** Vùng model dự đoán sai (Dương tính giả).
                """)
        else:
            st.info("Chế độ Dự đoán")
            c1, c2 = st.columns(2)
            disp_img = input_img
            if view_mode == "Grayscale": disp_img = input_img[:,:,0]
            elif view_mode == "CLAHE": disp_img = input_img[:,:,1]
            elif view_mode == "Combo": disp_img = input_img[:,:,2]
            with c1: st.image(disp_img, caption="Input", use_container_width=True)
            with c2:
                ov = create_result_overlay(disp_img, None, clean_pred)
                st.image(ov, caption="Dự đoán", use_container_width=True)

        # --- PHẦN BỔ SUNG: EXPANDER CHI TIẾT ---
        with st.expander("🔎 Chi tiết quy trình xử lý dữ liệu"):
            st.markdown("#### 1. Thông tin ảnh đầu vào")
            st.write(f"- **Kích thước gốc:** {st.session_state.original_shape}")
            st.write(f"- **Kích thước sau khi cắt vuông (Smart Crop):** {st.session_state.squared_shape}")
            st.write(f"- **Kích thước đầu vào Model (Resize):** {input_img.shape} (640x640)")

            st.markdown("#### 2. Thông số Tiền xử lý (Preprocessing)")
            if st.session_state.get("used_clean"):
                st.write(f"- **Bỏ biên cố định (px):** Trên {cut_top} | Dưới {cut_bottom} "
                         f"| Trái {cut_left} | Phải {cut_right}")
                st.write("- **Lọc nhiễu:** Morphology OPEN (ellipse 5×5 → ngang 15×1 → dọc 1×15)")
                st.write(f"- **Giữ contour lớn nhất** (bỏ nếu diện tích < 1000px), nới rộng mask {dilate_px + 5}px")
                b1, b2 = st.columns(2)
                with b1:
                    st.image(st.session_state.original_img, caption="Ảnh gốc", use_container_width=True)
                with b2:
                    st.image(st.session_state.cleaned_img, caption="Sau bỏ biên & lọc nhiễu", use_container_width=True)
            else:
                st.write("- **Bỏ biên & lọc nhiễu:** Tắt")
            st.write("- **CLAHE:** Clip Limit = 2.0, Tile Grid = (24, 24)")
            st.write("- **Gamma Correction:** Gamma = 1.5 (Làm tối nền)")
            st.write(f"- **Normalization Mean:** `{NORM_MEAN}`")
            st.write(f"- **Normalization Std:** `{NORM_STD}`")

            st.markdown("#### 3. Cấu trúc Tensor")
            st.code(f"""
            Input Tensor Shape: (1, 3, 640, 640)
            - Channel 0: Grayscale Original
            - Channel 1: CLAHE Enhanced
            - Channel 2: Combo (Gamma + CLAHE)
            """, language="text")

