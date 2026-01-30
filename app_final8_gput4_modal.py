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
MODAL_API_URL = "https://snksnhsis--mammogram-ensemble-backend-api-inference.modal.run"

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ ẢNH
# ==============================================================================

def read_image_data(uploaded_file):
    if uploaded_file.name.lower().endswith('.dcm'):
        dcm_data = pydicom.dcmread(uploaded_file)
        pixel_array = dcm_data.pixel_array.astype(np.float32)
        # Min-Max Scaling về 0-255
        pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)
        
        # Nếu ảnh dicom bị đảo ngược (nền trắng), đảo lại thành nền đen
        if dcm_data.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = 255 - pixel_array
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

def smart_crop_and_square(img, mask=None):
    h, w = img.shape
    _, bin_img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(bin_img)
    if coords is None: return img, mask
    
    x, y, w_box, h_box = cv2.boundingRect(coords)
    crop_img = img[y:y+h_box, x:x+w_box]
    
    crop_mask = None
    if mask is not None:
        crop_mask = mask[y:y+h_box, x:x+w_box]
    
    h_c, w_c = crop_img.shape[:2]
    square_size = max(h_c, w_c)
    
    final_img = np.zeros((square_size, square_size), dtype=np.uint8)
    final_mask = np.zeros((square_size, square_size), dtype=np.uint8) if mask is not None else None
    
    left_sum = np.sum(crop_img[:, :w_c//2])
    right_sum = np.sum(crop_img[:, w_c//2:])
    
    # Đặt ảnh vào giữa hoặc lệch về phía có vú (thường làm background đen phần còn lại)
    # Logic cũ: đặt vào góc. Ở đây giữ nguyên logic cũ của bạn.
    y_pos = 0
    x_pos = 0 if left_sum > right_sum else square_size - w_c
        
    final_img[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_img
    if mask is not None:
        final_mask[y_pos:y_pos+h_c, x_pos:x_pos+w_c] = crop_mask
        
    return final_img, final_mask

def resize_final(img, mask=None, target_size=(640, 640)):
    # Resize ảnh xám
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
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
    # Chuyển ảnh nền xám sang RGB để vẽ màu lên
    if len(bg_img.shape) == 2:
        bg = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2RGB)
    else:
        bg = bg_img.copy()
        
    overlay = bg.copy()
    
    if gt is None:
        overlay[pred==1] = [0, 0, 255] # Red cho dự đoán
    else:
        overlay[np.logical_and(gt==1, pred==1)] = [255, 215, 0] # TP: Vàng
        overlay[np.logical_and(gt==1, pred==0)] = [0, 255, 0]   # FN: Xanh lá
        overlay[np.logical_and(gt==0, pred==1)] = [255, 0, 0]   # FP: Đỏ
        
    return cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
st.set_page_config(page_title="Mammogram Detection (Grayscale)", layout="wide")

st.markdown("""
    <style>
        header[data-testid="stHeader"] { background-color: transparent !important; z-index: 99999; }
        .block-container { padding-top: 5rem !important; }
        .fixed-header {
            position: fixed; top: 0; left: 14; width: 100%; height: 3.75rem;
            background-color: white; z-index: 99990;
            display: flex; align-items: center; justify_content: flex-end;
            padding-right: 14rem; border-bottom: 1px solid #f0f0f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .header-text {
            font-family: 'Source Sans Pro', sans-serif; font-weight: 700;
            font-size: 2rem; color: #31333F; margin: 0; padding-top: 0.2rem;
            white-space: nowrap;
        }
        section[data-testid="stSidebar"] { top: 0rem !important; height: 100vh !important; }
    </style>
    <div class="fixed-header">
        <h1 class="header-text"> Hệ thống Chuẩn đoán Khối u (Grayscale)</h1>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1. Cấu hình")
    st.success("Backend: Modal.com (GPU T4)")
    st.header("2. Hậu xử lý")
    pixel_thresh = st.slider("Pixel Threshold", 0.1, 0.9, 0.3, 0.05)
    min_area = st.slider("Min Area Size (px)", 0, 500, 100, 10)
    # Đã bỏ phần chọn View Mode vì chỉ còn ảnh xám

col_up1, col_up2 = st.columns(2)
with col_up1: img_file = st.file_uploader("1. Ảnh Mammogram", type=["dcm", "png", "jpg"])
with col_up2: mask_file = st.file_uploader("2. Mask (Optional)", type=["xml", "dcm", "png"])

if img_file:
    file_id = f"{img_file.name}_{mask_file.name if mask_file else 'None'}"
    
    if 'curr_id' not in st.session_state or st.session_state.curr_id != file_id:
        with st.spinner("🚀 Đang xử lý ảnh & gửi sang Cloud GPU..."):
            try:
                # 1. Đọc Ảnh & Mask
                original_img = read_image_data(img_file) # Ảnh xám 2D
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
                        if original_mask.shape != (h_orig, w_orig):
                            original_mask = cv2.resize(original_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                # 2. Preprocessing ĐỒNG BỘ
                square_img, square_mask = smart_crop_and_square(original_img, original_mask)
                
                # Resize về 640x640 (Vẫn là 1 kênh)
                input_img_1ch, input_mask = resize_final(square_img, square_mask, target_size=(640, 640))
                
                # CHUẨN BỊ GỬI SANG MODEL
                # Model Unet++ yêu cầu 3 channels. Ta nhân bản kênh xám lên 3 lần.
                # Shape: (640, 640, 3)
                input_img_3ch = cv2.merge([input_img_1ch, input_img_1ch, input_img_1ch])
                
                # 3. Encode & Gửi
                _, buffer = cv2.imencode('.png', input_img_3ch)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                
                payload = {"image": img_b64}
                resp = requests.post(MODAL_API_URL, json=payload, timeout=300)
                
                if resp.status_code == 200:
                    data = resp.json()
                    probs = np.array(data["probs"], dtype=np.float32)
                    
                    st.session_state.probs = probs
                    st.session_state.input_img_1ch = input_img_1ch # Lưu ảnh xám để hiển thị
                    st.session_state.input_mask = input_mask
                    st.session_state.original_shape = (h_orig, w_orig)
                    st.session_state.squared_shape = square_img.shape
                    st.session_state.curr_id = file_id
                else:
                    st.error(f"Lỗi Server: {resp.text}"); st.stop()
            except Exception as e:
                st.error(f"Lỗi: {e}"); st.stop()

    # --- HIỂN THỊ ---
    if 'probs' in st.session_state:
        probs = st.session_state.probs
        input_img_1ch = st.session_state.input_img_1ch
        input_mask = st.session_state.input_mask
        
        raw_pred = (probs > pixel_thresh).astype(np.uint8)
        clean_pred = remove_small_objects(raw_pred, min_size=min_area)
        
        if input_mask is not None:
            dice, iou = calculate_metrics(input_mask, clean_pred)
            st.markdown(f"### 📊 Dice: **{dice:.4f}** | IoU: **{iou:.4f}**")
            c1, c2, c3 = st.columns(3)
            
            with c1: st.image(input_img_1ch, caption="Input (Grayscale)", use_container_width=True)
            with c2: st.image(input_mask*255, caption="Ground Truth", use_container_width=True)
            with c3: 
                ov = create_result_overlay(input_img_1ch, input_mask, clean_pred)
                st.image(ov, caption="Kết quả", use_container_width=True)
            
            st.info("""
                **Giải thích màu Overlay:**
                - 🟡 **Vàng (TP):** Model dự đoán đúng.
                - 🟢 **Xanh lá (FN):** Vùng khối u thực tế bị model bỏ sót.
                - 🔴 **Đỏ (FP):** Vùng model dự đoán sai (Dương tính giả).
                """)
        else:
            st.info("Chế độ Dự đoán (Không có Mask)")
            c1, c2 = st.columns(2)
            with c1: st.image(input_img_1ch, caption="Input (Grayscale)", use_container_width=True)
            with c2: 
                ov = create_result_overlay(input_img_1ch, None, clean_pred)
                st.image(ov, caption="Dự đoán", use_container_width=True)
        
        with st.expander("🔎 Chi tiết quy trình"):
            st.write(f"- **Kích thước gốc:** {st.session_state.original_shape}")
            st.write(f"- **Kích thước đầu vào Model:** {input_img_1ch.shape} (Resize về 640x640)")
            st.write("- **Chế độ:** Ảnh xám 3 kênh (R=G=B)")

