import cv2, numpy as np, re, pytesseract, json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ChartAnalyzer:
    """تحلیلگر چارت‌های مهندسی نفت"""
    
    def __init__(self, weights_path='runs/detect/train/weights/best.pt'):
        self.model = YOLO(weights_path)
        self.results_history = []
        # print(f"✔ Model loaded: {weights_path}")
        try:
            pass # print(f"✔ Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"⚠️ Tesseract warning: {e}")
        
    def detect_plot_area_yolo(self, img_bgr):
        names = self.model.names
        inv = {v:k for k,v in names.items()}
        if "plot_area" not in inv: return None
        res = self.model.predict(source=img_bgr, verbose=False, conf=0.25, iou=0.5)[0]
        boxes = []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if names[int(c)] == "plot_area":
                x1,y1,x2,y2 = map(int, b); area = (x2-x1)*(y2-y1)
                boxes.append((x1,y1,x2,y2,area))
        return sorted(boxes, key=lambda t:t[-1], reverse=True)[0][:4] if boxes else None

    def detect_plot_area_cv(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape; candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.1*W*H or area > 0.9*W*H: continue
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 4:
                x,y,w,h = cv2.boundingRect(approx)
                if 0.3 < (w/h) < 3.0: candidates.append((x,y,x+w,y+h, area))
        return sorted(candidates, key=lambda t:t[-1], reverse=True)[0][:4] if candidates else None

    def enhanced_ocr(self, img_gray, psm=11):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); img_gray = clahe.apply(img_gray)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]); img_gray = cv2.filter2D(img_gray, -1, kernel)
        configs = [f'--oem 3 --psm {psm} -l eng', f'--oem 3 --psm 8 -l eng', f'--oem 3 --psm 7 -l eng']
        all_results = []
        for cfg in configs:
            try:
                data = pytesseract.image_to_data(img_gray, config=cfg, output_type=pytesseract.Output.DICT)
                for txt, x, y, w, h, conf in zip(data['text'], data['left'], data['top'], data['width'], data['height'], data['conf']):
                    if int(conf) < 30: continue
                    matches = re.findall(r'[-+]?\d*\.?\d+', txt)
                    if matches:
                        try:
                            val = float(matches[0])
                            all_results.append((val, x, y, w, h, conf))
                        except: continue
            except Exception: continue
        unique_results = [];
        for res in all_results:
            is_dup = False
            for ex in unique_results:
                if abs(res[1] - ex[1]) < 20 and abs(res[2] - ex[2]) < 20:
                    if res[5] > ex[5]: unique_results.remove(ex); break
                    else: is_dup = True; break
            if not is_dup: unique_results.append(res)
        return unique_results

    def build_axis_mapping_simple(self, candidates, axis='x', W=640, H=640):
        vals, pix = [], []
        if axis == 'x':
            for v,x,y,w,h,conf in candidates:
                if y > H*0.75: vals.append(v); pix.append(x + w/2)
        else:
            for v,x,y,w,h,conf in candidates:
                if x < W*0.25: vals.append(v); pix.append(y + h/2)
        if len(vals) < 2: return None
        if len(vals) > 3:
            q1, q3 = np.percentile(vals, [25, 75]); iqr = q3 - q1
            lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
            filt = [(v, p) for v, p in zip(vals, pix) if lb <= v <= ub]
            if len(filt) >= 2: vals, pix = zip(*filt)
        if len(vals) >= 2:
            slope, intercept = np.polyfit(pix, vals, 1)
            return (slope, intercept)
        return None

    def create_visualization(self, img_bgr, roi, detections):
        vis = img_bgr.copy()
        x1_roi, y1_roi, x2_roi, y2_roi = roi
        for det in detections:
            cls, box = det['cls'], det['box']
            x1b, y1b, x2b, y2b = box
            cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0,255,0), 2)
            label_parts = [cls]
            if det.get('y_value') is not None: label_parts.append(f"y={det['y_value']:.1f}")
            if det.get('x_value') is not None: label_parts.append(f"x={det['x_value']:.2f}")
            label = " | ".join(label_parts)
            cv2.putText(vis, label, (x1b+3, max(15, y1b-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (x1_roi, y1_roi), (x2_roi, y2_roi), (255,0,0), 2)
        return vis

    def analyze_chart(self, image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise ValueError(f"Cannot load image: {image_path}")
        H0, W0 = img_bgr.shape[:2]
        roi = self.detect_plot_area_yolo(img_bgr)
        if roi is None: roi = self.detect_plot_area_cv(img_bgr)
        if roi is None: roi = (0, 0, W0, H0)
        x1r, y1r, x2r, y2r = roi
        crop = img_bgr[y1r:y2r, x1r:x2r].copy()
        H, W = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        all_candidates = []
        for scale in [1.0, 1.5]:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            candidates = self.enhanced_ocr(scaled, psm=11)
            all_candidates.extend([(v, x/scale, y/scale, w/scale, h/scale, conf) for v,x,y,w,h,conf in candidates])
        mapx = self.build_axis_mapping_simple(all_candidates, axis='x', W=W, H=H)
        mapy = self.build_axis_mapping_simple(all_candidates, axis='y', W=W, H=H)
        res = self.model.predict(source=crop, verbose=False, conf=0.2, iou=0.4)[0]
        detections = []
        for b, c, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            cls = self.model.names[int(c)]
            x1b, y1b, x2b, y2b = b.astype(int)
            cx, cy = (x1b + x2b) / 2, (y1b + y2b) / 2
            x_val = mapx[0] * cx + mapx[1] if mapx else None
            y_val = mapy[0] * cy + mapy[1] if mapy else None
            detections.append({
                'cls': cls, 'confidence': float(conf),
                'box': [int(x1r+x1b), int(y1r+y1b), int(x1r+x2b), int(y1r+y2b)],
                'x_value': float(x_val) if x_val is not None else None,
                'y_value': float(y_val) if y_val is not None else None
            })
        vis_image = self.create_visualization(img_bgr, roi, detections)
        plt.figure(figsize=(12, 8)); plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
        return detections

# توابع کمکی برای استفاده آسان
def analyze_chart(image_path, weights_path='runs/detect/train/weights/best.pt'):
    analyzer = ChartAnalyzer(weights_path)
    return analyzer.analyze_chart(image_path)

def test_multiple_charts(weights_path='runs/detect/train/weights/best.pt'):
    import glob, random
    paths = glob.glob('/content/fst_dataset_v6/images/test/*.png')
    if not paths: print("❌ No test images found!"); return
    random.shuffle(paths); selected = paths[:3]
    analyzer = ChartAnalyzer(weights_path)
    for path in selected:
        try:
            analyzer.analyze_chart(path)
        except Exception as e:
            print(f"❌ Error on {path}: {e}")
