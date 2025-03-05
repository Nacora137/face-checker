import sys
import json
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import warnings
from urllib3.exceptions import InsecureRequestWarning
import base64
import chardet

warnings.simplefilter('ignore', InsecureRequestWarning)

# 얼굴 인식 모델을 로드합니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def is_base64_encoded_data(img_url: str) -> bool:
    """Base64 인코딩된 이미지인지 확인"""
    return img_url.startswith('data:image')


def decode_base64_image(base64_data: str) -> np.ndarray:
    """Base64 인코딩된 이미지를 디코딩하여 NumPy 배열로 변환"""
    try:
        encoded_data = base64_data.split(',')[1]
        image_data = base64.b64decode(encoded_data)
        return np.frombuffer(image_data, dtype=np.uint8)
    except (IndexError, ValueError):
        return np.array([])


def find_faces_in_image(image_data: str, is_base64: bool = False) -> bool:
    """주어진 이미지에서 얼굴을 감지하고 존재 여부 반환"""
    if is_base64:
        image_array = decode_base64_image(image_data)
    else:
        try:
            response = requests.get(image_data, timeout=5)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        except requests.RequestException:
            return False

    if image_array.size == 0:
        return False

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return False

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def find_faces_in_website(url: str) -> str:
    """웹사이트에서 얼굴이 포함된 이미지 URL을 JSON 형식으로 반환"""
    try:
        response = requests.get(url, timeout=5)

        # chardet을 사용하여 인코딩 감지
        detected_encoding = chardet.detect(response.content)["encoding"]
        response.encoding = detected_encoding if detected_encoding else "utf-8"

        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')

        face_detected_urls = []

        # 각 이미지 URL에 대해 얼굴 인식 시도
        for img in img_tags:
            img_url = img.get('src')
            if not img_url:
                continue

            # 상대 경로를 절대 경로로 변환
            img_url = urljoin(url, img_url) if not img_url.startswith('http') else img_url

            # Base64 인코딩된 이미지 처리
            is_base64 = is_base64_encoded_data(img_url)
            if find_faces_in_image(img_url, is_base64):
                face_detected_urls.append(img_url)

        return json.dumps({"face_detected_images": face_detected_urls}, ensure_ascii=False, indent=4)

    except requests.RequestException as e:
        return json.dumps({"error": f"Request error: {e}"}, ensure_ascii=False, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"}, ensure_ascii=False, indent=4)


# 특정 사이트 URL
site_url =  sys.argv[1]
#site_url = "https://www.google.com/search?newwindow=1&sca_esv=341cca66a365eff2&rlz=1C5CHFA_enKR1081KR1082&q=%EC%82%AC%EC%A7%84&udm=2&fbs=ABzOT_CZsxZeNKUEEOfuRMhc2yCI6hbTw9MNVwGCzBkHjFwaK53DgNHTMxn53_XGiUHS2MsNx-P-UULZBcQ-UQXCEV7DoaPM-zca8yEQJp2_Hfl3P20U8caLo68Lsnf9szP71kHBhxmI78LlTCx5wooLEXUETY1S9vMrpt0NDCWAJwomTKdQ2ZL4cLpQlyU1fwnUIFUacgJJ_FmZ9LWRidtHZ5IdcIywFQ&sa=X&ved=2ahUKEwjYjdzelvKLAxWTslYBHaDWGzUQtKgLegQIJRAB&biw=949&bih=812&dpr=2"
result_json = find_faces_in_website(site_url)

# 결과 출력
print(result_json)