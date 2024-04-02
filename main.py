import sys
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import warnings
from urllib3.exceptions import InsecureRequestWarning
import base64

warnings.simplefilter('ignore', InsecureRequestWarning)

# 얼굴 인식 모델을 로드합니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def is_base64_encoded_data(img_url):
    # Base64 인코딩된 데이터 URL 여부 확인
    return img_url.startswith('data:image')


def decode_base64_image(base64_data):
    # Base64 데이터 디코딩
    base64_encoded_data = base64_data.split(',')[1]
    image_data = base64.b64decode(base64_encoded_data)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    return image_array


def find_faces_in_image(image_data, is_base64=False):
    if is_base64:
        image_array = decode_base64_image(image_data)
    else:
        response = requests.get(image_data)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def find_faces_in_website(url):
    try:
        # 웹 페이지에서 이미지 URL 추출
        page_content = requests.get(url).content
        soup = BeautifulSoup(page_content, 'html.parser')
        img_tags = soup.find_all('img')

        # 각 이미지 URL에 대해 얼굴 인식 시도
        for img in img_tags:
            img_url = img.get('src')
            if img_url:
                # 이미지 URL이 상대 경로인 경우 절대 경로로 변환
                if not img_url.startswith('http'):
                    img_url = urljoin(url, img_url)

                if is_base64_encoded_data(img_url):
                    # Base64 인코딩된 이미지 URL 처리
                    if find_faces_in_image(img_url, is_base64=True):
                        print(f"얼굴이 검출된 이미지: {img_url}")
                else:
                    # 일반 이미지 URL 처리
                    if find_faces_in_image(img_url):
                        print(f"얼굴이 검출된 이미지: {img_url}")
    except Exception as e:
        print(e)
        return False


# 특정 사이트 URL
site_url =  sys.argv[1]
find_faces_in_website(site_url)
