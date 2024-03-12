import requests
import time

def hello_world():
    url = 'http://0.0.0.0:8000'
    response = requests.get(url)
    print(response.status_code)
    print(response.json())

def upload_img(img_path):
    url = 'http://0.0.0.0:8000/upload/'
    img = img_path
    try:
        with open(img,'rb') as file:
            data = {'file':file}
            response = requests.post(url, files=data)
            if response.status_code == 200:
                return response.json()['task_id']

    except requests.exceptions.RequestException as e:
        print(e)

def resnet_inference(img_path,topk_result):
    task_id = upload_img(img_path)
    url = f'http://0.0.0.0:8000/resnet?task_id={task_id}&topk_result={topk_result}'
    response = requests.get(url)
    print(response.status_code)
    print(response.json())

def cogvlm_inference(img_path, question):
    task_id = upload_img(img_path)
    url = f'http://0.0.0.0:8000/cogvlm?task_id={task_id}&question={question}'
    response = requests.post(url)
    print(response.status_code)
    print(response.json())

