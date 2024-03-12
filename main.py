from resnet_inference import predict_class
import os 
import uuid
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from fastapi import FastAPI, File, UploadFile, HTTPException
from CogVLM.basic_demo.cli_demo_sqa import CogVLM_inference
app = FastAPI()

upload_dir = './upload/'
os.makedirs(upload_dir,exist_ok=True)
legal_extensions = ['png', 'jpg', 'jpeg']

endpoint = os.environ["OSS_ENDPOINT"]
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, endpoint, f'{os.environ["BUCKET_NAME"]}')

@app.get('/')
async def hello_world():
    return{
            'hello':'world'
        }

@app.post('/upload')
async def upload_image(file:UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1]
        if file_extension in legal_extensions:
            file_name = str(uuid.uuid4()) + f'.{file_extension}'
            file_path = os.path.join(upload_dir, file_name)
            with open(file_path, 'wb') as file_obj:
                file_obj.write(file.file.read())
            bucket.put_object_from_file(f'{os.environ["BUCKET_UPLOAD_PATH"]}', file_path)
            os.remove(file_path)

            return {
                    'task_id': file_name.split('.')[0]
                }
        else:
            raise HTTPException(
                    status_code = 500,
                    detail = 'invalid image extension format, we accepted [png] [jpg] [jpeg]'
                )
    except Exception as e:
        print(e)
        raise HTTPException(
                status_code = 500,
                detail = f'Upload failed please try again later'
            )
        
@app.get('/resnet')
async def resnet_inference(task_id: str, topk_result: int):
    for file_extension in legal_extensions:
        file_name = f'{task_id}.{file_extension}'
        local_file_path = f'./upload/{file_name}'
        bucket.get_object_to_file(f'{os.environ["BUCKET_UPLOAD_PATH"]}', local_file_path)
        if os.path.exists(local_file_path):
            inference_result = predict_class(local_file_path,topk_result)
            os.remove(local_file_path)
            return{
                    'result' : inference_result
                }
        else:
            raise HTTPException(
                    status_code = 404,
                    detail = f'{task_id} task not found'
                )

@app.post('/cogvlm')
async def resnet_inference(task_id: str, question: str):
    for file_extension in legal_extensions:
        file_name = f'{task_id}.{file_extension}'
        local_file_path = f'./upload/{file_name}'
        bucket.get_object_to_file(f'{os.environ["BUCKET_UPLOAD_PATH"]}', local_file_path)
        if os.path.exists(local_file_path):
            img_path = []
            img_path.append(local_file_path)
            inference_result = CogVLM_inference(img_path,question) 
            os.remove(local_file_path)
            return{
                    'result' : inference_result
                }
        else:
            raise HTTPException(
                    status_code = 404,
                    detail = f'{task_id} task not found'
                )
