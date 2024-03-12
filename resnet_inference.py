import torch
import json
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 加载并预处理图片
def preprocess_img(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with Image.open(img_path) as image:
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
    return input_batch

#加载预训练模型并预测
def predict_class(img_path,topk):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    with torch.no_grad():
        input_batch = preprocess_img(img_path)
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_classes = torch.topk(probabilities, topk, dim=1)

  # 获取类别名称
    imagenet_class_index = json.load(
        open("imagenet_class_index.json", encoding="utf-8")  #
    )

  # 获取前topk个预测的类别索引
    top_class_index = top_classes[0].tolist()  #

  # 获取前topk个预测的类别名称
    top_class_names = [imagenet_class_index[str(index)] for index in top_class_index]

  # 输出结果
    predicted_result = f'Top {topk} predicted classes:\n'
    for i, class_name in enumerate(top_class_names, start=1):
        probability = top_probs[0][i-1].item()
        predicted_result += f"{i}. Class: {class_name[-1]}, Probability: {probability:.2f}\n"
    return predicted_result


