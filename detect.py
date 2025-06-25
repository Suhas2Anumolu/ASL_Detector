import cv2
import torch
import torchvision.transforms as transforms
from train_model import ASLModel

labels = ['A', 'B', 'C', 'D', 'E']

model = ASLModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hand = frame[100:300, 100:300]
    img_tensor = transform(hand).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f'Prediction: {labels[pred]}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
