from flask import Flask, request, jsonify
import os
from PIL import Image
import cv2
import torch
import numpy as np

import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1
from model import siamese_model 
from live_check import predict
import datetime as dt
import mysql.connector
app = Flask(__name__)

#Database Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'company'

# Set paths and configurations
siamese_model_path = "saved_models/siamese_model"
db_path = "database/"
database_embeddings_path = os.path.join(db_path, "database_embeddings")
device = "cuda" if torch.cuda.is_available() else "cpu"
margin = 0
THRESHOLD = 0.55

#Database Connection
mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

##Face Recognition Model setup
mtcnn = MTCNN(image_size=128, margin=margin).eval()
resnet = InceptionResnetV1(pretrained="vggface2").to(device).eval()
loader = T.Compose([T.ToTensor()])

model = siamese_model()
model.load_state_dict(torch.load(siamese_model_path, map_location=torch.device('cpu')))
model.eval()
model.to(device)

#input image processing for face embedding
def process_input_image(input_img, mtcnn, resnet):
    boxes, probs, points = mtcnn.detect(input_img, landmarks=True)
    if boxes is not None and len(boxes) > 0:
        bbox = boxes[0]
        input_img = np.array(input_img)
        box = (np.array(bbox)).astype(int)
        cropped_face = input_img[box[1] : box[3] + 1, box[0] : box[2] + 1]
        input_img = cv2.resize(cropped_face, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)  # Normalizing and converting to tensor
        face_embedding = resnet(input_img.unsqueeze(0).to(device)).reshape((1, 1, 512))
        return face_embedding,bbox
    return None,None

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Face Recognition API!"

#Registration Process
@app.route('/register', methods=['POST'])
def register():
    try:
        uploaded_image = request.files['image']
        emp_id = int(request.form['emp_id'])
        full_name = str(request.form['full_name'])
        id=f'{str(emp_id)}'
        c = mysql.cursor()
        c.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
        existing_employee = c.fetchone()
        if existing_employee:
            return {"status": False, "message": "Employee with this ID already exists"}
        filename = f'{str(emp_id)}.jpg'
        if os.path.exists(database_embeddings_path):
            saved_data = torch.load(database_embeddings_path)
            if 'reference' in saved_data:
                reference_cropped_img = saved_data['reference']
                emp_ids = saved_data['emp_id']
            else:
                reference_cropped_img = []
                emp_ids = []
        else:
            reference_cropped_img = []
            emp_ids = []

        foldername = os.path.splitext(filename)[0]
        try:
            reference_img = Image.open(uploaded_image)
            try:
                reference_img = reference_img.convert("RGB")
            except:
                pass
            boxes, probs, points = mtcnn.detect(reference_img, landmarks=True)

            if boxes is None:
                return {"status": False, "message": "No face Detected"}

            boxes = (np.array(boxes[0])).astype(int)
            input_img = np.array(reference_img)[
                boxes[1] : boxes[3] + 1, boxes[0] : boxes[2] + 1
            ].copy()
            input_img = cv2.resize(
                input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC
            )
            input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)
            
            if id in emp_ids:
                index = emp_ids.index(id)
                reference_cropped_img[index] = input_img
            else:
                reference_cropped_img.append(input_img)
                emp_ids.append(id)
        except Exception as e:
            return {"status": False, "message": "Error occurred during image processing " + str(e)}

        query = "INSERT INTO employees (id,full_name) VALUES (%s,%s)"
        val = (emp_id,full_name)
        c.execute(query, val)
        mysql.commit()
        c.close()
        torch.save({"emp_id": emp_ids, "reference": reference_cropped_img}, database_embeddings_path)
        db_folder_path = os.path.join(db_path, foldername)
        os.makedirs(db_folder_path, exist_ok=True)

        image_db_full_path = os.path.join(db_folder_path, filename)
        reference_img.save(image_db_full_path)

        result = {"status": True, "message": "Face uploaded"}
        return jsonify(result)
    except Exception as e:
        return {"status": False, "message": "Error occurred during image processing " + str(e)}

#Update Embedding
@app.route('/update_embeddings', methods=['POST'])
def update_embeddings():
    try:
        new_reference_cropped_img = []
        new_emp_ids = []

        for tenant_folder in os.listdir(db_path):
            tenant_embeddings = []
            tenant_empnames = []

            tenant_folder_path = os.path.join(db_path, tenant_folder)
            if os.path.isdir(tenant_folder_path):
                for root, _, files in os.walk(tenant_folder_path):
                    for file in files:
                        if file.endswith('.jpg'):
                            emp_name=file.split('.')[0]
                            file_path = os.path.join(root, file)
                            reference_img = Image.open(file_path)
                            reference_img = reference_img.convert("RGB")

                            boxes, _, _ = mtcnn.detect(reference_img, landmarks=True)
                            if boxes is not None:
                                boxes = boxes[0].astype(int)
                                input_img = np.array(reference_img)[boxes[1]:boxes[3] + 1, boxes[0]:boxes[2] + 1].copy()
                                input_img = cv2.resize(input_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                                input_img = loader((input_img - 127.5) / 128.0).type(torch.FloatTensor)
                                tenant_embeddings.append(input_img)
                                emp_id = file.split('_')[-1].split('.')[0]  
                                tenant_empnames.append(emp_name)
                new_reference_cropped_img.extend(tenant_embeddings)
                new_emp_ids.extend(tenant_empnames)

        torch.save({"emp_id": new_emp_ids, "reference": new_reference_cropped_img}, database_embeddings_path)
        print("Embeddings updated and saved successfully!")

        result = {"status": True, "message": "Embeddings updated successfully"}
        return jsonify(result)
    except Exception as e:
        return {"status": False, "message": "Error occurred during embedding update " + str(e)}

#Face Recognition
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    current_time = dt.datetime.now().time()
    timestamp = current_time.strftime('%H:%M:%S')
    current_date = dt.date.today()
    try:
        uploaded_image = request.files['image']
        emp_id = int(request.form['emp_id'])
        come_at = 'come_at' in request.form
        return_at = 'return_at' in request.form
        filename = "image.jpg"
        image = Image.open(uploaded_image)
        try:
            image = image.convert("RGB")
        except:
            pass
        image_full_path = os.path.join('uploads', filename)
        image.save(image_full_path)
        #Real and Fake testing
        prediction=predict(image_full_path)
        if prediction == "fake":
            if os.path.exists(image_full_path):
                os.remove(image_full_path)
            return {"status": False, "message": "Fake image detected", "label": "Fake", "similarity_value": 0, "bbox": None}
        
        input_img = Image.open(image_full_path)
        face_embedding, image_bbox = process_input_image(input_img, mtcnn, resnet)
        
        if face_embedding is None:
            if os.path.exists(image_full_path):
                os.remove(image_full_path)
            return {"status": False, "message": "No face Detected", "label": "No Face", "similarity_value": 0, "bbox": None}

        if face_embedding is not None:
            reference_data = torch.load(database_embeddings_path)
            reference_embeddings = reference_data["reference"]
            reference_names = reference_data["emp_id"]
            max_similarity = -1
            image_bbox = list(image_bbox)
            label = ""
            
            for ref_embedding, ref_name in zip(reference_embeddings, reference_names):
                if ref_name == f'{str(emp_id)}':
                    ref_embedding = resnet(ref_embedding.unsqueeze(0).to(device)).reshape((1, 1, 512))
                    similarity = model(ref_embedding.to(device), face_embedding.to(device)).item()
                    max_similarity = similarity
                    label = ref_name
                    image_bbox = list(image_bbox)
                             
                    if similarity >= THRESHOLD:
                        cursor = mysql.cursor()
                        print("cursor")
                        if return_at:
                            get_come_at_query = f"SELECT come_at FROM employee_attendance WHERE employee_id = {emp_id} AND date = '{current_date}' ORDER BY come_at DESC LIMIT 1"
                            cursor.execute(get_come_at_query)
                            come_at_result = cursor.fetchone()

                            if come_at_result:
                                come_at_timestamp = come_at_result[0]
                                query = f"UPDATE employee_attendance SET return_at = '{timestamp}' WHERE employee_id = {emp_id} AND date = '{current_date}' AND come_at = '{come_at_timestamp}'"
                                message = "Attendance marked Successfully"
                                status=True
                                cursor.execute(query)

                            else:
                                message = "No check-in found for today to mark return"
                                status=False
                            mysql.commit()
                            cursor.close()
                        else:
                            
                            query = f"INSERT INTO employee_attendance (employee_id, date, come_at) VALUES ({emp_id}, '{current_date}', '{timestamp}') ON DUPLICATE KEY UPDATE come_at = '{timestamp}'"
                            message="Attendance marked Successfully"
                            status=True
                            cursor.execute(query)
                            mysql.commit()
                            cursor.close()

                        result = {"status": status, "message": f"{message}", "label": label, "similarity_value": max_similarity, "bbox": image_bbox}
                    else:
                        result = {"status": False, "message": "Face not recognized", "label": "unknown", "similarity_value": max_similarity, "bbox": image_bbox}
                    
                    break  
            else:
                result = {"status": False, "message": "Employee ID not found in database", "label": "unknown", "similarity_value": 0, "bbox": image_bbox}
        
        else:
            result = {"status": False, "message": "No face detected in the uploaded image", "label": "unknown", "similarity_value": 0, "bbox": None}
        
        if os.path.exists(image_full_path):
            os.remove(image_full_path)
        
        return jsonify(result)

    except Exception as e:
        if os.path.exists(image_full_path):
            os.remove(image_full_path)
        return jsonify({"status": False, "message": str(e), "label": "unknown", "similarity_value": 0, "bbox": None})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0',debug=True)
