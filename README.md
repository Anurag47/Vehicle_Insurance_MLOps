# 🚀 Vehicle Data Processing & ML Pipeline

## 📌 Project Overview
This project is a **production-grade AI agent** that leverages **MongoDB, FastAPI, AWS, Docker, and CI/CD pipelines** to build a scalable **Machine Learning Pipeline** for vehicle data processing. The system is deployed on an **EC2 instance** using a **Dockerized ECR image** and features robust **data ingestion, validation, transformation, model training, and deployment** workflows.

---

## 📂 Folder Structure
```bash
📦 vehicle-data-ml
├── src/                    # Source code
│   ├── components/         # Data pipeline components
│   ├── configuration/      # Config files (DB connections, AWS setup)
│   ├── data_access/        # MongoDB data access layer
│   ├── entity/             # Entity classes for data handling
│   ├── pipeline/           # ML training & prediction pipelines
│   ├── cloud_storage/      # AWS S3 storage interface
├── notebook/               # Jupyter notebooks (EDA, Data Processing)
├── static/                 # Frontend assets
├── templates/              # HTML templates for FastAPI
├── .github/workflows/      # CI/CD Pipeline setup
├── requirements.txt        # Required Python dependencies
├── Dockerfile              # Docker container setup
├── pyproject.toml          # Package configuration
├── setup.py                # Python package installer
└── README.md               # Project Documentation
```

---

## 🛠️ Setup & Installation

### 1️⃣ Create the Project Environment
```bash
# Generate project structure
python template.py
```

### 2️⃣ Install Dependencies
```bash
# Create and activate virtual environment
conda create -n vehicle python=3.10 -y
conda activate vehicle

# Install required packages
pip install -r requirements.txt

# Verify package installation
pip list
```

### 3️⃣ MongoDB Atlas Setup
```bash
# Sign up & create a MongoDB Atlas project
# Create a cluster & setup credentials
# Allow network access from 0.0.0.0/0
# Copy MongoDB connection string
```
Add the MongoDB connection string to your **.env** file:
```env
MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net"
```

### 4️⃣ Logging & Exception Handling
```bash
# Implement logging and exceptions
python demo.py  # Test logging & exception handling
```

### 5️⃣ Data Ingestion Pipeline
```bash
# Define constants, configuration, and database connection
python demo.py  # Run data ingestion
```

Set MongoDB credentials in your **PowerShell terminal**:
```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>......"
echo $env:MONGODB_URL
```

### 6️⃣ AWS Setup & Model Deployment
```bash
# Login to AWS and create IAM user
# Set AWS credentials as environment variables
$env:AWS_ACCESS_KEY_ID="<your-access-key>"
$env:AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```
Add AWS configurations in **src/configuration/aws_connection.py**
```python
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

### 7️⃣ Docker & CI/CD Deployment
```bash
# Build & Push Docker Image
# Create EC2 instance and setup self-hosted runner
# Set up GitHub Actions & ECR repository
# Deploy the FastAPI app
```
Check your app at:
```bash
http://<your-ec2-public-ip>:5080
```

---

## 🚀 Features
✅ **End-to-End ML Pipeline**: Data ingestion → Validation → Transformation → Training → Deployment  
✅ **Cloud Integration**: MongoDB Atlas for data storage, AWS S3 for model storage  
✅ **CI/CD & Docker**: Automated deployments via GitHub Actions & Docker  
✅ **FastAPI Backend**: High-performance API to serve predictions  
✅ **Scalable Deployment**: Hosted on AWS EC2 using ECR Docker images  

---

## 💡 Future Enhancements
- ✅ Implement real-time data streaming
- ✅ Deploy a model monitoring system
- ✅ Optimize pipeline efficiency with parallel processing

---

## 🤝 Contribution Guidelines
We welcome contributions! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your branch (`git push origin feature-branch`)
5. Create a Pull Request

---

## 📞 Contact
👤 **Anurag Bhamidipati**   
🌐 [LinkedIn Profile](https://linkedin.com/in/anuragbhamidipati)

