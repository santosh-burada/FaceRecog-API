
# Developing a Scalable and Distributed Face Recognition System using Kubernetes and Docker
[![GitHub last commit](https://img.shields.io/github/last-commit/santosh-burada/FaceRecog-API)](https://github.com/santosh-burada/FaceRecog-API)
![GitHub issues](https://img.shields.io/github/issues/santosh-burada/FaceRecog-API)
![GitHub stars](https://img.shields.io/github/stars/santosh-burada/FaceRecog-API?style=social)


## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation Guide](#installation-guide)
- [License](#license)
- [Contact](#contact)

## Project Description
This project aims to build a robust, scalable, and distributed face recognition system that leverages the power of Kubernetes and Docker. Designed to efficiently handle face recognition demands across various applications such as security and identity verification.

## Features
- **Scalability**: Handles an increasing number of requests without performance degradation.
- **Distributed Architecture**: Distributes computational load across multiple nodes.
- **High Availability**: Ensures minimal downtime and continuous operation.
- **Modularity**: Containerizes each component, simplifying deployment and scaling.
- **User-friendly Interface**: Features an intuitive interface for easy user interaction.

## Technologies Used
- **Kubernetes**: Manages and orchestrates the containerized applications.
- **Docker**: Provides containerization for each system component.
- **Face Recognition Algorithms**: Utilizes state-of-the-art algorithms for high accuracy.
- **AWS**: Utilized for deploying the application in Amazon Elastic Kubernetes Service (EKS), facilitating scalable and publicly accessible services.

## Installation Guide
This guide provides detailed instructions on how to set up and run the project both locally and on Kubernetes.

### Local Installation
<details>
<summary>Click to expand!</summary>

#### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

#### 2. Set up a Virtual Environment
Navigate to the Auth, MTCNN-detector, Recognition and Training Folders
Create and activate a virtual environment to isolate the project dependencies:
```bash
python3 -m venv env
# On Windows
env\Scripts\activate
# On macOS and Linux
source env/bin/activate
```

#### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
#### 4. create a mongodDB free tier account and get the MongoDB URI, Database Name and Collections named as below
```bash
Training_Data # required as it is
users # required as it is
```
#### 5. Create a random secret key for encrypting the password and for token generation
```bash
openssl rand -hex 32  #provides a random hex string
```

#### 6. Configure Environment Variables
Add the following variables and their respective values to an `.env` file:
```bash
SECRET_KEY='your_secret_key_here'
MONGO_URI='your_mongodb_uri_here'
DATABASE_NAME='your_database_name_here'
```

#### 7. Run the Application
Now there are main.py and client.py files in Auth MTCNN-detector Recognition and Training folders you can run thoes files by activating the virtual environments. First run the Auth/main.py and perform SignUp and Login for a valid 1 hour token. Then only you can acess the other services.

    The Password is full encrypted using the SECRECT_KEY we generated. Make sure it is not public.
Start the application locally:
```bash
python main.py
```
#### 8. When running the Mtcnn client file there are 3 keys we have to use for the program**
```bash
IMP: Stay on Camera Display Window

key c to capture the crop face
key s to save the captured Data
key ESC to exit the program
```
#### 9. When any of the client file is runinng it ask's for the Email and other command line Inputs Keep An Eye on them**

</details>

### Kubernetes Local Deployment
<details>
<summary>Click to expand!</summary>

#### 1. Follow the same steps as mentoed in the Step 1
First create a secret using the .env which configured in the step 1
```bash
# navigate to the folder where .env exists
kubectl create secret generic my-env-secret --from-env-file=.env
```
use the below yaml file to deploy the services in local cluster
```bash
---
apiVersion: v1
kind: Namespace
metadata:
    name: facerec
---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: crop-face-deployment
    namespace: facerec
spec:
    replicas: 3
    selector:
    matchLabels:
        app: crop-face
    template:
    metadata:
        labels:
        app: crop-face
    spec:
        containers:
        - name: crop-face
        image: santoshburada/crop_face:latest_multiArch
        ports:
        - containerPort: 8001
        envFrom:  # Use all keys in the secret as environment variables
        - secretRef:
            name: my-env-secret
---
apiVersion: v1
kind: Service
metadata:
    name: crop-face-service
    namespace: facerec
spec:
    type: NodePort
    ports:
    - port: 8001
    targetPort: 8001
    nodePort: 30001
    selector:
    app: crop-face
---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: train-face-deployment
    namespace: facerec
spec:
    replicas: 3
    selector:
    matchLabels:
        app: train-face
    template:
    metadata:
        labels:
        app: train-face
    spec:
        containers:
        - name: train-face
        image: santoshburada/train_face:latest-multiArch
        ports:
        - containerPort: 8003
        envFrom:  # Use all keys in the secret as environment variables
        - secretRef:
            name: my-env-secret
        
---
apiVersion: v1
kind: Service
metadata:
    name: train-face-service
    namespace: facerec
spec:
    type: NodePort
    ports:
    - port: 8003
    targetPort: 8003
    nodePort: 30003
    selector:
    app: train-face
---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: face-rec-deployment
    namespace: facerec
spec:
    replicas: 3
    selector:
    matchLabels:
        app: face-rec
    template:
    metadata:
        labels:
        app: face-rec
    spec:
        containers:
        - name: face-rec
        image: santoshburada/face_rec:amd64
        ports:
        - containerPort: 8005
        envFrom:  # Use all keys in the secret as environment variables
        - secretRef:
            name: my-env-secret
---
apiVersion: v1
kind: Service
metadata:
    name: face-rec-service
    namespace: facerec
spec:
    type: NodePort
    ports:
    - port: 8005
    targetPort: 8005
    nodePort: 30005
    selector:
    app: face-rec
```
#### Save the above file and use below command
```bash
kubectl apply -f <your-file-name>
```
#### Now before running the client files please change the url's of each client file with the respective services in the k8s cluster
```bash
The url must be in the below format
http://<your-laptop-ip-address>:<NodePort of the services>/path

# below is the example change the IP address wih your Ip

1) For Mtcnn which listens on NodePort 30001 in the k8s cluster
    send_url = 'http://<Ip-address>:30001/crop_face_mtcnn'
2) For traning which listens on NodePort 30003 in the k8s cluster
    send_url = 'http://<Ip-address>:30003/process_images'
3) For Recognition which listens on NodePort 30005 in the k8s cluster
    send_url = 'http://<Ip-address>:30003/recognize'
```

</details>

### Kubernetes Cloud Deployment
<details>
<summary>Click to expand!</summary>

#### Setting up on Cloud Providers. Below is example for AWS, same can be followed in any other Cloud Providers with few minor changes in the commands**
First configure the EKS cluster. Navigate to AWS Cli and paste below command
```bash
eksctl create cluster --name=final-cluster --region=us-east-1 --nodegroup-name=final-cluster-nodegroup --node-type=t3.2xlarge --nodes=3 --nodes-min=3 --nodes-max=5 --managed
```
The above command will create a 3 managed worker nodes EKS Cluster. I used a t3.2xlarge for better performace and speed. I suggest minumum 2 cores of cpu. It will take atleast 15-30 minitues

#### If eksctl is not installed in the AWS Cli, execute below commands
```bash
# for ARM systems, set ARCH to: `arm64`, `armv6` or `armv7`
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH

curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

# (Optional) Verify checksum
curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz

sudo mv /tmp/eksctl /usr/local/bin
```

Execute below command in the AWS Cli
```bash
kubectl apply -f https://github.com/santosh-burada/FaceRecog-API/blob/master/Deploy.yaml
```
#### Check if there is an IAM OIDC provider configured already
```bash
aws iam list-open-id-connect-providers | grep $oidc_id | cut -d "/" -f4\n
```
If not, run the below command
```bash
eksctl utils associate-iam-oidc-provider --cluster final-cluster --approve
```

#### Setup Application Load Balancer  add on

#### Download IAM policy

```bash
curl -O https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.5.4/docs/install/iam_policy.json
```

#### Create IAM Policy

```bash
aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document file://iam_policy.json
```

#### Create IAM Role and don't forget to add `your-aws-account-id` in the below command

```bash
eksctl create iamserviceaccount \
    --cluster=final-cluster \
    --namespace=kube-system \
    --name=aws-load-balancer-controller \
    --role-name AmazonEKSLoadBalancerControllerRole \
    --attach-policy-arn=arn:aws:iam::<your-aws-account-id>:policy/AWSLoadBalancerControllerIAMPolicy \
    --approve
```

### Deploy ALB controller

Add helm repo

```bash
helm repo add eks https://aws.github.io/eks-charts
```

Update the repo

```bash
helm repo update eks
```

#### Install aws alb in the cluster. Add you `vpcId` of the cluster which is under Networking section of the cluster.

```bash
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \            
    -n kube-system \
    --set clusterName=final-cluster \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller \
    --set region=us-east-1 \
    --set vpcId=<your-vpc-id>
```

#### Verify that the deployments are running.

```bash
kubectl get deployment -n kube-system aws-load-balancer-controller
```

#### wait until the deployments are running. Once done execute below command to get the exposed domain to access the services inside the cluster.
```bash
kubectl get ingress ingress-facerec -n facerec
```
#### Now copy the value under `Address` column and replace your `IP's` and `No need of ports` in the client.py files
```bash
# example format
http://<AWS-Address>/crop_face_mtcnn
http://<AWS-Address>/process_image
http://<AWS-Address>/recognize
```

# Finally Don't forgot to Delete the cluster if necessary
</details>

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
For support, collaboration, or issues, please email at [santu.burada99@gmail.com](mailto:santu.burada99@gmail.com).

---

Thank you for visiting the project!
