:: This file is used to generate requirements.txt for all the projects
:: first we activate the virtual environment and then cd to the project directory and then generate the requirements.txt file

call face-detector/detector/Scripts/activate
cd face-detector
pip freeze > requirements.txt
cd ..

call MTCNN-detector/mtsor/Scripts/activate
cd MTCNN-detector
pip freeze > requirements.txt
cd ..
dir

call myenv/Scripts/activate
pip freeze > requirements.txt

