from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from pymongo import MongoClient, errors
from wtforms import StringField,PasswordField,SubmitField, validators
from wtforms.validators import DataRequired, Email, ValidationError
import jwt
import os
from pydantic import BaseModel
import datetime
from passlib.context import CryptContext
import re
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
def connect_to_mongodb_atlas(connection_string, database_name):
    """Connect to MongoDB Atlas and return the database object."""
    try:
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)  # 5-second timeout
        db = client[database_name]
        db.command("ping")  # Quick operation to check the connection
        print("Connected successfully to MongoDB Atlas.")
        return db
    except errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        exit()

# MongoDB setup
connection_string = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')

# Secret key for JWT
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') # openssl rand -hex 32

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

class RegisterForm(FlaskForm):
    name = StringField("Name",validators=[DataRequired()])
    phone = StringField("Phone",validators=[DataRequired()])
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('repassword', message='Passwords must match')
    ])
    repassword = PasswordField("Repeat Password",validators=[DataRequired()])
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Login")

db = connect_to_mongodb_atlas(connection_string, database_name) 

@app.route('/')
def home():
    return 'Welcome to the Authentication API'

@app.route('/signup', methods=['POST','GET'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data
        phone= form.phone.data
        repassword = form.repassword.data

        if password!=repassword:
            flash("Password must match")
            return render_template("signup.html", form=form, message="Password must match")
        else:
            hashed_password = get_password_hash(password)

        users = db.users
        # Check if user already exists by username or email
        if users.find_one({'email': email}):
            return render_template("signup.html", form=form, message="Username or Email already exists")

        # Optional: Validate phone, password strength, etc.
        # Example: Validate password strength (customize according to your requirements)
        if not re.match(r'[A-Za-z0-9@#$%^&+=]{8,}', password):
            # flash("Password should be at least 8 characters long and include numbers and special characters.")
            return render_template("signup.html", form=form, message="Password should be at least 8 characters long and include numbers and special characters.")

        # Insert new user
        users.insert_one({'username': name, 'password': hashed_password, 'email': email, 'phone': phone})
        flash("You have successfully registered! Please login.")
        return redirect(url_for('login'))
    return render_template("signup.html", form=form)

@app.route('/login', methods=['POST','GET'])
def login():
    form = LoginForm()
    token=""
    if form.validate_on_submit():
        users = db.users
        email = form.email.data
        password = form.password.data

        user = users.find_one({'email': email})

        if not user:
            return render_template("login.html", form=form, message="Invalid credentials")

        # Verify the password with the hashed password in the database
        if not verify_password(password, user['password']):
            return render_template("login.html", form=form, message="Inavalid password")


        # Generate JWT token
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expires in 1 hour
        }, app.config['SECRET_KEY'])

        # Insert or update token in database
        users.update_one({'_id': user['_id']}, {'$set': {'token': token}})
        return render_template("login.html", form=form, message="Login Successful, Token Updated")

    return render_template("login.html",form=form)


if __name__ == '__main__':
    app.run(debug=True)