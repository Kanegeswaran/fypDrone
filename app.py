from flask import Flask,render_template,Response, request, redirect, url_for, session, make_response, jsonify, send_file
from werkzeug.serving import make_server
import cv2
import os
import signal
import torch
from time import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import MySQLdb as mysql
import json
import hashlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image
import io


app=Flask(__name__)
app.secret_key = "Plastic Detection Using Drone"
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=5)

db = mysql.connect(
        host="localhost",
        user="kanag",
        password="@Dm1n123",
        database="bonds"
    )


cv2.cuda.setDevice(0)
yolov5_path = os.path.join(os.getcwd(), 'yolov5')
model_path = os.path.join(os.getcwd(), 'yolov5', 'trained_model', 'best.pt')
model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local')


def generate_frames(camera, flight_id):
    # camera=cv2.VideoCapture(0)
    # camera=cv2.VideoCapture("rtmp://10.164.38.255:1935/live")
    cur = db.cursor()
    try:
        if not camera.isOpened():
            raise Exception('Video capture could not be opened!')   
    except Exception as e:
        print(e)
        custom_shutdown()
    while True:
        ## read the camera frame
        ret,frame=camera.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame, dsize=(640,640))
            results = model(frame)
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            count += 1
            if(count == 1):
                start_time = time()
            elif (count == 8 or count == 16 or count == 24):
                for i in range(n):
                    row = cord[i]
                    if row[4] >= 0.5:
                        x1,  y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                        img = frame[y1:y2, x1:x2]
                        bgr = (0, 255, 0)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                        pred_class =  model.names[int(labels[i])]
                        cv2.putText(frame, pred_class + ' : '+ str(round(row.cpu().numpy()[4]*100,2)), 
                                    (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
                        if pred_class=='Plastic bag':
                            cur.execute('INSERT INTO plastic_bag (flight_Id, bag_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            # cur.execute('SELECT LAST_INSERT_ID()')
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into bag table")
                        elif pred_class=='Plastic bottle' :
                            cur.execute('INSERT INTO plastic_bottle (flight_Id, bottle_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into bottle table")
                        elif pred_class=='Plastic cup' :
                            cur.execute('INSERT INTO plastic_cup (flight_Id, cup_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into cup table")
            elif (count >= 25):
                count = 0
                end_time = time()
                fps = 24/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_2(camera, flight_id):
    cur = db.cursor()

    count = 0
    fps = 0
    while True:
        ## read the camera frame
        ret,frame=camera.read()
        if not ret:
            break
        else:
            cuda_frame = cv2.cuda_GpuMat()
            cuda_frame.upload(frame)
            count += 1
            if(count == 1):
                start_time = time()
            elif (count == 8 or count == 16 or count == 24):
                # Resize frame on GPU
                dsize = (640, 640)
                cuda_resized_frame = cv2.cuda.resize(cuda_frame, dsize)
                # Download the resized frame to CPU memory
                resized_frame = cuda_resized_frame.download()

                results = model(resized_frame)
                labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
                n = len(labels)
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                for i in range(n):
                    row = cord[i]
                    if row[4] >= 0.5:
                        x1,  y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                        img = frame[y1:y2, x1:x2]
                        bgr = (0, 255, 0)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 1)
                        pred_class =  model.names[int(labels[i])]
                        cv2.putText(frame, pred_class + ' : '+ str(round(row.cpu().numpy()[4]*100,2)), 
                                    (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
                        if pred_class=='Plastic bag':
                            cur.execute('INSERT INTO plastic_bag (flight_Id, bag_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            # cur.execute('SELECT LAST_INSERT_ID()')
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into bag table")
                        elif pred_class=='Plastic bottle' :
                            cur.execute('INSERT INTO plastic_bottle (flight_Id, bottle_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into bottle table")
                        elif pred_class=='Plastic cup' :
                            cur.execute('INSERT INTO plastic_cup (flight_Id, cup_pic) VALUES (%s,%s)', (flight_id, img_bytes,))
                            db.commit()
                            last_id = cur.lastrowid
                            print(last_id, " ", cur.rowcount, "inserted into cup table")
            elif (count >= 25):
                count = 0
                end_time = time()
                fps = 24/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def custom_shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    print("Shutting down!!!")


@app.route("/")
def index():
    return render_template('index2.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        cur = db.cursor()
        msg=''
        if request.method=='POST' and 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            login_type = request.form['login_type']
            hashed_password = hashlib.md5(password.encode()).hexdigest()
            query = 'SELECT * FROM `%s` WHERE username=\'%s\' AND password=\'%s\'' % (login_type,username,hashed_password)
            cur.execute(query)
            record = cur.fetchone()
            if record:
                session['loggedin']= True
                session['login_type'] = login_type
                session['username']= record[0]
                session.permanent = True
                return redirect('/home')
            else:
                msg='Incorrect username/password.Try again!'
        cur.close()
        return render_template('index2.html',msg=msg)
    except Exception as e:
        print(e)
        # custom_shutdown()


@app.route('/logout')
def logout():
    session.pop('loggedin',None)
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/admin', defaults={'page': 1, 'err_msg': ''}, methods=['GET', 'POST'])
@app.route('/admin/<int:page>', defaults={'err_msg': ''}, methods=['GET', 'POST'])
@app.route('/admin/<int:page>/<err_msg>', methods=['GET', 'POST'])
def adminpg(page, err_msg):
    if 'loggedin' not in session:
        return redirect(url_for("login"))
    per_page = 15  # Number of users per page
    cur = db.cursor()
    query = 'SELECT * FROM `members` LIMIT %s OFFSET %s'
    offset = (page - 1) * per_page
    cur.execute(query, (per_page, offset))
    users = cur.fetchall()
    cur.close()
    return render_template('adminpg.html', users=users, page=page, err_msg=err_msg)

@app.route('/delete_users', methods=['GET', 'POST'])
def delete_users():
    username_to_delete = request.form.getlist('usernames')
    cur = db.cursor()
    for username in username_to_delete:
        query = 'DELETE FROM `members` WHERE username=\'%s\'' % username
        cur.execute(query)
        db.commit()
    cur.close()
    return redirect(url_for('adminpg'))

@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    try:
        new_username = request.form.get('new_username')
        new_password = request.form.get('new_password')
        new_login_type = request.form['new_login_type']
        hashed_new_password = hashlib.md5(new_password.encode()).hexdigest()
        cur = db.cursor()
        query = 'INSERT INTO `%s` (username,password) VALUES (\'%s\', \'%s\')' % (new_login_type, new_username, hashed_new_password)
        cur.execute(query)
        db.commit()
        cur.close()
    except mysql.IntegrityError as e:
        return redirect(url_for('adminpg', err_msg="There's already an existing account", page=1))
    return redirect(url_for('adminpg'))

@app.route('/edit_profile', defaults={'err_msg': ''}, methods=['GET', 'POST'])
@app.route('/edit_profile/<err_msg>', methods=['GET', 'POST'])
def edit_profile(err_msg):
    return render_template('edit_profile.html', err_msg=err_msg)

@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    try:
        username = request.form['username']
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        if password != password_confirm:
            return redirect(url_for('edit_profile', err_msg="Passwords do not match"))

        hashed_password = hashlib.md5(password.encode()).hexdigest()
        query = 'UPDATE `%s` SET `username`=\'%s\',`password`=\'%s\' WHERE `username`=\'%s\'' % (session['login_type'], username, hashed_password,session['username'])
        cur = db.cursor()
        cur.execute(query)
        db.commit()
        cur.close()
        session['username'] = username
    except mysql.IntegrityError as e:
        return redirect(url_for('edit_profile', err_msg="There's already an existing account with same username"))
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'loggedin' not in session:
        return redirect(url_for("login"))
    return render_template('home.html')


@app.route('/video/<path:rtmpURL>')
def video(rtmpURL):
    camera=cv2.VideoCapture(rtmpURL)

    try:
        if camera.isOpened():
            cur = db.cursor()
            cur.execute('INSERT INTO `flight` VALUES ()')
            db.commit()
            flight_id = cur.lastrowid   
            return Response(generate_frames_2(camera, flight_id),mimetype='multipart/x-mixed-replace; boundary=frame')
        # raise ValueError('Video capture could not be opened!')
    except:
        db.rollback()
        return None
        

@app.route('/report')
def report():
    if 'loggedin' not in session:
        return redirect(url_for("login"))

    # show the form, it wasn't submitted
    return render_template('report2.html')

@app.route('/data/<table>')
def data(table):
    cur = db.cursor()
    query = 'SELECT max(`flight_Id`) FROM `flight`'
    cur.execute(query)
    flight_Id = cur.fetchone()[0]
    print(flight_Id)
    query = 'SELECT timestamp FROM `%s` WHERE flight_Id=\'%s\'' % (table, flight_Id)
    cur.execute(query)
    results = cur.fetchall()

    # Count items per timestamp
    counts = {}
    for result in results:
        print(result)
        timestamp = result[0].strftime("%H:%M:%S")
        counts[timestamp] = counts.get(timestamp, 0) + 1

    timestamp = list(counts.keys())
    item_count = list(counts.values())
    total = sum(item_count)

    return jsonify({'timestamp': timestamp, 'item_count': item_count, 'total': total})

def convert_blob_to_image(blob_data):
    image_stream = io.BytesIO(blob_data)
    image = Image.open(image_stream)
    temp_image_path = "temp_image.png"  # Temporary file path
    image.save(temp_image_path, "PNG")
    return temp_image_path, image.size

def draw_title_and_headers(canvas, title, headers, start_x, start_y, header_gap):
    canvas.setFont("Helvetica-Bold", 16)
    canvas.setFillColor(colors.darkblue)
    canvas.drawString(start_x, start_y, title)

    canvas.setFont("Helvetica-Bold", 12)
    canvas.setFillColor(colors.black)
    y = start_y - header_gap
    for i, header in enumerate(headers):
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(2)
        canvas.drawString(start_x + i * 200, y, header)
        canvas.line(start_x + i * 200 - 5, y-5, start_x + i * 200 - 5, y+15) 
    canvas.setStrokeColor(colors.darkblue)
    canvas.setLineWidth(2)
    canvas.line(start_x-5, y-5, 600, y-5)  # Adjust line length to match headers
    canvas.line(start_x-5, y+15, 600, y+15)
    canvas.line(600, y-5, 600, y+15)

    return y

def draw_vertical_lines(canvas, positions, start_y, end_y):
    canvas.setLineWidth(1)
    canvas.setStrokeColor(colors.grey)
    for pos in positions:
        canvas.line(pos, start_y, pos, end_y)
    canvas.line(positions[0], end_y, positions[-1], end_y)

def generate_pdf_with_data(table):
    if table == "plastic_bag":
        headers = ["Bag Count", "Bag Images", "Timestamp"]
        title = "Platic Bag Data Report"
    elif table == "plastic_bottle":
        headers = ["Bottle Count", "Images", "Timestamp"]
        title = "Platic Bottle Data Report"
    else:
        headers = ["Cup Count", "Cup Images", "Timestamp"]
        title = "Platic Cup Data Report"
    cur = db.cursor()
    query = 'SELECT max(`flight_Id`) FROM `flight`'
    cur.execute(query)
    flight_Id = cur.fetchone()[0]
    query = "SELECT * FROM `%s` WHERE flight_Id =\'%s\'" % (table, flight_Id)
    cur.execute(query)
    data = cur.fetchall()

    c = canvas.Canvas("Report.pdf", pagesize=letter)
    width, height = letter

    margin_bottom = 50  # Margin at the bottom of the page
    row_height = 200  # Adjust as needed
    header_height = 60
    start_y = height - 100
    start_x = 50
    column_positions = [45, 245, 445, 600]  # Adjusted for three columns

    # Function to draw headers and return new y_position
    def draw_headers(y_position):
        return draw_title_and_headers(c, title, headers, start_x, y_position, header_height)

    y_position = draw_headers(start_y)

    for index, (count, flight, pic, timestamp) in enumerate(data):
        if y_position - row_height < margin_bottom:  # Check if new page is needed
            c.showPage()
            y_position = draw_headers(height - 100)

        y_position -= row_height
        image_path, (img_width, img_height) = convert_blob_to_image(pic)
        image_midpoint = y_position + row_height / 2 - img_height / 2
        c.drawImage(image_path, start_x + 200, image_midpoint, img_width, img_height)
        c.drawString(start_x, image_midpoint + img_height / 2, str(count))
        c.drawString(start_x + 400, image_midpoint + img_height / 2, str(timestamp))
        draw_vertical_lines(c, column_positions, start_y - 65, y_position)

    c.save()

@app.route('/download-pdf/<table>')
def download_pdf(table):
    generate_pdf_with_data(table)  # Call the function to generate PDF
    return send_file("Report.pdf", as_attachment=True)

  
@app.route('/shutdown', methods=['GET'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        try:
            raise RuntimeError('Not running with the Werkzeug Server')
        except Exception as e:
            print(e)
            custom_shutdown()
    func()
    return 'Server shutting down...'

@app.after_request
def add_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
    return response

if __name__=="__main__":
    # app.run(host="0.0.0.0")

    # loop = asyncio.get_event_loop()
    # loop.create_task(generate_frames_2())
    app.run( host="0.0.0.0", port=5000, debug=True)