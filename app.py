import os
import csv
import json
import base64
import shutil
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from PIL import Image
from io import BytesIO, StringIO
from flask import Flask, render_template, Response, request, redirect, url_for, session, Response as FlaskResponse
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask import jsonify
from flask import render_template
from sqlalchemy import func
from datetime import datetime, timedelta
import io
import openpyxl
from flask import send_file
import pytz  # Add this
IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

# ------------------- Flask Config -------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"
db = SQLAlchemy(app)

UPLOAD_FOLDER = "static/employees"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Database Models -------------------

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # "ID - Name"
    employee_id = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(10), nullable=False)  # "IN" or "OUT"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(50), nullable=False)
    image_folder = db.Column(db.String(100), nullable=False)
    encoding_data = db.Column(db.Text, nullable=True)  # JSON of face encoding

# ------------------- Admin Credentials -------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ------------------- Helper Functions -------------------

def calculate_average_encoding(image_paths):
    encodings = []
    for img_path in image_paths:
        img = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodings.append(enc[0])
    if len(encodings) > 0:
        avg_encoding = np.mean(encodings, axis=0)
        return avg_encoding.tolist()
    return None

def load_known_faces():
    known_encodings = []
    known_names = []
    employees = Employee.query.all()
    for emp in employees:
        if emp.encoding_data:
            encoding = np.array(json.loads(emp.encoding_data))
            known_encodings.append(encoding)
            known_names.append(f"{emp.employee_id} - {emp.name}")
    return known_encodings, known_names


# Directory where your employee images are stored

OUTPUT_JSON = "static/known_faces.json"
EMPLOYEES_DIR = "static/employees"  # Adjust to your folder structure

def generate_embeddings():
    known_faces = []

    # Loop through each employee folder
    for emp_folder in os.listdir(EMPLOYEES_DIR):
        folder_path = os.path.join(EMPLOYEES_DIR, emp_folder)
        if not os.path.isdir(folder_path):
            continue

        json_path = os.path.join(folder_path, "data.json")
        if not os.path.exists(json_path):
            print(f"[WARNING] No data.json found in {emp_folder}, skipping.")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            employee_id = data.get("employee_id")
            name = data.get("name")
            encoding = data.get("encoding")

            if not encoding:
                print(f"[WARNING] No encoding found for {emp_folder}.")
                continue

            # Store for known faces
            known_faces.append({
                "name": f"{employee_id}_{name}",
                "descriptor": encoding
            })

            print(f"Loaded encoding for {employee_id} - {name}")

        except Exception as e:
            print(f"[ERROR] Failed to load {json_path}: {e}")

    # Save all encodings to known_faces.json
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(known_faces, f, indent=4)

    print(f"✅ Embeddings saved to {OUTPUT_JSON}")

def refresh_face_encodings():
    """
    Regenerate known_faces.json from all employee data.json files
    and reload face encodings into memory.
    """
    print("🔄 Refreshing face encodings...")
    generate_embeddings()  # Regenerate known_faces.json

    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    try:
        with open(OUTPUT_JSON, "r") as f:
            data = json.load(f)
        
        for entry in data:
            name = entry.get("name")
            descriptor = entry.get("descriptor")
            if descriptor:
                known_face_encodings.append(np.array(descriptor))
                known_face_names.append(name)
        
        print(f"✅ Loaded {len(known_face_encodings)} known faces.")
    except Exception as e:
        print(f"❌ Failed to reload known faces: {e}")


def refresh_face_encodings():
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()

def mark_attendance(emp_full_name):
    today = datetime.now().strftime('%Y-%m-%d')
    existing_record = Attendance.query.filter(Attendance.name == emp_full_name, Attendance.timestamp.like(f"{today}%")).first()
    if not existing_record:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entry = Attendance(name=emp_full_name, timestamp=timestamp)
        db.session.add(new_entry)
        db.session.commit()

from datetime import datetime, timedelta

def can_mark_attendance(emp_id):
    now = datetime.now()
    today = now.date()

    # Fetch all today's entries for the employee
    today_entries = Attendance.query.filter(
        Attendance.employee_id == emp_id,
        Attendance.timestamp >= datetime.combine(today, datetime.min.time())
    ).order_by(Attendance.timestamp.desc()).all()

    # Separate IN and OUT entries
    in_entries = [entry for entry in today_entries if entry.status == "IN"]
    out_entries = [entry for entry in today_entries if entry.status == "OUT"]

    # Rule 1: If no entries yet, allow IN
    if not today_entries:
        return "IN", None

    last_entry = today_entries[0]

    # Ensure timestamp is a datetime object
    last_timestamp = last_entry.timestamp
    if isinstance(last_timestamp, str):  # Convert string to datetime
        try:
            last_timestamp = datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None, "Invalid timestamp format in attendance data"

    time_diff = now - last_timestamp

    # Rule 2: If already IN and no OUT yet, allow OUT (after 10 min)
    if last_entry.status == "IN":
        if len(out_entries) == 0:  # No OUT yet
            if time_diff < timedelta(minutes=10):
                return None, "Please wait at least 10 minutes after IN before marking OUT"
            return "OUT", None
        else:
            return None, "Attendance already completed for today"

    # Rule 3: If already OUT, don't allow another IN on the same day
    if last_entry.status == "OUT":
        return None, "Attendance already completed for today"

    return "IN", None

def check_attendance_rules(emp_id):
    """
    Checks the current attendance status of an employee
    and determines the next possible action (IN, OUT, or None).
    """
    now = datetime.now()
    today = now.date()

    today_entries = Attendance.query.filter(
        Attendance.employee_id == emp_id,
        Attendance.timestamp >= datetime.combine(today, datetime.min.time())
    ).order_by(Attendance.timestamp.desc()).all()

    in_entries = [entry for entry in today_entries if entry.status == "IN"]
    out_entries = [entry for entry in today_entries if entry.status == "OUT"]

    # Case 1: No entries yet
    if not today_entries:
        return {
            "status": "NOT_MARKED",
            "next_action": "IN",
            "message": "No attendance marked today."
        }

    last_entry = today_entries[0]
    last_timestamp = last_entry.timestamp

    # Convert string timestamps if needed
    if isinstance(last_timestamp, str):
        try:
            last_timestamp = datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return {
                "status": "ERROR",
                "next_action": None,
                "message": "Invalid timestamp format in attendance data."
            }

    time_diff = now - last_timestamp

    # Case 2: IN done but OUT not yet
    if last_entry.status == "IN" and len(out_entries) == 0:
        if time_diff < timedelta(minutes=10):
            return {
                "status": "IN_DONE",
                "next_action": None,
                "message": "Please wait at least 10 minutes before marking OUT."
            }
        return {
            "status": "IN_DONE",
            "next_action": "OUT",
            "message": "You can now mark OUT."
        }

    # Case 3: Both IN and OUT are done
    return {
        "status": "COMPLETED",
        "next_action": None,
        "message": "Attendance already completed for today."
    }


# ------------------- Webcam Video Stream -------------------
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        mark_attendance(name)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------- Context Processor -------------------
@app.context_processor
def inject_os():
    return dict(os=os)

# ------------------- Routes -------------------
@app.route('/')
def default():
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------- Login -----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            refresh_face_encodings()
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings_route():
    try:
        generate_embeddings()  # Your function that generates known_faces.json
        return jsonify({"status": "success", "message": "Embeddings generated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------- Dashboard -----------
@app.route('/dashboard')
def dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    date_filter = request.args.get('date')
    name_filter = request.args.get('name')

    query = Attendance.query
    if date_filter:
        query = query.filter(Attendance.timestamp.like(f"{date_filter}%"))
    if name_filter:
        query = query.filter(Attendance.name.contains(name_filter))

    records = query.order_by(Attendance.timestamp.desc()).all()

    # Today's Summary
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = Attendance.query.filter(Attendance.timestamp.like(f"{today}%")).all()

    summary = {}
    for rec in today_records:
        if rec.employee_id not in summary:
            summary[rec.employee_id] = {
                'employee_id': rec.employee_id,
                'name': rec.name,
                'in_count': 0,
                'out_count': 0
            }
        if rec.status == 'IN':
            summary[rec.employee_id]['in_count'] += 1
        elif rec.status == 'OUT':
            summary[rec.employee_id]['out_count'] += 1

    today_summary = list(summary.values())

    return render_template('dashboard.html',
                           records=records,
                           date_filter=date_filter or '',
                           name_filter=name_filter or '',
                           today_summary=today_summary)

@app.route("/live_attendance")
def api_live_attendance():
    if not session.get('admin_logged_in'):
        return jsonify([])

    latest_records = Attendance.query.order_by(Attendance.timestamp.desc()).limit(10).all()
    data = []
    for r in latest_records:
        data.append({
            "id": r.id,
            "employee_id": r.employee_id,
            "name": r.name,
            "status": r.status,
            "timestamp": r.timestamp
        })
    return jsonify(data)

@app.route('/last_attendance/<employee_id>', methods=['GET'])
def last_attendance(employee_id):
    try:
        last_record = Attendance.query.filter_by(employee_id=employee_id)\
                                      .order_by(Attendance.timestamp.desc())\
                                      .first()
        if not last_record:
            return jsonify({"message": "No attendance found"}), 200
        
        return jsonify({
            "status": last_record.status,
            "name": last_record.name,
            "timestamp": last_record.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/export_csv')
def export_csv():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['ID', 'Name', 'Timestamp'])
    for r in records:
        writer.writerow([r.id, r.name, r.timestamp])
    output = si.getvalue()
    return FlaskResponse(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=attendance.csv"})

# ----------- Employee Management -----------
@app.route('/mark_attendance_api', methods=['POST'])
def mark_attendance_endpoint():
    if not request.is_json:
        return jsonify({"message": "Invalid request"}), 400

    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"message": "No image received"}), 400

    # Decode the image
    try:
        image_data = image_data.split(",")[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"message": f"Image decode error: {str(e)}"}), 400

    # Detect faces
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_img)

    if not face_encodings:
        return jsonify({"message": "No face detected"}), 200

    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    if len(face_distances) == 0 or not any(matches):
        return jsonify({"message": "Unknown face"}), 200

    # Find best match
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        emp_full_name = known_face_names[best_match_index]
        emp_id, emp_name = emp_full_name.split(" - ", 1)

        # Check attendance rules
        rule_check = check_attendance_rules(emp_id)
        if rule_check["next_action"] is None:
            return jsonify({"message": rule_check["message"]}), 200

        # Save attendance
        new_record = Attendance(
            employee_id=emp_id,
            name=emp_name,
            status=rule_check["next_action"],
            timestamp=datetime.now(IST)            
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({"message": f"{emp_name} marked as {rule_check['next_action']}"}), 200

    return jsonify({"message": "No match found"}), 200

@app.route('/api/mark_attendance', methods=['POST'])
def api_mark_attendance():
    if not request.is_json:
        return jsonify({"message": "Invalid request"}), 400

    data = request.get_json()
    emp_id = data.get("employee_id")
    emp_name = data.get("employee_name")

    if not emp_id or not emp_name:
        return jsonify({"message": "Missing employee_id or employee_name"}), 400

    # Check last attendance entry for today
    last_entry = Attendance.query.filter_by(employee_id=emp_id) \
        .order_by(Attendance.timestamp.desc()) \
        .first()

    now = datetime.now()
    new_status = "IN"

    if last_entry:
        # Ensure at least 10 minutes gap between IN and OUT
        if (now - last_entry.timestamp) < timedelta(minutes=10):
            return jsonify({"message": "Wait at least 10 minutes before next mark"}), 200

        # Toggle IN/OUT
        if last_entry.status == "IN":
            new_status = "OUT"

    # Save new attendance record
    new_record = Attendance(
        employee_id=emp_id,
        name=emp_name,
        status=new_status,
        timestamp=now
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({"message": f"{emp_name} marked as {new_status}"}), 200


@app.route('/mark_attendance', methods=['GET'])
def mark_attendance_page():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    return render_template('mark_attendance_new.html')

@app.route('/employees')
def employees():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    employee_list = Employee.query.all()
    return render_template('employees.html', employees=employee_list)

@app.route('/edit_employee/<int:emp_id>', methods=['GET', 'POST'])
def edit_employee(emp_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    emp = Employee.query.get_or_404(emp_id)
    if request.method == 'POST':
        new_emp_id = request.form['employee_id']
        new_name = request.form['name']

        # Rename folder if ID or Name changes
        old_folder = os.path.join(UPLOAD_FOLDER, emp.image_folder)
        new_folder_name = f"{new_emp_id}_{new_name}".replace(" ", "_")
        new_folder = os.path.join(UPLOAD_FOLDER, new_folder_name)
        if emp.image_folder != new_folder_name and os.path.exists(old_folder):
            os.rename(old_folder, new_folder)

        emp.employee_id = new_emp_id
        emp.name = new_name
        emp.image_folder = new_folder_name
        db.session.commit()
        refresh_face_encodings()
        return redirect(url_for('employees'))

    return render_template('edit_employee.html', emp=emp)

@app.route('/delete_employee/<int:emp_id>')
def delete_employee(emp_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    emp = Employee.query.get_or_404(emp_id)
    folder_path = os.path.join(UPLOAD_FOLDER, emp.image_folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    db.session.delete(emp)
    db.session.commit()
    refresh_face_encodings()
    return redirect(url_for('employees'))

@app.route('/employee_attendance/<int:emp_id>')
def employee_attendance(emp_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    emp = Employee.query.get_or_404(emp_id)

    # Fetch attendance by employee_id ordered by timestamp
    records = (
        Attendance.query
        .filter(Attendance.employee_id == emp.employee_id)
        .order_by(Attendance.timestamp.asc())
        .all()
    )

    # Group by date and compute first IN, last OUT, total hours
    daily_summary = {}
    for r in records:
        date_key = r.timestamp.date()
        if date_key not in daily_summary:
            daily_summary[date_key] = {
                "in_time": None,
                "out_time": None,
                "worked_seconds": 0
            }

        # Set first IN
        if r.status == "IN" and not daily_summary[date_key]["in_time"]:
            daily_summary[date_key]["in_time"] = r.timestamp

        # Set last OUT
        if r.status == "OUT":
            daily_summary[date_key]["out_time"] = r.timestamp
            # Calculate duration if there was an IN before this OUT
            if daily_summary[date_key]["in_time"]:
                worked_time = r.timestamp - daily_summary[date_key]["in_time"]
                daily_summary[date_key]["worked_seconds"] = int(worked_time.total_seconds())

    # Convert summary to list
    summary_list = []
    for date_key, data in daily_summary.items():
        hours = data["worked_seconds"] // 3600
        minutes = (data["worked_seconds"] % 3600) // 60
        seconds = data["worked_seconds"] % 60
        total_hours = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if data["worked_seconds"] > 0 else "—"

        summary_list.append({
            "date": date_key,
            "in_time": data["in_time"],
            "out_time": data["out_time"],
            "total_hours": total_hours
        })

    # Sort by date (latest first)
    summary_list.sort(key=lambda x: x["date"], reverse=True)

    return render_template('employee_attendance.html', emp=emp, summary=summary_list)



@app.route('/attendance_status', methods=['POST'])
def attendance_status():
    if not request.is_json:
        return jsonify({"message": "Invalid request"}), 400

    data = request.get_json()
    emp_id = data.get("employee_id")

    if not emp_id:
        return jsonify({"message": "Missing employee_id"}), 400

    rule_check = check_attendance_rules(emp_id)
    return jsonify(rule_check), 200


# ----------- Employee Registration -----------
@app.route('/register_employee', methods=['GET', 'POST'])
def register_employee():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        employee_id = request.form['employee_id']
        name = request.form['name']
        photo_data = request.form.get('photo_data')  # Single photo

        if employee_id and name and photo_data:
            emp_folder = f"{employee_id}_{name}".replace(" ", "_")
            folder_path = os.path.join(UPLOAD_FOLDER, emp_folder)
            os.makedirs(folder_path, exist_ok=True)

            # Save image
            img_data = photo_data.split(",")[1]
            image_bytes = base64.b64decode(img_data)
            filename = f"{name}.jpg"
            filepath = os.path.join(folder_path, filename)
            image = Image.open(BytesIO(image_bytes))
            image.save(filepath)

            # Generate encoding
            avg_encoding = calculate_average_encoding([filepath])
            if avg_encoding:
                # Save to DB
                new_employee = Employee(
                    employee_id=employee_id,
                    name=name,
                    image_folder=emp_folder,
                    encoding_data=json.dumps(avg_encoding)
                )
                db.session.add(new_employee)
                db.session.commit()

                # ** Save JSON file in same folder **
                employee_json_path = os.path.join(folder_path, "data.json")
                with open(employee_json_path, "w") as json_file:
                    json.dump({
                        "employee_id": employee_id,
                        "name": name,
                        "encoding": avg_encoding
                    }, json_file, indent=4)

                # Refresh encodings for recognition
                refresh_face_encodings()
                
                return redirect(url_for('employees'))
            else:
                return render_template('register_employee.html', error="No face detected in captured photo.")
        else:
            return render_template('register_employee.html', error="Employee ID, Name, and photo are required.")

    return render_template('register_employee.html')

@app.route('/attendance_report', methods=['GET', 'POST'])
def attendance_report():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    employees = Employee.query.order_by(Employee.name).all()
    selected_date = request.args.get('date', '')
    selected_emp_id = request.args.get('employee_id', '')

    query = Attendance.query
    if selected_date:
        query = query.filter(Attendance.timestamp.like(f"{selected_date}%"))
    if selected_emp_id:
        emp = Employee.query.get(int(selected_emp_id))
        if emp:
            full_name = f"{emp.employee_id} - {emp.name}"
            query = query.filter(Attendance.name == full_name)

    records = query.order_by(Attendance.timestamp.desc()).all()

    if 'export' in request.args:
        # Export CSV
        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(['ID', 'Employee ID', 'Name', 'Status', 'Timestamp'])
        for r in records:
            writer.writerow([r.id, r.employee_id, r.name, r.status, r.timestamp])
        output = si.getvalue()
        return FlaskResponse(output, mimetype="text/csv",
                             headers={"Content-Disposition": "attachment;filename=attendance_report.csv"})

    return render_template('attendance_report.html',
                           employees=employees,
                           selected_date=selected_date,
                           selected_emp_id=selected_emp_id,
                           records=records)

@app.route('/attendance_summary')
def attendance_summary():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    search = request.args.get('search', '')

    query = Attendance.query
    if start_date:
        query = query.filter(Attendance.timestamp >= start_date)
    if end_date:
        query = query.filter(Attendance.timestamp <= end_date)
    if search:
        query = query.filter(or_(
            Attendance.name.ilike(f'%{search}%'),
            Attendance.employee_id.ilike(f'%{search}%')
        ))

    records = query.order_by(Attendance.timestamp.asc()).all()

    # Group by day and employee
    from collections import defaultdict
    attendance_by_day = defaultdict(lambda: defaultdict(list))
    for rec in records:
        day = rec.timestamp.strftime('%Y-%m-%d')
        attendance_by_day[day][rec.employee_id].append(rec)

    return render_template(
        'attendance_summary.html',
        attendance_by_day=attendance_by_day,
        start_date=start_date,
        end_date=end_date
    )

@app.route('/export_attendance_excel')
def export_attendance_excel():
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    records = (
        db.session.query(Attendance.employee_id, Attendance.name, Attendance.status, Attendance.timestamp)
        .filter(Attendance.timestamp >= today_start, Attendance.timestamp < today_end)
        .order_by(Attendance.employee_id, Attendance.timestamp.asc())
        .all()
    )

    summary = {}
    for r in records:
        if r.employee_id not in summary:
            summary[r.employee_id] = {
                "employee_id": r.employee_id,
                "name": r.name,
                "in_time": None,
                "out_time": None
            }
        if r.status == "IN" and not summary[r.employee_id]["in_time"]:
            summary[r.employee_id]["in_time"] = r.timestamp
        elif r.status == "OUT":
            summary[r.employee_id]["out_time"] = r.timestamp

    # Create Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance Summary"

    # Header
    ws.append(["Employee ID", "Employee Name", "IN Time", "OUT Time", "Total Hours"])

    # Data rows
    for emp in summary.values():
        total_hours = "—"
        if emp["in_time"] and emp["out_time"]:
            time_diff = emp["out_time"] - emp["in_time"]
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            total_hours = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        ws.append([
            emp["employee_id"],
            emp["name"],
            emp["in_time"].strftime("%H:%M:%S") if emp["in_time"] else "—",
            emp["out_time"].strftime("%H:%M:%S") if emp["out_time"] else "—",
            total_hours
        ])

    # Save to in-memory file
    file_stream = io.BytesIO()
    wb.save(file_stream)
    file_stream.seek(0)

    return send_file(
        file_stream,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"attendance_summary_{today_start.strftime('%Y-%m-%d')}.xlsx"
    )


# ------------------- Main -------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        refresh_face_encodings()
    app.run(host="0.0.0.0", port=5001, debug=True)
