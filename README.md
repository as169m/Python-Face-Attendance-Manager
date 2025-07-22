AI Face Recognition Attendance System
This project is a web-based attendance management system that uses face recognition for marking attendance.
It leverages Flask (Python) for backend APIs, face-api.js for real-time face detection on the frontend, and a SQL database to store attendance logs.

Features
Face Recognition Login

Detects and recognizes employee faces using face-api.js.

Works directly in the browser (no server-side face detection delay).

Attendance Marking

Marks IN and OUT entries with a minimum gap of 10 minutes.

Captures only one IN and OUT per day.

Stores data in the database with IST (Indian Standard Time).

Employee Management

Register employees with photo capture (via webcam).

Saves employee images in static/employees/{employee_id_name} folder.

Generates face embeddings and stores them in static/known_faces.json for faster recognition.

Attendance Summary

Date-wise filter and search options.

Displays total time spent and total break time (sum of OUT-IN durations per day).

Mobile-Friendly UI

Full-screen camera view with live face detection and bounding boxes.

Status message (e.g., "John Doe Detected, Marking Attendance...").

Tech Stack
Backend: Python (Flask)

Frontend: HTML, CSS, JavaScript, face-api.js

Database: SQLAlchemy (SQLite/MySQL)

Image Processing: face_recognition (Dlib)

Templates: Jinja2 (Flask base.html)

Timezone: pytz for IST support

Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/as169m/Python-Face-Attendance-Manager.git
cd face-attendance-system
2. Create a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the application
bash
Copy
Edit
python app.py
Open http://127.0.0.1:5000 in your browser.

Project Structure
perl
Copy
Edit
face-attendance-system/
│
├── app.py                   # Flask backend
├── models.py                # Database models
├── static/
│   ├── employees/           # Employee folders with photos
│   ├── known_faces.json     # Face embeddings
│   └── js/
│       └── face-api.min.js  # Face recognition frontend
├── templates/
│   ├── base.html            # Common layout
│   ├── mark_attendance.html # Live camera and attendance marking
│   ├── attendance_summary.html
│   └── register_employee.html
├── requirements.txt
└── README.md
Key Endpoints
GET / – Dashboard / Home

GET /mark_attendance – Live camera attendance marking

POST /api/mark_attendance – Backend API to mark attendance

GET /attendance_summary – Attendance reports

GET|POST /register_employee – Register new employee

Future Improvements
Add admin dashboard with charts.

Export attendance data as Excel/CSV.

Integrate push notifications for attendance.

Multi-camera and multi-location support.

License
This project is licensed under the MIT License.
