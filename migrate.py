from app import db, Attendance
from sqlalchemy import Column, String
from datetime import datetime

# Add new columns dynamically (if not exists)
with db.engine.connect() as conn:
    # Add employee_id column
    try:
        conn.execute('ALTER TABLE attendance ADD COLUMN employee_id TEXT')
    except:
        print("employee_id column already exists.")

    # Add status column
    try:
        conn.execute('ALTER TABLE attendance ADD COLUMN status TEXT')
    except:
        print("status column already exists.")

# Update existing rows with defaults
with db.session.begin():
    all_records = Attendance.query.all()
    for rec in all_records:
        # Extract employee_id from "ID - Name"
        if " - " in rec.name:
            emp_id = rec.name.split(" - ")[0]
        else:
            emp_id = "UNKNOWN"

        rec.employee_id = emp_id
        rec.status = "IN"  # default
        db.session.add(rec)

db.session.commit()
print("Migration completed.")
