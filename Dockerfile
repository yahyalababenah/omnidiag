# استخدم نسخة بايثون خفيفة
FROM python:3.13-slim

# تحديد المجلد داخل الحاوية
WORKDIR /app

# نسخ ملف متطلبات المكتبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# الأمر الذي سيشغل مشروعك (استبدل main.py بالملف الرئيسي لديك)
CMD ["python", "main.py"]