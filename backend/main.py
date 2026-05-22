from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. تهيئة التطبيق وتوثيقه
app = FastAPI(
    title="Heart Disease Advanced AI API",
    description="النظام الخلفي لـ V3: محرك تشخيص أمراض القلب باستخدام معمارية TabNet والذكاء الاصطناعي القابل للتفسير.",
    version="3.0.0"
)

# 2. حماية الـ CORS (ضرورية جداً لربط واجهة React لاحقاً بدون مشاكل أمنية)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في بيئة الإنتاج الحقيقية سنحدد رابط الـ React فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. بناء نموذج البيانات الطبي (Patient Schema) للتحقق التلقائي من المدخلات
class PatientInput(BaseModel):
    age: int = Field(..., description="العمر بالسنوات", ge=0, le=120)
    sex: int = Field(..., description="الجنس: 1 للذكور، 0 للإناث", ge=0, le=1)
    chest_pain_type: int = Field(..., description="نوع ألم الصدر (0-3)", ge=0, le=3)
    resting_bp: int = Field(..., description="ضغط الدم الانقباضي في الراحة (mm Hg)", ge=50, le=250)
    cholesterol: int = Field(..., description="مستوى الكولسترول في الدم (mg/dl)", ge=0, le=600)
    fasting_bs: int = Field(..., description="سكر الدم الصائم > 120 (1=نعم، 0=لا)", ge=0, le=1)
    resting_ecg: int = Field(..., description="نتائج تخطيط القلب في الراحة (0-2)", ge=0, le=2)
    max_hr: int = Field(..., description="أقصى نبض للقلب تم تسجيله", ge=60, le=220)
    exercise_angina: int = Field(..., description="ذبحة صدرية ناتجة عن المجهود (1=نعم، 0=لا)", ge=0, le=1)
    oldpeak: float = Field(..., description="انخفاض مقطع ST الناتج عن المجهول")
    st_slope: int = Field(..., description="ميل مقطع ST في ذروة المجهود (0-2)", ge=0, le=2)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 54, "sex": 1, "chest_pain_type": 0, "resting_bp": 140,
                "cholesterol": 289, "fasting_bs": 0, "resting_ecg": 1, "max_hr": 122,
                "exercise_angina": 0, "oldpeak": 0.0, "st_slope": 1
            }
        }

# 4. مسار فحص النظام (Health Check)
@app.get("/", tags=["System"])
def health_check():
    return {
        "status": "Healthy",
        "database_status": "Ready for Harmonization",
        "model_version": "V3-TabNet (Pending Training)"
    }

# 5. مسار استقبال البيانات والتشخيص
@app.post("/api/v3/predict", tags=["Clinical Diagnosis"])
def predict_heart_disease(patient: PatientInput):
    # هنا سيتم لاحقاً استدعاء معمارية الـ TabNet المدمجة لتعطي القرار وتفسير SHAP
    # حالياً نضع استجابة تجريبية للتأكد من سلامة تدفق البيانات
    return {
        "status": "Success",
        "received_data": patient.dict(),
        "clinical_note": "Backend pipeline is fully operational. TabNet brain connection is next."
    }