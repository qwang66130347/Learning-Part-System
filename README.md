# แดชบอร์ด Personalized English Learning ด้วย Deep Q-Network (DQN)

## ภาพรวมโครงการ

แดชบอร์ดนี้เป็นส่วนหนึ่งของงานวิจัยเชิงวิชาการที่ศึกษาการใช้ **Deep Q-Network (DQN)** ในการแนะนำเส้นทางการเรียนภาษาอังกฤษแบบเฉพาะบุคคล ระบบใช้ Reinforcement Learning เพื่อเลือกกิจกรรมการเรียน (ฟัง อ่าน คำศัพท์ ไวยากรณ์) ที่เหมาะสมที่สุดสำหรับผู้เรียนแต่ละคน โดยมีเป้าหมายพัฒนาความสามารถให้สูงสุดภายในงบเวลาที่กำหนด

> **หมายเหตุสำคัญ:** ผู้เรียนทุกคนในระบบเป็นข้อมูลจำลอง (simulated learners) ไม่มีข้อมูลส่วนตัวจริงใด ๆ ถูกใช้

---

## สถาปัตยกรรมระบบ

```
DQN Agent (สมอง)
    ↓  ส่ง action (กิจกรรม)
EnglishLearningEnv (สภาพแวดล้อม)
    ↓  คืน state (11 มิติ) + reward
Streamlit Dashboard (หน้าจอ)
    ↓  แสดงผล charts, KPIs, timeline
```

### DQN Network Architecture

```
Input (11) → Linear(128) → ReLU → Linear(128) → ReLU → Output (4)
```

- **State (11 มิติ):** ความสามารถ 4 ทักษะ (normalized), ความคืบหน้า, เวลาคงเหลือ, เป้าหมาย, จำนวนครั้งที่เลือกแต่ละกิจกรรม
- **Actions (4):** Listening (15 นาที), Reading (20 นาที), Vocabulary (10 นาที), Grammar (25 นาที)
- **Seeds ที่ใช้:** 42, 43, 44, 45, 46

---

## โครงสร้างโฟลเดอร์

```
DQN_Streamlit/
├── app.py                              ← หน้าหลัก (Modern Analytics Home)
├── dqn_core.py                         ← DQN model + environment logic
├── shared_ui.py                        ← Design System, CSS, Sidebar, Plotly Theme
├── requirements.txt                    ← dependencies
├── README.md                           ← เอกสารโครงการ
│
├── data/
│   ├── simulated_learners.csv          ← ข้อมูลผู้เรียนจำลอง ~100k คน
│   ├── training_history_all_seeds.csv  ← ประวัติการ train ทั้ง 5 seeds
│   ├── evaluation_results_all_seeds.csv← ผลการประเมิน (DQN vs Baselines)
│   ├── evaluation_summary_by_seed.csv  ← สรุปผลแยกตาม seed
│   ├── statistical_results_all_seeds.csv← ผลการทดสอบทางสถิติ
│   ├── robustness_results_all_seeds.csv← ผลการทดสอบความทนทาน
│   └── robustness_summary_by_seed.csv  ← สรุปผล robustness
│
├── models/
│   ├── dqn_model_seed_42.pth           ← model weights (Seed 42)
│   ├── dqn_model_seed_43.pth           ← model weights (Seed 43)
│   ├── dqn_model_seed_44.pth           ← model weights (Seed 44)
│   ├── dqn_model_seed_45.pth           ← model weights (Seed 45)
│   └── dqn_model_seed_46.pth           ← model weights (Seed 46)
│
├── pages/
│   ├── 1_Learner_Simulation.py         ← หน้า 1: จำลองการเรียน (Inference)
│   ├── 2_Recommended_Learning_Path.py  ← หน้า 2: ลำดับกิจกรรมและ Q-values
│   ├── 3_Model_Evaluation.py           ← หน้า 3: ประเมินเปรียบเทียบโมเดล
│   └── 4_Training_Analytics.py         ← หน้า 4: วิเคราะห์กระบวนการ Train
│
└── .streamlit/
    └── config.toml                     ← ธีมและการตั้งค่า Streamlit
```

---

## การติดตั้งและเตรียมสภาพแวดล้อม

### ขั้นตอนที่ 1: ตรวจสอบ Python

ต้องการ **Python 3.11** ขึ้นไป

```powershell
python --version
```

### ขั้นตอนที่ 2: สร้าง Virtual Environment (แนะนำ)

```powershell
cd "c:\Users\warisara.ch\Desktop\DQN_Streamlit"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### ขั้นตอนที่ 3: ติดตั้ง Dependencies

```powershell
pip install -r requirements.txt
```

### ขั้นตอนที่ 4: รัน Streamlit

```powershell
streamlit run app.py
```

เปิด browser ที่ `http://localhost:8501`

---

## วิธีใช้งานแดชบอร์ด

### หน้า 1 — 👤 Learner Simulation

1. เลือก **Learner Profile** (เช่น Low_Vocab, Low_Grammar)
2. เลือก **Learner ID** จาก dropdown
3. เลือก **Training Seed** (42–46) เพื่อกำหนด DQN model ที่ใช้
4. กำหนด **Simulation Random Seed** เพื่อ reproducibility
5. คลิก **▶ Run DQN Simulation**
6. ดู KPI cards, charts, และตาราง episode

### หน้า 2 — 🗺️ Recommended Learning Path

- ต้องรัน simulation จาก หน้า 1 ก่อน
- ดู timeline ของกิจกรรมที่ DQN เลือก
- ดู Q-values แต่ละ step ผ่าน step selector
- ดู skill progression และ reward timeline

### หน้า 3 — 📊 Model Evaluation

- เลือก seeds และ agents ที่ต้องการเปรียบเทียบ
- ดูผลการเปรียบเทียบ DQN vs Random vs Weakest-Skill
- ดู robustness analysis (Normal, High Random Noise, Low Learning Rate)
- ดูผลทดสอบทางสถิติ (Wilcoxon, Cohen's D)

### หน้า 4 — 📈 Training Analytics

- ดู reward curve (raw และ moving average)
- ดู epsilon decay across seeds
- ดู avg ability gain during training
- เปรียบเทียบ stability ระหว่าง 5 seeds

---

## การตีความ Charts

### Q-Values คืออะไร?
Q-value คือ **ค่าประมาณ cumulative future reward** ที่คาดหวังจากการเลือก action นั้น ณ state ปัจจุบัน **ไม่ใช่** ความน่าจะเป็นของความสำเร็จ

### Reward Function
```
reward = ability_gain_total - (time_cost / time_budget) × 10
```

### Episode Termination
- เฉลี่ยความสามารถ ≥ learning_goal → สำเร็จ
- เวลาที่ใช้ ≥ time_budget → หมดเวลา

---

## Troubleshooting

| ปัญหา | แนวทางแก้ไข |
|-------|------------|
| `ModuleNotFoundError: dqn_core` | ต้องรัน `streamlit run app.py` จาก root directory |
| Model load error | ตรวจสอบว่าไฟล์ `.pth` อยู่ใน `models/` |
| CSV not found | ตรวจสอบว่าไฟล์ CSV อยู่ใน `data/` |
| Page 2 shows "no data" | ต้องรัน simulation จาก Page 1 ก่อน |
| Streamlit rerun loop | ลอง clear cache ด้วย `st.cache_data.clear()` |

---

## ข้อจำกัดของงานวิจัย

1. **ข้อมูลจำลอง:** ผู้เรียนทั้งหมดเป็น simulated learners ไม่มีการทดสอบกับผู้เรียนจริง
2. **Reward Function:** การออกแบบ reward function มีผลต่อพฤติกรรมของ agent อย่างมาก
3. **Environment Simplification:** สภาพแวดล้อมจำลองมีความซับซ้อนน้อยกว่าการเรียนรู้จริง
4. **Generalization:** ผลลัพธ์อาจไม่ generalize ได้ดีต่อผู้เรียนจริงโดยไม่มีการ fine-tuning
5. **Interactive Simulation:** ผลจากหน้า Simulation เป็นการจำลองใหม่ด้วย DQN inference จริง ซึ่งอาจต่างจากผล evaluation ใน CSV เนื่องจากใช้ random seed ต่างกัน
