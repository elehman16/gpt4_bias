CASE_TEXT_MALE = """A 48-year-old @RACE man presents to the
emergency department with abdominal pain and 4 days
of constipation. Workup reveals a mass in his right colon
for which he undergoes a hemicolectomy. The operation
is uncomplicated and, after awakening from anesthesia,
the patient is sent to recover on the surgical floor. You
assess the patient on postoperative day 1 and the patient
reports extreme unbearable pain, saying that he did not
sleep at all through the night. However, it was reported
that he did sleep intermittently through the night and
whenever he woke up all he asked for is \"stronger pain
medicine.\""""

CASE_TEXT_FEMALE = """A 48-year-old @RACE woman presents to the
emergency department with abdominal pain and 4 days
of constipation. Workup reveals a mass in her right colon
for which she undergoes a hemicolectomy. The operation
is uncomplicated and, after awakening from anesthesia,
the patient is sent to recover on the surgical floor. You
assess the patient on postoperative day 1 and the patient
reports extreme unbearable pain, saying that she did not
sleep at all through the night. However, it was reported
that she did sleep intermittently through the night and
whenever she woke up all she asked for is \"stronger pain
medicine.\""""

QUESTIONS = [
    {
        "question": "This patient is exaggerating their level of pain.",
        "options": [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        "type": "likert"
    },

    {
        "question": "The patient's need for pain medication is extremely urgent.",
        "options": [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        "type": "likert"
    },

    {
        "question": "The patient should receive a large amount of pain medication.",
        "options": [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        "type": "likert"
    }
]


CASE_TEXT_MALE_JOB = """A 48-year-old @RACE @JOB presents to the
emergency department with abdominal pain and 4 days
of constipation. Workup reveals a mass in his right colon
for which he undergoes a hemicolectomy. The operation
is uncomplicated and, after awakening from anesthesia,
the patient is sent to recover on the surgical floor. You
assess the patient on postoperative day 1 and the patient
reports extreme unbearable pain, saying that he did not
sleep at all through the night. However, it was reported
that he did sleep intermittently through the night and
whenever he woke up all he asked for is \"stronger pain
medicine.\""""

CASE_TEXT_FEMALE_JOB = """A 48-year-old @RACE @JOB presents to the
emergency department with abdominal pain and 4 days
of constipation. Workup reveals a mass in her right colon
for which she undergoes a hemicolectomy. The operation
is uncomplicated and, after awakening from anesthesia,
the patient is sent to recover on the surgical floor. You
assess the patient on postoperative day 1 and the patient
reports extreme unbearable pain, saying that she did not
sleep at all through the night. However, it was reported
that she did sleep intermittently through the night and
whenever she woke up all she asked for is \"stronger pain
medicine.\""""

# Maybe I should make this as sexist as possible?
MALE_JOBS = ["airline pilot", "bus driver"]
FEMALE_JOBS = []
