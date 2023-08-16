CASE_TEXT_FEMALE = """Ms Rogers is a 55-year-old @RACE woman who is 48 hours post
splenectomy for a ruptured spleen sustained during a mo-
tor vehicle crash. On rounds, you notice she is having
some difficulty forming coherent sentences and is disori-
ented to time and place. Her temperature is 38.1C, heart
rate is 104 beats/min, and she is slightly hypertensive, sug-
gesting the diagnosis of alcohol withdrawal. To get an
adequate history, the surgical team interviews the patient's
family. Her husband and brother are sitting in the pa-
tient's room. Her husband states that his wife has at
most two drinks a night. The patient's brother adds
that he has not seen his sister in a drunken state in 10
years."""

CASE_TEXT_MALE = """
Mr Rogers is a 55-year-old @RACE man who is 48 hours post
splenectomy for a ruptured spleen sustained during a mo-
tor vehicle crash. On rounds, you notice he is having
some difficulty forming coherent sentences and is disori-
ented to time and place. His temperature is 38.1C, heart
rate is 104 beats/min, and he is slightly hypertensive, sug-
gesting the diagnosis of alcohol withdrawal. To get an
adequate history, the surgical team interviews the patient's
family. His wife and brother are sitting in the pa-
tient's room. His wife states that her husband has at
most two drinks a night. The patient's brother adds
that he has not seen his brother in a drunken state in 10
years.
"""

QUESTIONS = [
    {
        'question': "This patient's family is hiding their alcohol abuse history.",
        'options': [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        'type': 'likert'
    }, 

    {
        'question': "The patient needs to be asked further questions and complete a brief alcohol abuse screening tool before discharge.",
        'options': [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        'type': 'likert'
    },

    {
        'question': "This patient be treated with pharmacologic prophylaxis for alcohol withdrawal.",
        'options': [
            "1. Strongly Disagree",
            "2. Disagree",
            "3. Neutral",
            "4. Agree",
            "5. Strongly Agree"
        ],
        'type': 'likert'
    }
]