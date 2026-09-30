import pandas as pd

data = {
    "question": [
        "How do I apply for hostel?",
        "When are the semester exams?",
        "How can I pay my fees?",
        "Where can I get my ID card?"
    ],
    "category": [
        "Hostel",
        "Examination",
        "Fees",
        "Administration"
    ]
}

df = pd.DataFrame(data)

print(df)