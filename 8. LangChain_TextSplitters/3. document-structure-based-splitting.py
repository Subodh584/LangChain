# we use recursive-text-splitter only but we use separate separators

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language



text = '''class Student:
    def __init__(self, student_id, name, age):
        self.student_id = student_id
        self.name = name
        self.age = age

    def display_info(self):
        print(f"ID: {self.student_id}")
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")

    def celebrate_birthday(self):
        self.age += 1
        print(f"Happy Birthday, {self.name}! You are now {self.age} years old.")


# Create objects
student1 = Student(101, "Alice", 20)
student2 = Student(102, "Bob", 21)

# Use methods
student1.display_info()
print()

student2.display_info()
print()

student1.celebrate_birthday()'''


splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 500,
    chunk_overlap = 0
)



chunks = splitter.split_text(text)

print(chunks[1])



