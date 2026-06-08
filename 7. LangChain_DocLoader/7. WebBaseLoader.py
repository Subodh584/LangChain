from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchin_core.messages import MessagesPlaceHolder



llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm)

chatPrompt = ChatPromptTemplate([
    ('system', "You are a utility agent, your task is to generate a summary of the data which is being provided to you by the user"),
    MessagesPlaceHolder()
    ('human', "here is the data: \n {scraped_data}")],
    input_variables = ['scraped_data']
)
loader = WebBaseLoader(web_paths=["https://www.amazon.in/Apple-iPhone-13-128GB-Starlight/dp/B09G9D8KRQ/ref=sr_1_1_sspa?crid=INQQFHL8Z9B7&dib=eyJ2IjoiMSJ9.gr7v5GqsATb_E3kzSc815tTI39nS_WyWWGeEqJWQhT5aInfFH-9LMVFkZRbafQWK7b1hFHxvHyyCYmsm2-oI6s3A7l3iLFmMT85SvxjuwXa8NXeWbdl5V_QJbE-D2_q1QH_7lEo52gAg4y50zhFLTF3M6xZ5jCn2sqavqfUyiK8yqEpqin1tU5KBeCS5KsDurics8Ht2KXJJkowdg9SeC6j3PR4dU3_gYoVFXz0cFfA.eOBdcWfcx3jThnr1hN65lf8pADN5GsZMefKfAVzgEHE&dib_tag=se&keywords=iphone&qid=1780568362&sprefix=ipho%2Caps%2C269&sr=8-1-spons&aref=tsHJk4ajXI&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"])
docs = loader.load()
print(len(docs))
print(docs[0].page_content)


